#!/usr/bin/env python
"""MIMIC-IV patient-specificity + bootstrap CI.

复用已生成的逐锚点预测 (seed 42), 补 ICU-stay 级 cluster bootstrap CI.
方法与 GMUICU 版 (eval_patient_specific_r_v2.py) 完全一致:
  - seed 42 的逐锚点 prob + label
  - stay-level cluster bootstrap, n_boot=2000
  - 在每个 bootstrap sample 内算 matched vs cond 的 AUPRC 差

输出: results_mimic/frozen_mimic_patient_specific.json (覆盖, 加 CI)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.mimic_dataset import MIMICDataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results_mimic")
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_model(spec, seed, dev):
    ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(dev).eval()
    return m


def encode_all_R(model, loader, dev):
    all_R = []; all_stays = []
    with torch.inference_mode():
        for b in loader:
            b = move(b, dev)
            branch = model.encode_branches(b)
            all_R.append(branch.residual.cpu())
            all_stays.append(b["stay_id"].cpu())
    return torch.cat(all_R, dim=0), torch.cat(all_stays, dim=0)


def stay_level_shuffle_R(R_all, stays, rng):
    unique_stays = torch.unique(stays); n = len(unique_stays)
    perm = torch.randperm(n, generator=rng)
    identity = torch.arange(n)
    while (perm == identity).any():
        perm = torch.randperm(n, generator=rng)
    stay_map = {unique_stays[i].item(): unique_stays[perm[i]].item() for i in range(n)}
    stay_repr = {}
    for s in unique_stays:
        mask = stays == s
        stay_repr[s.item()] = R_all[mask].mean(dim=0)
    R_shuffled = torch.zeros_like(R_all)
    for i in range(len(stays)):
        R_shuffled[i] = stay_repr[stay_map[stays[i].item()]]
    return R_shuffled


def rollout_with_R(model, loader, dev, R_override):
    """跑 rollout, 收集 prob, label, organ pred/label/mask, stays."""
    all_p = []; all_y = []; all_stays = []
    organ_preds = []; organ_labels = []; organ_masks = []
    idx = 0
    with torch.inference_mode():
        for b in loader:
            b = move(b, dev); B = b["organ"].shape[0]
            branch = model.encode_branches(b)
            if R_override is not None:
                branch.residual = R_override[idx:idx+B].to(dev)
            out = model.rollout_from_state(b, branch, stage="conditioned", future_treatment_mode="actual")
            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:, 0])
            organ_preds.append(out["organ_future"][:, H6, :].float().cpu().numpy())
            organ = b["organ"].cpu().numpy(); omask = b["organ_mask"].cpu().numpy()
            organ_labels.append(organ[:, H6+1]); organ_masks.append(omask[:, H6+1])
            o_now = organ[:, 0]; m_now = omask[:, 0]
            o_6h = organ[:, H6+1]; m_6h = omask[:, H6+1]
            delta = ((m_now*m_6h)*(o_6h-o_now)).sum(axis=1)
            all_y.append((delta >= 2).astype(float))
            all_stays.append(b["stay_id"].cpu().numpy())
            idx += B
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    op = np.concatenate(organ_preds); ol = np.concatenate(organ_labels); om = np.concatenate(organ_masks)
    stays = np.concatenate(all_stays)
    return p, y, op, ol, om, stays


def main():
    configure_cuda(); dev = DEVICE
    print("="*70, flush=True)
    print("MIMIC-IV patient-specificity + bootstrap CI (4条件×3seed)", flush=True)
    print("="*70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = MIMICDataset(split="test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"MIMIC test: {len(ds)}\n", flush=True)

    CONDITIONS = ["matched", "shuffled", "mean", "query_only"]
    per_seed = {}
    seed42_data = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model = build_model(spec, seed, dev)
        print("  编码全部 R...", flush=True)
        R_all, stays_t = encode_all_R(model, loader, dev)
        print(f"  R shape: {R_all.shape}, stays: {stays_t.shape}", flush=True)

        rng = torch.Generator(device="cpu").manual_seed(seed * 100)
        seed_results = {}
        for cond in CONDITIONS:
            if cond == "matched":
                R_override = None
            elif cond == "shuffled":
                R_override = stay_level_shuffle_R(R_all, stays_t, rng)
            elif cond == "mean":
                R_override = R_all.mean(dim=0, keepdim=True).expand_as(R_all)
            elif cond == "query_only":
                R_override = torch.zeros_like(R_all)

            p, y, op, ol, om, stays_arr = rollout_with_R(model, loader, dev, R_override)
            mask = np.ones(len(y), dtype=bool)
            auc = float(roc_auc_score(y[mask], p[mask])) if y[mask].sum() > 5 else float("nan")
            ap = float(average_precision_score(y[mask], p[mask])) if y[mask].sum() > 5 else float("nan")
            organ_maes = [float(np.abs(op[om[:,o]>0,o]-ol[om[:,o]>0,o]).mean()) if (om[:,o]>0).sum()>0 else float("nan") for o in range(6)]
            macro = float(np.nanmean(organ_maes))
            seed_results[cond] = {"auprc": ap, "auroc": auc, "macro_mae": macro}
            print(f"  {cond:<14} AUPRC={ap:.4f} AUROC={auc:.4f} macroMAE={macro:.4f}", flush=True)

            if seed == 42:
                seed42_data[cond] = {"p": p, "y": y, "op": op, "ol": ol, "om": om, "stays": stays_arr}

        per_seed[f"seed_{seed}"] = seed_results
        del model; torch.cuda.empty_cache()

    # Bootstrap CI (seed 42)
    print(f"\n=== Bootstrap CI (seed 42, n_boot={N_BOOT}) ===", flush=True)
    m_data = seed42_data["matched"]
    stays_arr = m_data["stays"]
    unique_stays = np.unique(stays_arr)
    idx_map = {s: np.where(stays_arr == s)[0] for s in unique_stays}
    n_st = len(idx_map)
    rng_boot = np.random.RandomState(BOOT_SEED)

    bootstrap_results = {}
    for cond in ["shuffled", "mean", "query_only"]:
        c_data = seed42_data[cond]
        d_ap_b, d_mae_b = [], []
        for _ in range(N_BOOT):
            sampled = rng_boot.choice(unique_stays, size=n_st, replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            yy = m_data["y"][idx]
            if len(set(yy)) < 2 or yy.sum() < 3:
                continue
            try:
                ap_m = average_precision_score(yy, m_data["p"][idx])
                ap_c = average_precision_score(yy, c_data["p"][idx])
                d_ap_b.append(ap_m - ap_c)
                mae_m = np.abs(m_data["op"][idx]*m_data["om"][idx] - m_data["ol"][idx]*m_data["om"][idx]).sum() / m_data["om"][idx].sum()
                mae_c = np.abs(c_data["op"][idx]*c_data["om"][idx] - c_data["ol"][idx]*c_data["om"][idx]).sum() / c_data["om"][idx].sum()
                d_mae_b.append(mae_c - mae_m)
            except Exception:
                continue
        a = 0.025
        d_ap_pt = per_seed["seed_42"]["matched"]["auprc"] - per_seed["seed_42"][cond]["auprc"]
        bootstrap_results[cond] = {
            "delta_auprc_point": float(d_ap_pt),
            "delta_auprc_ci": [float(np.percentile(d_ap_b, a*100)), float(np.percentile(d_ap_b, (1-a)*100))],
            "delta_mae_ci": [float(np.percentile(d_mae_b, a*100)), float(np.percentile(d_mae_b, (1-a)*100))],
            "n_boot": len(d_ap_b),
        }
        r = bootstrap_results[cond]
        print(f"  {cond:<14} ΔAUPRC={r['delta_auprc_point']:+.3f} (CI {r['delta_auprc_ci'][0]:+.3f},{r['delta_auprc_ci'][1]:+.3f})  "
              f"ΔMAE CI ({r['delta_mae_ci'][0]:+.3f},{r['delta_mae_ci'][1]:+.3f})", flush=True)

    # 汇总 (3-seed mean ± std + bootstrap CI from seed42)
    print(f"\n{'='*70}", flush=True)
    print(f"{'条件':<14}{'3-seed AUPRC':<16}{'3-seed AUROC':<14}{'3-seed MAE':<12}{'ΔAUPRC':<12}")
    print("-"*70)
    final = {}
    matched_ap = np.mean([per_seed[f"seed_{s}"]["matched"]["auprc"] for s in SEEDS])
    matched_auc = np.mean([per_seed[f"seed_{s}"]["matched"]["auroc"] for s in SEEDS])
    matched_mae = np.mean([per_seed[f"seed_{s}"]["matched"]["macro_mae"] for s in SEEDS])
    for cond in CONDITIONS:
        ap = float(np.mean([per_seed[f"seed_{s}"][cond]["auprc"] for s in SEEDS]))
        ap_std = float(np.std([per_seed[f"seed_{s}"][cond]["auprc"] for s in SEEDS]))
        auc = float(np.mean([per_seed[f"seed_{s}"][cond]["auroc"] for s in SEEDS]))
        auc_std = float(np.std([per_seed[f"seed_{s}"][cond]["auroc"] for s in SEEDS]))
        mae = float(np.mean([per_seed[f"seed_{s}"][cond]["macro_mae"] for s in SEEDS]))
        d_ap = ap - matched_ap
        ci_info = bootstrap_results.get(cond, {})
        final[cond] = {"auprc_mean": ap, "auprc_std": ap_std,
                       "auroc_mean": auc, "auroc_std": auc_std,
                       "macro_mae_mean": mae,
                       "delta_auprc": d_ap,
                       "bootstrap_ci": ci_info}
        print(f"{cond:<14}{ap:<10.4f}±{ap_std:<5.4f}{auc:<10.4f}±{auc_std:<5.4f}{mae:<12.4f}{d_ap:<+12.4f}", flush=True)
    print("="*70, flush=True)

    out_data = {"per_seed": per_seed, "summary": final, "bootstrap_seed42": bootstrap_results,
                "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                "note": "MIMIC-IV. Stay-level shuffle (derangement). ICU-stay cluster bootstrap on seed42."}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(OUTDIR / "frozen_mimic_patient_specific.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n保存: {OUTDIR / 'frozen_mimic_patient_specific.json'}", flush=True)


if __name__ == "__main__":
    main()
