#!/usr/bin/env python
"""Patient-specific R negative controls v2: stay-level shuffle + bootstrap CI.

修正 v1 的两个问题:
  1. shuffle 改为 stay-level (不是 batch-level)
  2. 补 ICU-stay 级配对 cluster bootstrap CI

方法:
  Step 1: 对每个 seed, 编码全部测试集锚点的 R (6,640)
  Step 2: stay-level shuffle: 随机重排 stay, 每个锚点用映射后 stay 的 R
  Step 3: 用 matched/shuffled/mean/query_only 的 R rollout, 收集 prob + MAE
  Step 4: 配对 bootstrap CI (seed 42, 与前面分析同口径)

输出: results/v4/frozen_patient_specific_r_v2.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results/v4")
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_model(spec, seed, dev):
    ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(dev).eval()
    return m


def encode_all_R(model, loader, dev):
    """编码全部锚点的 R (N,6,128) + 收集 stay_id."""
    all_R = []; all_stays = []
    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            branch = model.encode_branches(batch)
            all_R.append(branch.residual.cpu())  # (B,6,128)
            all_stays.append(batch["stay_id"].cpu())
    return torch.cat(all_R, dim=0), torch.cat(all_stays, dim=0)


def rollout_with_R(model, loader, dev, R_override):
    """用指定的 R 跑 rollout. R_override: (N,6,128) tensor or None."""
    all_p = []; all_y = []
    organ_preds = []; organ_labels = []; organ_masks = []
    idx_offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            B = batch["organ"].shape[0]
            branch = model.encode_branches(batch)
            if R_override is not None:
                branch.residual = R_override[idx_offset:idx_offset+B].to(dev)
            out = model.rollout_from_state(batch, branch, stage="conditioned",
                                            future_treatment_mode="actual")
            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:, 0])
            organ_preds.append(out["organ_future"][:, H6, :].float().cpu().numpy())
            organ = batch["organ"].cpu().numpy(); omask = batch["organ_mask"].cpu().numpy()
            organ_labels.append(organ[:, H6+1]); organ_masks.append(omask[:, H6+1])
            o_now = organ[:, 0]; m_now = omask[:, 0]
            o_6h = organ[:, H6+1]; m_6h = omask[:, H6+1]
            delta = ((m_now*m_6h)*(o_6h-o_now)).sum(axis=1)
            all_y.append((delta >= 2).astype(float))
            idx_offset += B
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    op = np.concatenate(organ_preds); ol = np.concatenate(organ_labels); om = np.concatenate(organ_masks)
    return p, y, op, ol, om


def stay_level_shuffle_R(R_all, stays, rng):
    """Stay-level shuffle: 重排 stay, 每个锚点用映射后 stay 的代表性 R.
    保证不会换回同一 stay.
    """
    unique_stays = torch.unique(stays)
    n_stays = len(unique_stays)
    # 生成 stay 映射 (derangement: 不允许映射到自身)
    perm = torch.randperm(n_stays, generator=rng)
    # 确保 derangement
    identity = torch.arange(n_stays)
    while (perm == identity).any():
        perm = torch.randperm(n_stays, generator=rng)
    stay_map = {unique_stays[i].item(): unique_stays[perm[i]].item() for i in range(n_stays)}
    # 对每个 stay, 取其所有锚点 R 的均值作为"代表性 R"
    stay_repr = {}
    for s in unique_stays:
        mask = stays == s
        stay_repr[s.item()] = R_all[mask].mean(dim=0)  # (6,128)
    # 替换: 每个锚点的 R -> 映射后 stay 的代表性 R
    R_shuffled = torch.zeros_like(R_all)
    for i in range(len(stays)):
        target_stay = stay_map[stays[i].item()]
        R_shuffled[i] = stay_repr[target_stay]
    return R_shuffled


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("Patient-specific R v2: stay-level shuffle + bootstrap CI", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    # 对 seed 42 做完整分析 (含 bootstrap), 对 seed 52/62 只做点估计汇总
    CONDITIONS = ["matched", "shuffled", "mean", "query_only"]
    per_seed = {}
    seed42_data = {}  # 存 seed42 的逐锚点数据用于 bootstrap

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model = build_model(spec, seed, dev)

        # Step 1: 编码全部 R
        print("  编码全部 R...", flush=True)
        R_all, stays = encode_all_R(model, loader, dev)
        print(f"  R shape: {R_all.shape}, stays: {stays.shape}", flush=True)

        rng = torch.Generator(device="cpu").manual_seed(seed * 100)

        seed_results = {}
        for cond in CONDITIONS:
            if cond == "matched":
                R_override = None
            elif cond == "shuffled":
                R_override = stay_level_shuffle_R(R_all, stays, rng)
            elif cond == "mean":
                R_override = R_all.mean(dim=0, keepdim=True).expand_as(R_all)
            elif cond == "query_only":
                R_override = torch.zeros_like(R_all)

            p, y, op, ol, om = rollout_with_R(model, loader, dev, R_override)
            mask = np.ones(len(y), dtype=bool)
            auc = float(roc_auc_score(y[mask], p[mask])) if y[mask].sum() > 5 else float("nan")
            ap = float(average_precision_score(y[mask], p[mask])) if y[mask].sum() > 5 else float("nan")
            organ_maes = [float(np.abs(op[om[:,o]>0,o]-ol[om[:,o]>0,o]).mean()) if (om[:,o]>0).sum()>0 else float("nan") for o in range(6)]
            macro = float(np.nanmean(organ_maes))
            seed_results[cond] = {"auprc": ap, "auroc": auc, "macro_mae": macro}
            print(f"  {cond:<14} AUPRC={ap:.4f} AUROC={auc:.4f} macroMAE={macro:.4f}", flush=True)

            if seed == 42:
                seed42_data[cond] = {"p": p, "y": y, "op": op, "ol": ol, "om": om, "stays": stays.numpy()}

        per_seed[f"seed_{seed}"] = seed_results
        del model; torch.cuda.empty_cache()

    # Step 2: Bootstrap CI (seed 42 的逐锚点数据)
    print(f"\n=== Bootstrap CI (seed 42, n_boot={N_BOOT}) ===", flush=True)
    m_data = seed42_data["matched"]
    stays_arr = m_data["stays"]
    idx_map = {s: np.where(stays_arr == s)[0] for s in np.unique(stays_arr)}
    n_st = len(idx_map)
    rng_boot = np.random.RandomState(BOOT_SEED)

    bootstrap_results = {}
    for cond in ["shuffled", "mean", "query_only"]:
        c_data = seed42_data[cond]
        d_ap_b, d_mae_b = [], []
        for _ in range(N_BOOT):
            sampled = rng_boot.choice(list(idx_map.keys()), size=n_st, replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            # AUPRC: matched[idx] vs cond[idx] — 分别算 AUPRC 再差? 还是直接比较 prob?
            # 直接比较: 在同一 bootstrap sample 内算两者的 AUPRC 差
            yy = m_data["y"][idx]
            if len(set(yy)) < 2 or yy.sum() < 3: continue
            try:
                ap_m = average_precision_score(yy, m_data["p"][idx])
                ap_c = average_precision_score(yy, c_data["p"][idx])
                d_ap_b.append(ap_m - ap_c)
                # MAE: 逐锚点 SOFA MAE
                mae_m = np.abs(m_data["op"][idx]*m_data["om"][idx] - m_data["ol"][idx]*m_data["om"][idx]).sum() / m_data["om"][idx].sum()
                mae_c = np.abs(c_data["op"][idx]*c_data["om"][idx] - c_data["ol"][idx]*c_data["om"][idx]).sum() / c_data["om"][idx].sum()
                d_mae_b.append(mae_c - mae_m)  # 正 = cond 更差
            except: continue
        a = 0.025
        d_ap_pt = per_seed["seed_42"][cond]["auprc"]
        m_ap_pt = per_seed["seed_42"]["matched"]["auprc"]
        d_ap = m_ap_pt - d_ap_pt  # matched - cond (正 = matched更好)
        d_mae_pt = per_seed["seed_42"][cond]["macro_mae"] - per_seed["seed_42"]["matched"]["macro_mae"]

        bootstrap_results[cond] = {
            "delta_auprc_point": float(d_ap),
            "delta_auprc_ci": [float(np.percentile(d_ap_b, a*100)), float(np.percentile(d_ap_b, (1-a)*100))],
            "delta_mae_ci": [float(np.percentile(d_mae_b, a*100)), float(np.percentile(d_mae_b, (1-a)*100))],
            "n_boot": len(d_ap_b),
        }
        r = bootstrap_results[cond]
        print(f"  {cond:<14} ΔAUPRC={r['delta_auprc_point']:+.3f} (CI {r['delta_auprc_ci'][0]:+.3f},{r['delta_auprc_ci'][1]:+.3f})  "
              f"ΔMAE CI ({r['delta_mae_ci'][0]:+.3f},{r['delta_mae_ci'][1]:+.3f})", flush=True)

    # 汇总
    print(f"\n{'='*70}", flush=True)
    print(f"{'条件':<14}{'3-seed AUPRC':<14}{'3-seed MAE':<12}{'ΔAUPRC':<12}{'ΔMAE':<10}")
    print("-" * 70)
    final = {}
    matched_ap = np.mean([per_seed[f"seed_{s}"]["matched"]["auprc"] for s in SEEDS])
    matched_mae = np.mean([per_seed[f"seed_{s}"]["matched"]["macro_mae"] for s in SEEDS])
    for cond in CONDITIONS:
        ap = float(np.mean([per_seed[f"seed_{s}"][cond]["auprc"] for s in SEEDS]))
        ap_std = float(np.std([per_seed[f"seed_{s}"][cond]["auprc"] for s in SEEDS]))
        mae = float(np.mean([per_seed[f"seed_{s}"][cond]["macro_mae"] for s in SEEDS]))
        d_ap = ap - matched_ap; d_mae = mae - matched_mae
        ci_info = bootstrap_results.get(cond, {})
        final[cond] = {"auprc_mean": ap, "auprc_std": ap_std, "macro_mae_mean": mae,
                       "delta_auprc": d_ap, "delta_mae": d_mae,
                       "bootstrap_ci": ci_info}
        print(f"{cond:<14}{ap:<14.4f}{mae:<12.4f}{d_ap:<+12.4f}{d_mae:<+10.4f}", flush=True)
    print("=" * 70, flush=True)

    out_data = {"per_seed": per_seed, "summary": final, "bootstrap_seed42": bootstrap_results,
                "note": "Stay-level shuffle (derangement). Bootstrap on seed42 predictions."}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(OUTDIR / "frozen_patient_specific_r_v2.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n保存: {OUTDIR / 'frozen_patient_specific_r_v2.json'}", flush=True)


if __name__ == "__main__":
    main()
