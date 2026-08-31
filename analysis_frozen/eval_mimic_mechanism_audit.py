#!/usr/bin/env python
"""MIMIC-IV 机制审计: S/R 2×2 + non-CV + care-off ΔAUPRC CI (3-seed).

一次推理跑完三项 MIMIC 缺失审计, 与 GMUICU 同口径:
  1. S/R 2×2 通路阻断 (block_s_treat/block_r_treat, 3-seed 中位数+IQR)
  2. non-CV SOFA 分类 (排除心血管分量后的 ΔAUROC/ΔBrier)
  3. care-off 配对 ΔAUPRC/ΔAUROC + cluster bootstrap CI

输出: results_mimic/mimic_mechanism_audit.json
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results_mimic/mimic_mechanism_audit.json")
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_model(spec, seed, dev):
    ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                          event_layers=2, concept_layers=1, residual_layers=1,
                          transition_layers=2, dropout=0.0, transition_mode="modulation",
                          n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(dev).eval()
    return model


def run_condition(model, loader, dev, mode, block_s=False, block_r=False):
    """跑一个条件 (TCR/care-off/S-blocked/R-blocked), 返回 worsen prob + organ_future + 标签."""
    model.eval()
    all_p, all_y, all_stays = [], [], []
    organ_preds, organ_labels, organ_masks = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            kw = dict(stage="conditioned", future_treatment_mode=mode)
            if block_s or block_r:
                kw["block_s_treat"] = block_s
                kw["block_r_treat"] = block_r
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(batch, **kw)
            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:, 0])
            organ_preds.append(out["organ_future"][:, H6, :].float().cpu().numpy())
            organ = batch["organ"].cpu().numpy()
            omask = batch["organ_mask"].cpu().numpy()
            organ_labels.append(organ[:, H6 + 1, :])
            organ_masks.append(omask[:, H6 + 1, :])
            # 标签 (总 SOFA)
            o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
            o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
            valid = m_now * m_6h
            delta = (valid * (o_6h - o_now)).sum(axis=1)
            all_y.append((delta >= 2).astype(float))
            all_stays.append(batch["stay_id"].cpu().numpy())
    return (np.concatenate(all_p), np.concatenate(all_y), np.concatenate(all_stays),
            np.concatenate(organ_preds), np.concatenate(organ_labels), np.concatenate(organ_masks))


def calc_noncv(y_total, p_tcr, p_careoff, organ_labels, organ_masks, cv_idx=1):
    """non-CV SOFA 标签: 排除心血管(idx=1)后的 ΔSOFA >= 2."""
    mask_ncv = np.ones(6, dtype=bool); mask_ncv[cv_idx] = False
    delta_ncv = ((organ_masks[:, mask_ncv]) * (organ_labels[:, mask_ncv] - organ_labels[:, mask_ncv])).sum(axis=1)
    # 实际 non-CV delta 要用 锚点 vs 6h
    # 这里 organ_labels 已是 6h 值, 需锚点值 — 从 run_condition 已返回的 labels 是 6h
    # 需要锚点值, 简化: 用 y_total 的 non-CV 版本在 run_condition 外重算
    return None  # 在 main 里用完整数据重算


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("MIMIC-IV 机制审计: S/R 2×2 + non-CV + care-off CI (3-seed)", flush=True)
    print("=" * 70, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = MIMICDataset(split="test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"MIMIC test: {len(ds)} samples\n", flush=True)

    CONDITIONS = [("SR", False, False, "actual"), ("N", True, True, "zero"),
                  ("S", False, True, "actual"), ("R", True, False, "actual")]

    # 先收集锚点 organ (用于 non-CV)
    all_organ_now = []
    for batch in loader:
        all_organ_now.append(batch["organ"][:, 0, :].cpu().numpy())
    organ_anchor = np.concatenate(all_organ_now)

    per_seed_2x2 = {}
    tcr_all = {}; careoff_all = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model = build_model(spec, seed, dev)

        seed_2x2 = {}
        for name, bs, br, mode in CONDITIONS:
            p, y, stays, op, ol, om = run_condition(model, loader, dev, mode, bs, br)
            # 用 care-off (zero) 的标签作为统一标签
            if name == "SR":
                tcr_all = {"p": p, "y": y, "stays": stays, "op": op, "ol": ol, "om": om}
            if name == "N":
                careoff_all = {"p": p, "y": y, "stays": stays, "op": op, "ol": ol, "om": om}
            mask = y > -1  # all valid
            auc = float(roc_auc_score(y[mask], p[mask])) if len(set(y[mask])) > 1 else float("nan")
            ap = float(average_precision_score(y[mask], p[mask])) if y[mask].sum() > 0 else float("nan")
            brier = float(brier_score_loss(y[mask], np.clip(p[mask], 1e-8, 1 - 1e-8)))
            # organ macro MAE (care-off 口径的 organ 不用于 SR/S, 只算总)
            organ_maes = []
            for o in range(6):
                m = om[:, o] > 0
                if m.sum() > 0:
                    organ_maes.append(float(np.abs(op[m, o] - ol[m, o]).mean()))
            macro = float(np.mean(organ_maes)) if organ_maes else float("nan")
            seed_2x2[name] = {"auprc": ap, "auroc": auc, "brier": brier, "macro_mae": macro}
            print(f"  {name}: AUPRC={ap:.4f} AUROC={auc:.4f} Brier={brier:.4f} macroMAE={macro:.4f}", flush=True)

        # Shapley
        U_b = {k: -seed_2x2[k]["brier"] for k in ["SR", "S", "R", "N"]}
        U_m = {k: -seed_2x2[k]["macro_mae"] for k in ["SR", "S", "R", "N"]}
        phi_sb = 0.5 * ((U_b["S"] - U_b["N"]) + (U_b["SR"] - U_b["R"]))
        phi_rb = 0.5 * ((U_b["R"] - U_b["N"]) + (U_b["SR"] - U_b["S"]))
        phi_sm = 0.5 * ((U_m["S"] - U_m["N"]) + (U_m["SR"] - U_m["R"]))
        phi_rm = 0.5 * ((U_m["R"] - U_m["N"]) + (U_m["SR"] - U_m["S"]))
        seed_2x2["shapley"] = {
            "phi_S_brier": phi_sb, "phi_R_brier": phi_rb, "ratio_brier": phi_rb / max(abs(phi_sb), 1e-8),
            "phi_S_mae": phi_sm, "phi_R_mae": phi_rm, "ratio_mae": phi_rm / max(abs(phi_sm), 1e-8)}
        print(f"  φ_R/φ_S Brier={seed_2x2['shapley']['ratio_brier']:.2f}x MAE={seed_2x2['shapley']['ratio_mae']:.2f}x", flush=True)
        per_seed_2x2[f"seed_{seed}"] = seed_2x2
        del model; torch.cuda.empty_cache()

    # === care-off ΔAUPRC/ΔAUROC + CI (用 seed42 的 TCR vs care-off) ===
    print("\n=== care-off 配对 CI ===", flush=True)
    p_tcr, y_tcr, st_tcr = tcr_all["p"], tcr_all["y"], tcr_all["stays"]
    p_co, y_co, st_co = careoff_all["p"], careoff_all["y"], careoff_all["stays"]
    # care-off AUPRC/AUROC
    auc_tcr = roc_auc_score(y_tcr, p_tcr); ap_tcr = average_precision_score(y_tcr, p_tcr)
    auc_co = roc_auc_score(y_co, p_co); ap_co = average_precision_score(y_co, p_co)
    # bootstrap CI
    idx_map = {s: np.where(st_tcr == s)[0] for s in np.unique(st_tcr)}
    n_st = len(idx_map); rng = np.random.RandomState(BOOT_SEED)
    d_auc_b, d_ap_b = [], []
    for _ in range(N_BOOT):
        sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        yy = y_tcr[idx]
        if len(set(yy)) < 2: continue
        try:
            d_auc_b.append(roc_auc_score(yy, p_tcr[idx]) - roc_auc_score(yy, p_co[idx]))
            d_ap_b.append(average_precision_score(yy, p_tcr[idx]) - average_precision_score(yy, p_co[idx]))
        except Exception: continue
    a = 0.025
    careoff_ci = {
        "tcr_auprc": float(ap_tcr), "careoff_auprc": float(ap_co),
        "delta_auprc": float(ap_tcr - ap_co),
        "delta_auprc_lo": float(np.percentile(d_ap_b, a * 100)),
        "delta_auprc_hi": float(np.percentile(d_ap_b, (1 - a) * 100)),
        "tcr_auroc": float(auc_tcr), "careoff_auroc": float(auc_co),
        "delta_auroc": float(auc_tcr - auc_co),
        "delta_auroc_lo": float(np.percentile(d_auc_b, a * 100)),
        "delta_auroc_hi": float(np.percentile(d_auc_b, (1 - a) * 100)),
        "n_boot": len(d_auc_b), "n_clusters": n_st}
    print(f"  ΔAUPRC {careoff_ci['delta_auprc']:+.3f} ({careoff_ci['delta_auprc_lo']:.3f}-{careoff_ci['delta_auprc_hi']:.3f})", flush=True)
    print(f"  ΔAUROC {careoff_ci['delta_auroc']:+.3f} ({careoff_ci['delta_auroc_lo']:.3f}-{careoff_ci['delta_auroc_hi']:.3f})", flush=True)

    # === non-CV SOFA ===
    print("\n=== non-CV SOFA 分类 ===", flush=True)
    organ_6h = tcr_all["ol"]; omask_6h = tcr_all["om"]
    mask_ncv = np.ones(6, dtype=bool); mask_ncv[1] = False  # 排除 CV (idx=1)
    delta_ncv = ((omask_6h[:, mask_ncv]) * (organ_6h[:, mask_ncv] - organ_anchor[:, mask_ncv])).sum(axis=1)
    y_ncv = (delta_ncv >= 2).astype(float)
    valid_ncv = omask_6h[:, mask_ncv].sum(axis=1) > 0
    if y_ncv[valid_ncv].sum() > 5 and len(set(y_ncv[valid_ncv])) > 1:
        auc_ncv_tcr = float(roc_auc_score(y_ncv[valid_ncv], p_tcr[valid_ncv]))
        auc_ncv_co = float(roc_auc_score(y_ncv[valid_ncv], p_co[valid_ncv]))
        brier_ncv_tcr = float(brier_score_loss(y_ncv[valid_ncv], np.clip(p_tcr[valid_ncv], 1e-8, 1-1e-8)))
        brier_ncv_co = float(brier_score_loss(y_ncv[valid_ncv], np.clip(p_co[valid_ncv], 1e-8, 1-1e-8)))
        noncv = {"n_valid": int(valid_ncv.sum()), "prev": float(y_ncv[valid_ncv].mean()),
                 "delta_auroc": float(auc_ncv_tcr - auc_ncv_co),
                 "delta_brier": float(brier_ncv_co - brier_ncv_tcr)}
        print(f"  non-CV ΔAUROC {noncv['delta_auroc']:+.4f} ΔBrier {noncv['delta_brier']:+.4f}", flush=True)
    else:
        noncv = {"note": "non-CV 事件不足"}
        print("  non-CV 事件不足", flush=True)

    # === 汇总 2×2 ===
    sr_auprcs = [per_seed_2x2[f"seed_{s}"]["SR"]["auprc"] for s in SEEDS]
    s_auprcs = [per_seed_2x2[f"seed_{s}"]["S"]["auprc"] for s in SEEDS]
    r_auprcs = [per_seed_2x2[f"seed_{s}"]["R"]["auprc"] for s in SEEDS]
    n_auprcs = [per_seed_2x2[f"seed_{s}"]["N"]["auprc"] for s in SEEDS]
    ratios_b = [per_seed_2x2[f"seed_{s}"]["shapley"]["ratio_brier"] for s in SEEDS]
    ratios_m = [per_seed_2x2[f"seed_{s}"]["shapley"]["ratio_mae"] for s in SEEDS]
    summary_2x2 = {
        "SR_median": float(np.median(sr_auprcs)), "SR_iqr": [float(np.percentile(sr_auprcs, 25)), float(np.percentile(sr_auprcs, 75))],
        "S_median": float(np.median(s_auprcs)), "R_median": float(np.median(r_auprcs)), "N_median": float(np.median(n_auprcs)),
        "ratio_brier_median": float(np.median(ratios_b)), "ratio_mae_median": float(np.median(ratios_m)),
    }
    print(f"\n2×2 中位数: SR={summary_2x2['SR_median']:.3f} S={summary_2x2['S_median']:.3f} R={summary_2x2['R_median']:.3f} N={summary_2x2['N_median']:.3f}", flush=True)

    results = {"per_seed": per_seed_2x2, "summary_2x2": summary_2x2,
               "careoff_ci": careoff_ci, "non_cv": noncv}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
