#!/usr/bin/env python
"""Frozen single-source-of-truth reporting script.

一次推理产出全部正文/表格需要的数字, 输出 CSV + JSON:
  - PLF TCR (3-seed ensemble)
  - PLF care-off (= OLP, 同一权重 future_treatment_mode='zero')
  - Transformer TCR (3-seed ensemble)
  - Persistence baseline

输出文件 (全部从同一推理):
  results/v4/frozen_main_table2.csv      (Table 2 主表)
  results/v4/frozen_supp_s3.csv          (Supplement S3 多时距轨迹)
  results/v4/frozen_result2_numbers.json  (Result 2 全部数字)
  results/v4/frozen_result3_numbers.json  (Result 3 care-off配对数字)

关键: TCR/care-off/Transformer/persistence 四者使用完全相同的
  test anchors / mask / dtype / aggregation / bootstrap units.
"""
from __future__ import annotations
import os, sys, json, csv
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
from v6.models.std_transformer import StdTransformer
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results/v4")
SEEDS = [42, 52, 62]
HORIZONS = [(1, 0), (3, 2), (6, 5), (12, 11)]
N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_plf(spec, seed, dev):
    ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(dev).eval()
    return m


def build_tr(seed, dev):
    ck = torch.load(REPO / f"runs/baselines/transformer_tcr_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = StdTransformer(prior_dim=14, mode="TCR")
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.to(dev).eval()
    return m


def run_plf_mode(loader, dev, spec, mode):
    """PLF 3-seed ensemble: logits (N,24,3) + organ_future (N,12,6)."""
    ens_logits = []; ens_organ = []
    for seed in SEEDS:
        m = build_plf(spec, seed, dev)
        ls = []; os_ = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b, stage="conditioned", future_treatment_mode=mode)
                ls.append(out["class_logits"].float().cpu().numpy())
                os_.append(out["organ_future"][:, :12, :].float().cpu().numpy())
        ens_logits.append(np.concatenate(ls))
        ens_organ.append(np.concatenate(os_))
        print(f"    PLF {mode} seed {seed}: done", flush=True)
        del m; torch.cuda.empty_cache()
    return np.mean(ens_logits, axis=0), np.mean(ens_organ, axis=0)


def run_tr(loader, dev):
    """Transformer 3-seed ensemble TCR: logits (N,24,3) + organ_future (N,12,6)."""
    ens_logits = []; ens_organ = []
    for seed in SEEDS:
        m = build_tr(seed, dev)
        ls = []; os_ = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b)
                ls.append(out["class_logits"].float().cpu().numpy())
                os_.append(out["organ_future"][:, :12, :].float().cpu().numpy())
        ens_logits.append(np.concatenate(ls))
        ens_organ.append(np.concatenate(os_))
        print(f"    TR seed {seed}: done", flush=True)
        del m; torch.cuda.empty_cache()
    return np.mean(ens_logits, axis=0), np.mean(ens_organ, axis=0)


def sofa_mae_per_anchor(pred_6, true_6, mask_6):
    """逐锚点 SOFA 总分绝对误差. (N,6)->(N,)"""
    return np.abs((pred_6 * mask_6).sum(axis=1) - (true_6 * mask_6).sum(axis=1))


def macro_mae_per_anchor(pred_6, true_6, mask_6):
    """逐锚点 macro MAE. (N,6)->(N,) 对有效器官取平均"""
    n_valid = mask_6.sum(axis=1)
    abs_err = np.abs(pred_6 - true_6) * mask_6
    return np.where(n_valid > 0, abs_err.sum(axis=1) / np.maximum(n_valid, 1), 0.0)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("FROZEN single-source-of-truth reporting", flush=True)
    print("=" * 70, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    # 标签
    all_organ, all_mask, all_delta, all_cmask, all_stays = [], [], [], [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
        all_delta.append(b["delta_sofa"].numpy()); all_cmask.append(b["class_mask"].numpy())
        all_stays.append(b["stay_id"].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    delta = np.concatenate(all_delta); cmask = np.concatenate(all_cmask)
    stays = np.concatenate(all_stays)

    # 推理: 4 个条件
    print("=== PLF TCR ===", flush=True)
    plf_tcr_logits, plf_tcr_organ = run_plf_mode(loader, dev, spec, "actual")
    print("\n=== PLF care-off (OLP/zero) ===", flush=True)
    plf_co_logits, plf_co_organ = run_plf_mode(loader, dev, spec, "zero")
    print("\n=== Transformer TCR ===", flush=True)
    tr_logits, tr_organ = run_tr(loader, dev)

    # ================================================================
    # Table 2 主表: 6h trajectory (changed-state分层) + discrimination
    # ================================================================
    print("\n=== 生成 Table 2 ===", flush=True)
    H6 = 5
    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid_traj = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa_6h = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)

    # trajectory per-anchor
    persist_sofa = sofa_mae_per_anchor(o_now, o_6h, m_6h)
    persist_macro = macro_mae_per_anchor(o_now, o_6h, m_6h)
    plf_sofa = sofa_mae_per_anchor(plf_tcr_organ[:, H6, :], o_6h, m_6h)
    plf_macro = macro_mae_per_anchor(plf_tcr_organ[:, H6, :], o_6h, m_6h)
    tr_sofa = sofa_mae_per_anchor(tr_organ[:, H6, :], o_6h, m_6h)
    tr_macro = macro_mae_per_anchor(tr_organ[:, H6, :], o_6h, m_6h)

    subsets = [
        ("Overall", valid_traj),
        ("ΔSOFA=0", valid_traj & (delta_sofa_6h == 0)),
        ("|ΔSOFA|≥1", valid_traj & (np.abs(delta_sofa_6h) >= 1)),
        ("ΔSOFA≥2", valid_traj & (delta_sofa_6h >= 2)),
    ]
    table2_traj = []
    for label, mask in subsets:
        table2_traj.append({
            "subset": label, "n": int(mask.sum()),
            "persistence_sofa": float(persist_sofa[mask].mean()),
            "transformer_sofa": float(tr_sofa[mask].mean()),
            "plf_ogt_sofa": float(plf_sofa[mask].mean()),
        })
    # discrimination (6h)
    m6_cls = cmask[:, H6] > 0
    y6 = (delta[:, H6] >= 2).astype(float)
    plf_tcr_p = softmax_np(plf_tcr_logits[:, H6, :])[:, 0]
    plf_co_p = softmax_np(plf_co_logits[:, H6, :])[:, 0]
    tr_p = softmax_np(tr_logits[:, H6, :])[:, 0]

    table2_disc = {
        "n_valid": int(m6_cls.sum()),
        "plf_tcr_auroc": float(roc_auc_score(y6[m6_cls], plf_tcr_p[m6_cls])),
        "plf_tcr_auprc": float(average_precision_score(y6[m6_cls], plf_tcr_p[m6_cls])),
        "plf_co_auroc": float(roc_auc_score(y6[m6_cls], plf_co_p[m6_cls])),
        "plf_co_auprc": float(average_precision_score(y6[m6_cls], plf_co_p[m6_cls])),
        "tr_auroc": float(roc_auc_score(y6[m6_cls], tr_p[m6_cls])),
        "tr_auprc": float(average_precision_score(y6[m6_cls], tr_p[m6_cls])),
    }

    # ================================================================
    # Result 3: care-off 配对 (6h, 同一推理)
    # ================================================================
    print("=== 生成 Result 3 numbers ===", flush=True)
    # trajectory care-off (用同一 PLF 的 care-off organ)
    plf_co_sofa = sofa_mae_per_anchor(plf_co_organ[:, H6, :], o_6h, m_6h)
    plf_co_macro = macro_mae_per_anchor(plf_co_organ[:, H6, :], o_6h, m_6h)

    # bootstrap CI for care-off ΔAUPRC/ΔAUROC
    sv = stays[m6_cls]
    idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
    n_st = len(idx_map); rng = np.random.RandomState(BOOT_SEED)
    d_auc_b, d_ap_b, d_sofa_b, d_macro_b = [], [], [], []
    yv = y6[m6_cls]; ptv = plf_tcr_p[m6_cls]; pov = plf_co_p[m6_cls]
    sofa_t = plf_sofa[valid_traj]; sofa_c = plf_co_sofa[valid_traj]
    sv_traj = stays[valid_traj]
    idx_traj = {s: np.where(sv_traj == s)[0] for s in np.unique(sv_traj)}
    n_st_traj = len(idx_traj)
    for _ in range(N_BOOT):
        s_d = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        idx = np.concatenate([idx_map[s] for s in s_d])
        yy = yv[idx]
        if len(set(yy)) < 2: continue
        try:
            d_auc_b.append(roc_auc_score(yy, ptv[idx]) - roc_auc_score(yy, pov[idx]))
            d_ap_b.append(average_precision_score(yy, ptv[idx]) - average_precision_score(yy, pov[idx]))
        except: continue
        # trajectory bootstrap
        s_t = rng.choice(list(idx_traj.keys()), size=n_st_traj, replace=True)
        idx_t = np.concatenate([idx_traj[s] for s in s_t])
        d_sofa_b.append(sofa_t[idx_t].mean() - sofa_c[idx_t].mean())
        d_macro_b.append(plf_macro[valid_traj][idx_t].mean() - plf_co_macro[valid_traj][idx_t].mean())

    a = 0.025
    result3 = {
        "plf_tcr_sofa_6h": float(plf_sofa[valid_traj].mean()),
        "plf_co_sofa_6h": float(plf_co_sofa[valid_traj].mean()),
        "delta_sofa": float(plf_sofa[valid_traj].mean() - plf_co_sofa[valid_traj].mean()),
        "delta_sofa_ci": [float(np.percentile(d_sofa_b, a*100)), float(np.percentile(d_sofa_b, (1-a)*100))],
        "plf_tcr_macro_6h": float(plf_macro[valid_traj].mean()),
        "plf_co_macro_6h": float(plf_co_macro[valid_traj].mean()),
        "delta_macro": float(plf_macro[valid_traj].mean() - plf_co_macro[valid_traj].mean()),
        "delta_macro_ci": [float(np.percentile(d_macro_b, a*100)), float(np.percentile(d_macro_b, (1-a)*100))],
        "delta_auprc": float(average_precision_score(yv, ptv) - average_precision_score(yv, pov)),
        "delta_auprc_ci": [float(np.percentile(d_ap_b, a*100)), float(np.percentile(d_ap_b, (1-a)*100))],
        "delta_auroc": float(roc_auc_score(yv, ptv) - roc_auc_score(yv, pov)),
        "delta_auroc_ci": [float(np.percentile(d_auc_b, a*100)), float(np.percentile(d_auc_b, (1-a)*100))],
        "valid_traj_n": int(valid_traj.sum()), "valid_cls_n": int(m6_cls.sum()),
    }

    # ================================================================
    # Supplement S3: 多时距轨迹完整
    # ================================================================
    print("=== 生成 Supplement S3 ===", flush=True)
    s3_rows = []
    for h, hi in HORIZONS:
        o_h = organ[:, hi + 1, :]; m_h = omask[:, hi + 1, :]
        v_h = (m_now * m_h).sum(axis=1) > 0
        if v_h.sum() == 0: continue
        plf_h = plf_tcr_organ[:, hi, :]; co_h = plf_co_organ[:, hi, :]
        plf_s = sofa_mae_per_anchor(plf_h, o_h, m_h)
        co_s = sofa_mae_per_anchor(co_h, o_h, m_h)
        plf_m = macro_mae_per_anchor(plf_h, o_h, m_h)
        co_m = macro_mae_per_anchor(co_h, o_h, m_h)
        s3_rows.append({
            "horizon": h, "valid_n": int(v_h.sum()),
            "plf_tcr_sofa": float(plf_s[v_h].mean()),
            "plf_tcr_macro": float(plf_m[v_h].mean()),
            "plf_co_sofa": float(co_s[v_h].mean()),
            "plf_co_macro": float(co_m[v_h].mean()),
            "delta_sofa": float(plf_s[v_h].mean() - co_s[v_h].mean()),
            "delta_macro": float(plf_m[v_h].mean() - co_m[v_h].mean()),
        })

    # ================================================================
    # 输出
    # ================================================================
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Table 2 CSV
    with open(OUTDIR / "frozen_main_table2.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["6h trajectory", "n", "Persistence SOFA MAE", "Transformer SOFA MAE", "PLF-OGT SOFA MAE"])
        for r in table2_traj:
            w.writerow([r["subset"], r["n"], f"{r['persistence_sofa']:.3f}", f"{r['transformer_sofa']:.3f}", f"{r['plf_ogt_sofa']:.3f}"])
        w.writerow([])
        w.writerow(["6h deterioration", "n", "AUROC", "AUPRC"])
        w.writerow(["PLF-OGT (TCR)", table2_disc["n_valid"], f"{table2_disc['plf_tcr_auroc']:.3f}", f"{table2_disc['plf_tcr_auprc']:.3f}"])
        w.writerow(["PLF-OGT (care-off)", table2_disc["n_valid"], f"{table2_disc['plf_co_auroc']:.3f}", f"{table2_disc['plf_co_auprc']:.3f}"])
        w.writerow(["Transformer (TCR)", table2_disc["n_valid"], f"{table2_disc['tr_auroc']:.3f}", f"{table2_disc['tr_auprc']:.3f}"])

    # S3 CSV
    with open(OUTDIR / "frozen_supp_s3.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Horizon", "Valid n", "PLF TCR sofa MAE", "PLF TCR macro MAE", "care-off sofa MAE", "care-off macro MAE", "Δ sofa", "Δ macro"])
        for r in s3_rows:
            w.writerow([f"{r['horizon']}h", r["valid_n"], f"{r['plf_tcr_sofa']:.3f}", f"{r['plf_tcr_macro']:.3f}",
                        f"{r['plf_co_sofa']:.3f}", f"{r['plf_co_macro']:.3f}", f"{r['delta_sofa']:.3f}", f"{r['delta_macro']:.3f}"])

    # JSON
    json.dump({"table2_trajectory": table2_traj, "table2_discrimination": table2_disc},
              open(OUTDIR / "frozen_result2_numbers.json", "w"), indent=2, ensure_ascii=False, default=float)
    json.dump(result3, open(OUTDIR / "frozen_result3_numbers.json", "w"), indent=2, ensure_ascii=False, default=float)
    json.dump(s3_rows, open(OUTDIR / "frozen_supp_s3.json", "w"), indent=2, ensure_ascii=False, default=float)

    # 打印汇总
    print(f"\n{'='*70}", flush=True)
    print("FROZEN OUTPUT SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print("\nTable 2 trajectory (6h):")
    for r in table2_traj:
        print(f"  {r['subset']:<16} n={r['n']:<8} persist={r['persistence_sofa']:.3f} TR={r['transformer_sofa']:.3f} PLF={r['plf_ogt_sofa']:.3f}")
    print(f"\nTable 2 discrimination (6h, n={table2_disc['n_valid']}):")
    print(f"  PLF TCR: AUROC={table2_disc['plf_tcr_auroc']:.3f} AUPRC={table2_disc['plf_tcr_auprc']:.3f}")
    print(f"  PLF co:  AUROC={table2_disc['plf_co_auroc']:.3f} AUPRC={table2_disc['plf_co_auprc']:.3f}")
    print(f"  TR TCR:  AUROC={table2_disc['tr_auroc']:.3f} AUPRC={table2_disc['tr_auprc']:.3f}")
    print(f"\nResult 3 (care-off配对, traj n={result3['valid_traj_n']}, cls n={result3['valid_cls_n']}):")
    print(f"  sofa: TCR={result3['plf_tcr_sofa_6h']:.3f} co={result3['plf_co_sofa_6h']:.3f} Δ={result3['delta_sofa']:+.3f} (CI {result3['delta_sofa_ci'][0]:+.3f},{result3['delta_sofa_ci'][1]:+.3f})")
    print(f"  macro: TCR={result3['plf_tcr_macro_6h']:.3f} co={result3['plf_co_macro_6h']:.3f} Δ={result3['delta_macro']:+.3f}")
    print(f"  AUPRC Δ={result3['delta_auprc']:+.3f} (CI {result3['delta_auprc_ci'][0]:+.3f},{result3['delta_auprc_ci'][1]:+.3f})")
    print(f"  AUROC Δ={result3['delta_auroc']:+.3f} (CI {result3['delta_auroc_ci'][0]:+.3f},{result3['delta_auroc_ci'][1]:+.3f})")
    print(f"\nS3 multi-horizon:")
    for r in s3_rows:
        print(f"  {r['horizon']}h: PLF={r['plf_tcr_sofa']:.3f} co={r['plf_co_sofa']:.3f} Δ={r['delta_sofa']:+.3f}")
    print(f"\n输出: {OUTDIR}/frozen_*.csv / *.json", flush=True)


if __name__ == "__main__":
    main()
