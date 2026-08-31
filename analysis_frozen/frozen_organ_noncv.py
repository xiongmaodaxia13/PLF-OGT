#!/usr/bin/env python
"""Frozen Result 3 organ decomposition + non-CV + treatment strata.

从同一 frozen 推理产出:
  - 逐器官 6h MAE (TCR vs care-off, delta, CV share)
  - non-CV SOFA 分类 (ΔAUROC, ΔBrier)
  - 治疗升级/降阶/稳定 ΔP

输出: results/v4/frozen_organ_noncv.json + frozen_s11.csv
"""
from __future__ import annotations
import os, sys, json, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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
ORGAN_NAMES = ["Resp", "Cv", "Renal", "Coag", "Hepatic", "CNS"]


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


def run_plf_mode(loader, dev, spec, mode):
    """PLF 3-seed ensemble: logits (N,24,3) + organ_future (N,12,6)."""
    ens_logits, ens_organ = [], []
    for seed in SEEDS:
        m = build_plf(spec, seed, dev)
        ls, os_ = [], []
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


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("Frozen organ decomposition + non-CV + treatment strata", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    # 标签
    all_organ, all_mask, all_delta, all_cmask, all_stays = [], [], [], [], []
    all_hist_on, all_fut_on = [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
        all_delta.append(b["delta_sofa"].numpy()); all_cmask.append(b["class_mask"].numpy())
        all_stays.append(b["stay_id"].numpy())
        all_hist_on.append(b["hist_act_on"].numpy())
        all_fut_on.append(b["future_act_on"][:, H6, :].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    delta = np.concatenate(all_delta); cmask = np.concatenate(all_cmask)
    stays = np.concatenate(all_stays)
    hist_on = np.concatenate(all_hist_on); fut_on = np.concatenate(all_fut_on)

    # 推理
    print("=== PLF TCR ===", flush=True)
    tcr_logits, tcr_organ = run_plf_mode(loader, dev, spec, "actual")
    print("\n=== PLF care-off ===", flush=True)
    co_logits, co_organ = run_plf_mode(loader, dev, spec, "zero")

    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid_traj = (m_now * m_6h).sum(axis=1) > 0
    m6_cls = cmask[:, H6] > 0
    y6_total = (delta[:, H6] >= 2).astype(float)  # 总SOFA恶化

    # ================================================================
    # 1. 逐器官 6h MAE (TCR vs care-off)
    # ================================================================
    print("\n=== 逐器官 MAE ===", flush=True)
    organ_results = {}
    total_delta_sum = 0
    for o in range(6):
        m = m_6h[:, o] > 0
        if m.sum() == 0: continue
        tcr_mae = float(np.abs(tcr_organ[valid_traj & m, H6, o] - o_6h[valid_traj & m, o]).mean())
        # care-off organ
        co_mae = float(np.abs(co_organ[valid_traj & m, H6, o] - o_6h[valid_traj & m, o]).mean())
        d = co_mae - tcr_mae  # care-off 更差 = 正 delta
        organ_results[ORGAN_NAMES[o]] = {"tcr_mae": tcr_mae, "co_mae": co_mae, "delta": d}
        total_delta_sum += d
        print(f"  {ORGAN_NAMES[o]:<10} TCR={tcr_mae:.3f}  co={co_mae:.3f}  Δ={d:+.3f}", flush=True)

    cv_delta = organ_results["Cv"]["delta"]
    cv_share = cv_delta / total_delta_sum if total_delta_sum > 0 else 0
    print(f"\n  CV share: {cv_share*100:.1f}% (Δ_CV={cv_delta:.3f} / Σ={total_delta_sum:.3f})", flush=True)

    # ================================================================
    # 2. non-CV SOFA 分类
    # ================================================================
    print("\n=== non-CV SOFA 分类 ===", flush=True)
    # non-CV ΔSOFA: 排除 CV(idx=1)
    mask_ncv = np.ones(6, dtype=bool); mask_ncv[1] = False
    delta_ncv = ((m_now * m_6h)[:, mask_ncv] * (o_6h[:, mask_ncv] - o_now[:, mask_ncv])).sum(axis=1)
    y_ncv = (delta_ncv >= 2).astype(float)
    valid_ncv = (m_now * m_6h)[:, mask_ncv].sum(axis=1) > 0

    tcr_p = softmax_np(tcr_logits[:, H6, :])[:, 0]
    co_p = softmax_np(co_logits[:, H6, :])[:, 0]

    v = valid_ncv & m6_cls
    if y_ncv[v].sum() > 5 and len(set(y_ncv[v])) > 1:
        auc_tcr = float(roc_auc_score(y_ncv[v], tcr_p[v]))
        auc_co = float(roc_auc_score(y_ncv[v], co_p[v]))
        brier_tcr = float(brier_score_loss(y_ncv[v], np.clip(tcr_p[v], 1e-8, 1-1e-8)))
        brier_co = float(brier_score_loss(y_ncv[v], np.clip(co_p[v], 1e-8, 1-1e-8)))
        noncv = {"n_valid": int(v.sum()), "prev": float(y_ncv[v].mean()),
                 "tcr_auroc": auc_tcr, "co_auroc": auc_co, "delta_auroc": auc_tcr - auc_co,
                 "tcr_brier": brier_tcr, "co_brier": brier_co, "delta_brier": brier_co - brier_tcr}
        print(f"  n={v.sum()} prev={y_ncv[v].mean()*100:.2f}%", flush=True)
        print(f"  ΔAUROC={auc_tcr-auc_co:+.4f} ΔBrier={brier_co-brier_tcr:+.4f}", flush=True)
    else:
        noncv = {"note": "insufficient events"}
        print("  insufficient events", flush=True)

    # total SOFA for comparison
    auc_t_total = float(roc_auc_score(y6_total[m6_cls], tcr_p[m6_cls]))
    auc_c_total = float(roc_auc_score(y6_total[m6_cls], co_p[m6_cls]))
    total_cls = {"delta_auroc": auc_t_total - auc_c_total}

    # ================================================================
    # 3. 治疗升级/降阶/稳定 ΔP
    # ================================================================
    print("\n=== 治疗分层 ===", flush=True)
    # 治疗变化: hist_on (B,23) vs fut_on (B,23)
    n_hist = hist_on.sum(axis=1)  # 锚点时在治数量
    n_fut = fut_on.sum(axis=1)    # 6h时在治数量
    escalation = (n_fut > n_hist) & m6_cls
    deescalation = (n_fut < n_hist) & m6_cls
    stable_treat = (n_fut == n_hist) & m6_cls

    strata = {}
    for name, mask in [("escalation", escalation), ("deescalation", deescalation), ("stable", stable_treat)]:
        if mask.sum() > 0:
            tcr_mean = float(tcr_p[mask].mean())
            co_mean = float(co_p[mask].mean())
            strata[name] = {"n": int(mask.sum()), "tcr_p": tcr_mean, "co_p": co_mean, "delta": tcr_mean - co_mean}
            print(f"  {name}: n={mask.sum()} ΔP={tcr_mean-co_mean:+.3f}", flush=True)

    # ================================================================
    # 输出
    # ================================================================
    results = {
        "organ_mae": organ_results,
        "cv_share": float(cv_share),
        "total_delta_sum": float(total_delta_sum),
        "non_cv": noncv,
        "total_cls": total_cls,
        "treatment_strata": strata,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTDIR / "frozen_organ_noncv.json", "w"), indent=2, ensure_ascii=False, default=float)

    # S11 CSV
    with open(OUTDIR / "frozen_s11.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Organ", "TCR MAE", "care-off MAE", "ΔMAE (co-TCR)", "Share of total Δ"])
        for o in range(6):
            name = ORGAN_NAMES[o]
            if name in organ_results:
                r = organ_results[name]
                share = r["delta"] / total_delta_sum if total_delta_sum > 0 else 0
                w.writerow([name, f"{r['tcr_mae']:.3f}", f"{r['co_mae']:.3f}", f"{r['delta']:+.3f}", f"{share*100:.1f}%"])

    print(f"\n{'='*70}", flush=True)
    print("FROZEN ORGAN/NON-CV SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nCV share of trajectory Δ: {cv_share*100:.1f}%")
    print(f"non-CV ΔAUROC: {noncv.get('delta_auroc', 'N/A')}")
    print(f"total ΔAUROC: {total_cls['delta_auroc']:+.3f}")
    print(f"\n输出: {OUTDIR}/frozen_organ_noncv.json + frozen_s11.csv", flush=True)


if __name__ == "__main__":
    main()
