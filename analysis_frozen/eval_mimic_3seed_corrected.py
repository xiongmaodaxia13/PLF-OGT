#!/usr/bin/env python
"""MIMIC-IV 3-seed 多时距判别 + care-off 配对 CI (修正时距索引bug).

修复 eval_v4_mimic_3seed.py 的 h_idx bug: 用 h-1 而非 enumerate序号.
一次推理同时产出:
  1. TCR/OLP 多时距 AUROC/AUPRC + cluster bootstrap CI
  2. care-off (OLP/carry) 配对 ΔAUPRC/ΔAUROC + CI (3-seed 集成)

标签口径: delta_sofa[:,h-1], class_mask[:,h-1], h∈{1,3,6,12}
  6h 真实值: n=42623, prev=4.34% (非之前错误报的3.06%)

输出: results_mimic/mimic_3seed_corrected.json
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
OUT = Path(r"F:/MIMIC3_1/V13/results_mimic/mimic_3seed_corrected.json")
SEEDS = [42, 52, 62]
HORIZONS = [(1, 0), (3, 2), (6, 5), (12, 11)]  # (小时, 列索引 h-1)
N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("MIMIC 3-seed 多时距判别 + care-off CI (修正时距索引)", flush=True)
    print("=" * 70, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = MIMICDataset(split="test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"MIMIC test: {len(ds)}\n", flush=True)

    # 标签 + stays
    all_delta, all_mask, all_stays = [], [], []
    for b in loader:
        all_delta.append(b["delta_sofa"].numpy())
        all_mask.append(b["class_mask"].numpy())
        all_stays.append(b["stay_id"].numpy())
    delta = np.concatenate(all_delta); mask = np.concatenate(all_mask); stays = np.concatenate(all_stays)

    # 3-seed ensemble: TCR + OLP(care-off)
    for mode_name, mode in [("TCR", "actual"), ("OLP", "zero")]:
        print(f"\n=== {mode_name} 3-seed ensemble ===", flush=True)
        ens_logits = []
        for seed in SEEDS:
            ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
            model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                                  event_layers=2, concept_layers=1, residual_layers=1,
                                  transition_layers=2, dropout=0.0, transition_mode="modulation",
                                  n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
            model.load_state_dict(ck["model_state_dict"], strict=False)
            model.to(dev).eval()
            sl = []
            with torch.inference_mode():
                for b in loader:
                    b = move(b, dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = model(b, stage="conditioned", future_treatment_mode=mode)
                    sl.append(out["class_logits"].float().cpu().numpy())
            ens_logits.append(np.concatenate(sl))
            print(f"  {mode_name} seed {seed}: done", flush=True)
            del model; torch.cuda.empty_cache()
        p_ens = softmax_np(np.mean(ens_logits, axis=0))  # (N, 24, 3) 注意: logits 是 24 步!
        if mode_name == "TCR":
            p_tcr = p_ens
        else:
            p_olp = p_ens

    # 多时距判别 + CI (修正 h_idx)
    print("\n=== 多时距判别 (修正索引) ===", flush=True)
    results = {"TCR": {}, "OLP": {}}
    for h, hi in HORIZONS:
        m = mask[:, hi] > 0
        y = (delta[:, hi] >= 2).astype(float)
        n = int(m.sum()); prev = float(y[m].mean())
        if n == 0 or y[m].sum() <= 5:
            continue
        for mode_name, p_ens in [("TCR", p_tcr), ("OLP", p_olp)]:
            p = p_ens[:, hi, 0]
            auc = float(roc_auc_score(y[m], p[m])); ap = float(average_precision_score(y[m], p[m]))
            # cluster bootstrap CI
            sv = stays[m]; idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
            n_st = len(idx_map); rng = np.random.RandomState(BOOT_SEED)
            auc_b, ap_b = [], []
            yv, pv = y[m], p[m]
            for _ in range(N_BOOT):
                sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
                idx = np.concatenate([idx_map[s] for s in sampled])
                yy = yv[idx]
                if len(set(yy)) < 2: continue
                try:
                    auc_b.append(roc_auc_score(yy, pv[idx])); ap_b.append(average_precision_score(yy, pv[idx]))
                except Exception: continue
            a = 0.025
            results[mode_name][f"{h}h"] = {
                "auroc": auc, "auprc": ap, "n": n, "prev": prev,
                "auroc_ci": [float(np.percentile(auc_b, a*100)), float(np.percentile(auc_b, (1-a)*100))],
                "auprc_ci": [float(np.percentile(ap_b, a*100)), float(np.percentile(ap_b, (1-a)*100))],
                "n_clusters": n_st}
        tr = results["TCR"][f"{h}h"]; ol = results["OLP"][f"{h}h"]
        print(f"  {h:>2}h: TCR AUROC={tr['auroc']:.3f}({tr['auroc_ci'][0]:.3f}-{tr['auroc_ci'][1]:.3f}) "
              f"AUPRC={tr['auprc']:.3f} OLP AUROC={ol['auroc']:.3f} | n={n} prev={prev*100:.2f}%", flush=True)

    # care-off 配对 Δ CI (3-seed 集成, 修正索引)
    print("\n=== care-off 配对 CI (3-seed 集成) ===", flush=True)
    careoff_results = {}
    for h, hi in HORIZONS:
        m = mask[:, hi] > 0; y = (delta[:, hi] >= 2).astype(float)
        if y[m].sum() <= 5: continue
        pt = p_tcr[:, hi, 0]; po = p_olp[:, hi, 0]
        sv = stays[m]; idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
        n_st = len(idx_map); rng = np.random.RandomState(BOOT_SEED)
        yv = y[m]; ptv = pt[m]; pov = po[m]
        d_auc, d_ap = [], []
        for _ in range(N_BOOT):
            sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            yy = yv[idx]
            if len(set(yy)) < 2: continue
            try:
                d_auc.append(roc_auc_score(yy, ptv[idx]) - roc_auc_score(yy, pov[idx]))
                d_ap.append(average_precision_score(yy, ptv[idx]) - average_precision_score(yy, pov[idx]))
            except Exception: continue
        a = 0.025
        careoff_results[f"{h}h"] = {
            "delta_auroc": float(roc_auc_score(yv, ptv) - roc_auc_score(yv, pov)),
            "delta_auroc_ci": [float(np.percentile(d_auc, a*100)), float(np.percentile(d_auc, (1-a)*100))],
            "delta_auprc": float(average_precision_score(yv, ptv) - average_precision_score(yv, pov)),
            "delta_auprc_ci": [float(np.percentile(d_ap, a*100)), float(np.percentile(d_ap, (1-a)*100))],
            "n_boot": len(d_auc), "n_clusters": n_st}
        r = careoff_results[f"{h}h"]
        print(f"  {h:>2}h: ΔAUROC {r['delta_auroc']:+.3f}({r['delta_auroc_ci'][0]:.3f},{r['delta_auroc_ci'][1]:.3f}) "
              f"ΔAUPRC {r['delta_auprc']:+.3f}({r['delta_auprc_ci'][0]:.3f},{r['delta_auprc_ci'][1]:.3f})", flush=True)

    out_data = {"discrimination": results, "careoff_paired": careoff_results,
                "note": "修正时距索引bug: h_idx=h-1 (非enumerate序号). 6h真实prev=4.34%."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
