#!/usr/bin/env python
"""OLP→TCR 配对 cluster bootstrap 差值 CI (全时距) —— 第二档 (#8 部分).

复用 results/v4/allhorizon_ci_logits.npz 缓存的 3-seed ensemble logits,
对 1/3/6/12h 做 OLP→TCR 的配对患者层聚类 bootstrap 差值 CI.
与 frozen_ot_paired.json (仅6h) 同口径, 扩展到全时距.

注: 这是"同一 PLF-OGT 模型 OLP→TCR"的差值 CI;
陆老师 #8 表7/8 要的 "V4 vs Transformer @ TCR" 模型间差值另需 Transformer 推理.

输出: results/v4/ot_paired_allhorizon.json

用法: python scripts/eval_v4_ot_paired_allhorizon.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/v4"
HORIZONS = [1, 3, 6, 12]
N_BOOT = 2000
BOOT_SEED = 42


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    cache = OUT_DIR / "allhorizon_ci_logits.npz"
    if not cache.exists():
        print(f"!! 缓存不存在: {cache}. 请先跑 eval_v4_allhorizon_ci.py", flush=True)
        return
    c = np.load(cache)
    p_tcr = softmax_np(c["tcr_logits"] if "tcr_logits" in c else c["tcr"])  # 兼容
    p_olp = softmax_np(c["olp_logits"] if "olp_logits" in c else c["olp"])
    delta = c["delta"]; mask = c["mask"]; stays = c["stays"]
    # 注: 缓存存的是 softmax 后的 prob (allhorizon_ci.py 存的是 ens_p). 检查
    # allhorizon_ci.py: np.savez(..., tcr=ens_p_tcr, olp=ens_p_olp, ...) -> 已是 prob
    p_tcr = c["tcr"]; p_olp = c["olp"]

    print("=" * 60, flush=True)
    print("第二档 #8(部分): OLP→TCR 配对差值 CI (全时距)", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for h in HORIZONS:
        hi = h - 1
        pt = p_tcr[:, hi, 0]; po = p_olp[:, hi, 0]
        m = mask[:, hi] > 0
        y = (delta[:, hi] >= 2).astype(float)
        if m.sum() == 0 or y[m].sum() <= 5:
            continue
        # 点估计
        auc_t = roc_auc_score(y[m], pt[m]); auc_o = roc_auc_score(y[m], po[m])
        ap_t = average_precision_score(y[m], pt[m]); ap_o = average_precision_score(y[m], po[m])
        # 配对 cluster bootstrap
        idx_map = {s: np.where(stays[m] == s)[0] for s in np.unique(stays[m])}
        n_st = len(idx_map)
        rng = np.random.RandomState(BOOT_SEED)
        d_auc, d_ap = [], []
        for _ in range(N_BOOT):
            sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            yy = y[m][idx]
            if len(set(yy)) < 2:
                continue
            try:
                d_auc.append(roc_auc_score(yy, pt[m][idx]) - roc_auc_score(yy, po[m][idx]))
                d_ap.append(average_precision_score(yy, pt[m][idx]) - average_precision_score(yy, po[m][idx]))
            except Exception:
                continue
        a = 0.025
        results[f"{h}h"] = {
            "tcr_auroc": float(auc_t), "olp_auroc": float(auc_o),
            "delta_auroc": float(auc_t - auc_o),
            "delta_auroc_lo": float(np.percentile(d_auc, a * 100)),
            "delta_auroc_hi": float(np.percentile(d_auc, (1 - a) * 100)),
            "tcr_auprc": float(ap_t), "olp_auprc": float(ap_o),
            "delta_auprc": float(ap_t - ap_o),
            "delta_auprc_lo": float(np.percentile(d_ap, a * 100)),
            "delta_auprc_hi": float(np.percentile(d_ap, (1 - a) * 100)),
            "n_valid": int(m.sum()), "n_clusters": n_st, "n_boot": len(d_auc),
        }
        r = results[f"{h}h"]
        print(f"  {h:>2}h: ΔAUROC {r['delta_auroc']:+.3f} ({r['delta_auroc_lo']:+.3f},{r['delta_auroc_hi']:+.3f})  "
              f"ΔAUPRC {r['delta_auprc']:+.3f} ({r['delta_auprc_lo']:+.3f},{r['delta_auprc_hi']:+.3f})", flush=True)

    out = OUT_DIR / "ot_paired_allhorizon.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
