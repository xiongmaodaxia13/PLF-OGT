#!/usr/bin/env python
"""V4 vs Transformer @ TCR 模型间配对差值 CI —— 第二档 (#8 表7/表8 缺口).

陆老师 #8: 表7/表8 (TCR AUROC/AUPRC) 对 TCR 的模型间差值补配对 Bootstrap CI.
本脚本: Transformer TCR 3-seed 推理 + 与已缓存的 V4 TCR 3-seed logits 配对 cluster bootstrap.

复用: results/v4/allhorizon_ci_logits.npz 的 V4 TCR prob (字段 'tcr').
新增: Transformer TCR 3-seed 推理 (缓存 transformertcr_logits.npz).
输出: results/v4/v4_vs_tr_tcr_ci.json

用法: python scripts/eval_v4_vs_tr_tcr_ci.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 1=RTX 4090 D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.std_transformer import StdTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
HORIZONS = [(1, 0), (3, 2), (6, 5), (12, 11)]
SEEDS = [42, 52, 62]
N_BOOT = 2000
BOOT_SEED = 42
OUT_DIR = RESULTS_DIR / "v4"


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True)); return e / e.sum(axis=axis, keepdims=True)


def run_transformer_tcr(loader, dev):
    """Transformer TCR 3-seed ensemble prob (N,12,3)."""
    ens = []
    for seed in SEEDS:
        ckpt = REPO / f"runs/baselines/transformer_tcr_s{seed}/best.pt"
        sd = torch.load(ckpt, map_location=dev, weights_only=False)
        model = StdTransformer(prior_dim=14, mode="TCR")
        model.load_state_dict(sd.get("model_state_dict", sd))
        model.to(dev).eval()
        sl = []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                out = model(batch)
                sl.append(out["class_logits"].float().cpu().numpy())
        ens.append(np.concatenate(sl))
        print(f"    TR-TCR seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    return softmax_np(np.mean(ens, axis=0))


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第二档 #8(表7/8): V4 vs Transformer @ TCR 配对差值 CI", flush=True)
    print("=" * 60, flush=True)

    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)} samples", flush=True)

    # 标签
    all_delta, all_mask, all_st = [], [], []
    for batch in loader:
        all_delta.append(batch["delta_sofa"].numpy())
        all_mask.append(batch["class_mask"].numpy())
        all_st.append(batch["stay_id"].numpy())
    delta = np.concatenate(all_delta); mask = np.concatenate(all_mask); stays = np.concatenate(all_st)

    # V4 TCR prob (复用缓存)
    v4_cache = OUT_DIR / "allhorizon_ci_logits.npz"
    if not v4_cache.exists():
        print(f"!! V4 缓存不存在: {v4_cache}", flush=True); return
    v4_tcr_p = np.load(v4_cache)["tcr"]  # (N,12,3) prob

    # Transformer TCR prob (新推理, 缓存)
    tr_cache = OUT_DIR / "transformertcr_logits.npz"
    if tr_cache.exists():
        print(f"\n加载 Transformer 缓存: {tr_cache}", flush=True)
        tr_p = np.load(tr_cache)["tr_p"]
    else:
        print("\n=== Transformer TCR 3-seed 推理 ===", flush=True)
        tr_p = run_transformer_tcr(loader, dev)
        np.savez(tr_cache, tr_p=tr_p)
        print(f"缓存已存: {tr_cache}", flush=True)

    # 配对 cluster bootstrap 差值 CI (V4 - Transformer), 全时距
    results = {}
    print("\n=== V4 vs Transformer @ TCR 差值 CI ===", flush=True)
    rng = np.random.RandomState(BOOT_SEED)

    for h, hi in HORIZONS:
        m = mask[:, hi] > 0
        y = (delta[:, hi] >= 2).astype(float)
        if m.sum() == 0 or y[m].sum() <= 5:
            continue
        pv = v4_tcr_p[:, hi, 0]; pt = tr_p[:, hi, 0]
        # 筛选后的有效样本 + 对应 stays
        yv = y[m]; pvv = pv[m]; ptv = pt[m]; sv = stays[m]
        # 点估计
        auc_v = roc_auc_score(yv, pvv); auc_t = roc_auc_score(yv, ptv)
        ap_v = average_precision_score(yv, pvv); ap_t = average_precision_score(yv, ptv)
        # idx_map 基于筛选后样本 (与 eval_v4_allhorizon_ci.py 同口径)
        idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
        n_st = len(idx_map)
        # 配对 bootstrap
        d_auc, d_ap = [], []
        for _ in range(N_BOOT):
            sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            yy = yv[idx]
            if len(set(yy)) < 2: continue
            try:
                d_auc.append(roc_auc_score(yy, pvv[idx]) - roc_auc_score(yy, ptv[idx]))
                d_ap.append(average_precision_score(yy, pvv[idx]) - average_precision_score(yy, ptv[idx]))
            except Exception:
                continue
        a = 0.025
        results[f"{h}h"] = {
            "v4_auroc": float(auc_v), "tr_auroc": float(auc_t),
            "delta_auroc": float(auc_v - auc_t),
            "delta_auroc_lo": float(np.percentile(d_auc, a * 100)),
            "delta_auroc_hi": float(np.percentile(d_auc, (1 - a) * 100)),
            "v4_auprc": float(ap_v), "tr_auprc": float(ap_t),
            "delta_auprc": float(ap_v - ap_t),
            "delta_auprc_lo": float(np.percentile(d_ap, a * 100)),
            "delta_auprc_hi": float(np.percentile(d_ap, (1 - a) * 100)),
            "n_valid": int(m.sum()), "n_clusters": n_st, "n_boot": len(d_auc),
        }
        r = results[f"{h}h"]
        print(f"  {h:>2}h: ΔAUROC {r['delta_auroc']:+.3f} ({r['delta_auroc_lo']:.3f},{r['delta_auroc_hi']:.3f})  "
              f"ΔAUPRC {r['delta_auprc']:+.3f} ({r['delta_auprc_lo']:.3f},{r['delta_auprc_hi']:.3f})", flush=True)

    out = OUT_DIR / "v4_vs_tr_tcr_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
