#!/usr/bin/env python
"""V4 全时距 (1/3/6/12h) cluster bootstrap CI —— 第二档 (#6 补算).

策略: 一次推理拿到 OLP 和 TCR 的 3-seed ensemble logits, 缓存为 npz;
然后对 4 个时距分别做患者层面聚类 bootstrap CI (n_boot=2000).
与 eval_v4_frozen_main.py 同口径, 只是 CI 从"仅6h"扩到"全部时距".

输出:
  results/v4/allhorizon_ci_logits.npz   (缓存, 便于重跑bootstrap)
  results/v4/allhorizon_ci.json         (全时距 AUROC/AUPRC 点估计 + CI)

用法: python scripts/eval_v4_allhorizon_ci.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 1=RTX 4090 D (46GB); 0=4080 Laptop(12GB)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
SEEDS = [42, 52, 62]
N_BOOT = 2000
BOOT_SEED = 42
OUT_DIR = RESULTS_DIR / "v4"


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def run_inference(loader, dev, spec, mode):
    """对 3 seed 做 ensemble, 返回 ens_p (N,12,3). mode: 'actual'(TCR) / 'zero'(OLP)."""
    ens_logits = []
    for seed in SEEDS:
        ckpt = REPO / f"runs/v4/full_s5_s{seed}/best.pt"
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()
        sl = []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(batch, stage="conditioned", future_treatment_mode=mode)
                sl.append(out["class_logits"].float().cpu().numpy())
        ens_logits.append(np.concatenate(sl))
        print(f"    {mode} seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    ens_logits = np.mean(ens_logits, axis=0)  # (N,12,3)
    return softmax_np(ens_logits)


def cluster_bootstrap_ci(p, y, stays, n_boot, seed):
    """患者层面聚类 bootstrap. p/y/stays 已按有效样本筛选(同一索引集)."""
    idx_map = {s: np.where(stays == s)[0] for s in np.unique(stays)}
    n_stays = len(idx_map)
    rng = np.random.RandomState(seed)
    auc_b, ap_b = [], []
    for _ in range(n_boot):
        sampled = rng.choice(list(idx_map.keys()), size=n_stays, replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        try:
            auc_b.append(roc_auc_score(yy, p[idx]))
            ap_b.append(average_precision_score(yy, p[idx]))
        except Exception:
            continue
    a = 0.025
    return {
        "auroc_lo": float(np.percentile(auc_b, a * 100)),
        "auroc_hi": float(np.percentile(auc_b, (1 - a) * 100)),
        "auprc_lo": float(np.percentile(ap_b, a * 100)),
        "auprc_hi": float(np.percentile(ap_b, (1 - a) * 100)),
        "n_boot": len(auc_b), "n_clusters": n_stays,
    }


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第二档 #6: 全时距 cluster bootstrap CI", flush=True)
    print("=" * 60, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)} samples", flush=True)

    # 标签
    all_delta, all_mask, all_stays = [], [], []
    for batch in loader:
        all_delta.append(batch["delta_sofa"].numpy())
        all_mask.append(batch["class_mask"].numpy())
        all_stays.append(batch["stay_id"].numpy())
    delta = np.concatenate(all_delta); mask = np.concatenate(all_mask)
    stays = np.concatenate(all_stays)
    print(f"labels: delta {delta.shape}, mask {mask.shape}, stays {stays.shape}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "allhorizon_ci_logits.npz"

    if cache.exists():
        print(f"\n加载缓存: {cache}", flush=True)
        c = np.load(cache)
        ens_p_tcr, ens_p_olp = c["tcr"], c["olp"]
    else:
        print("\n=== TCR (actual) 3-seed ensemble ===", flush=True)
        ens_p_tcr = run_inference(loader, dev, spec, "actual")
        print("\n=== OLP (zero) 3-seed ensemble ===", flush=True)
        ens_p_olp = run_inference(loader, dev, spec, "zero")
        np.savez(cache, tcr=ens_p_tcr, olp=ens_p_olp,
                 delta=delta, mask=mask, stays=stays)
        print(f"缓存已存: {cache}", flush=True)

    # 全时距 CI
    results = {}
    for mode_name, ens_p in [("TCR", ens_p_tcr), ("OLP", ens_p_olp)]:
        print(f"\n=== {mode_name} 全时距 CI ===", flush=True)
        mode_res = {}
        for h in HORIZONS:
            hi = h - 1
            p = ens_p[:, hi, 0]  # worsen 概率
            m = mask[:, hi] > 0
            y = (delta[:, hi] >= 2).astype(float)
            n = int(m.sum())
            if n == 0 or y[m].sum() <= 5 or len(set(y[m])) < 2:
                print(f"  {h:>2}h: 有效样本不足, 跳过", flush=True)
                continue
            auc = float(roc_auc_score(y[m], p[m]))
            ap = float(average_precision_score(y[m], p[m]))
            ci = cluster_bootstrap_ci(p[m], y[m], stays[m], N_BOOT, BOOT_SEED)
            mode_res[f"{h}h"] = {"auroc": auc, "auprc": ap,
                                 "n": n, "prev": float(y[m].mean()),
                                 "ci": ci}
            print(f"  {h:>2}h: AUROC {auc:.4f} ({ci['auroc_lo']:.4f}-{ci['auroc_hi']:.4f})  "
                  f"AUPRC {ap:.4f} ({ci['auprc_lo']:.4f}-{ci['auprc_hi']:.4f})  "
                  f"n={n} prev={y[m].mean()*100:.1f}%  nboot={ci['n_boot']}", flush=True)
        results[mode_name] = mode_res

    out = OUT_DIR / "allhorizon_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
