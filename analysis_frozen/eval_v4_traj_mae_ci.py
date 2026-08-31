#!/usr/bin/env python
"""V4 连续器官轨迹 MAE 的 cluster bootstrap CI —— 第二档 (#5), 口径修正版.

陆老师 #5: TCR 主指标 (6h macro-MAE, SOFA总分MAE) 缺 CI.
严格复刻 eval_v4_organ_mae.py 的 compute_mae 全局加权平均口径:
    diff = |pred-target| * mask;  mae = diff.sum() / mask.sum()
bootstrap 每次重采样患者后, 用同一 compute_mae 重算 (保证点估计与原文件 4 位小数一致).

输出: results/v4/traj_mae_ci.json
用法: python scripts/eval_v4_traj_mae_ci.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/v4"
HORIZONS = [1, 3, 6, 12]
N_BOOT = 2000
BOOT_SEED = 42


def compute_mae(pred, target, mask):
    """与 eval_v4_organ_mae.py 完全一致的全局加权平均."""
    diff = np.abs(pred - target) * mask
    return diff.sum() / max(mask.sum(), 1)


def metrics_at_h(pred, target, mask_all, hi):
    """返回该时距的 (sofa_total_mae, macro_mae) — 原口径."""
    omh = mask_all[:, hi, :]  # (N,6)
    pred_sum = (pred[:, hi, :] * omh).sum(axis=-1)
    target_sum = (target[:, hi, :] * omh).sum(axis=-1)
    n_valid = omh.sum(axis=-1)
    mask_total = (n_valid > 0).astype(float)
    sofa_mae = compute_mae(pred_sum, target_sum, mask_total)
    organ_maes = [compute_mae(pred[:, hi, o], target[:, hi, o], mask_all[:, hi, o]) for o in range(6)]
    return float(sofa_mae), float(np.mean(organ_maes))


def main():
    cache = OUT_DIR / "traj_mae_ci_cache.npz"
    if not cache.exists():
        print(f"!! 缓存不存在: {cache}. 请先跑 (含推理的版本).", flush=True)
        return
    c = np.load(cache)
    pred = c["pred"]                 # (N,12,6) 3-seed TCR ensemble organ_future
    organ_lab = c["organ_lab"]       # (N,25,6)
    organ_mask = c["organ_mask"]     # (N,25,6)
    stays = c["stays"]
    target = organ_lab[:, 1:13, :]   # (N,12,6)
    mask_all = organ_mask[:, 1:13, :]

    print("=" * 60, flush=True)
    print("第二档 #5: TCR 轨迹 MAE 的 cluster bootstrap CI (原口径)", flush=True)
    print("=" * 60, flush=True)
    # 点估计核对
    for h in HORIZONS:
        s, mm = metrics_at_h(pred, target, mask_all, h - 1)
        print(f"  {h:>2}h 点估计: 总分MAE={s:.4f}  macroMAE={mm:.4f}", flush=True)

    # cluster bootstrap
    results = {}
    idx_map = {s: np.where(stays == s)[0] for s in np.unique(stays)}
    n_st = len(idx_map)
    rng = np.random.RandomState(BOOT_SEED)
    print(f"\nbootstrap (n_boot={N_BOOT}, n_clusters={n_st})...", flush=True)

    # 预采样 bootstrap 索引 (所有时距共用, 保证配对)
    boot_idxs = []
    for _ in range(N_BOOT):
        sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        boot_idxs.append(np.concatenate([idx_map[s] for s in sampled]))

    for h in HORIZONS:
        hi = h - 1
        sofa_b, mac_b = [], []
        for idx in boot_idxs:
            s, mm = metrics_at_h(pred[idx], target[idx], mask_all[idx], hi)
            sofa_b.append(s); mac_b.append(mm)
        a = 0.025
        s_pe, mm_pe = metrics_at_h(pred, target, mask_all, hi)
        results[f"{h}h"] = {
            "sofa_total_mae": s_pe,
            "sofa_total_ci": [float(np.percentile(sofa_b, a * 100)), float(np.percentile(sofa_b, (1 - a) * 100))],
            "macro_mae": mm_pe,
            "macro_ci": [float(np.percentile(mac_b, a * 100)), float(np.percentile(mac_b, (1 - a) * 100))],
            "n_clusters": n_st, "n_boot": N_BOOT,
        }
        print(f"  {h:>2}h: 总分 {s_pe:.4f} ({results[f'{h}h']['sofa_total_ci'][0]:.4f}-{results[f'{h}h']['sofa_total_ci'][1]:.4f})  "
              f"macro {mm_pe:.4f} ({results[f'{h}h']['macro_ci'][0]:.4f}-{results[f'{h}h']['macro_ci'][1]:.4f})", flush=True)

    out = OUT_DIR / "traj_mae_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
