#!/usr/bin/env python
"""Measurement-conditioned evaluation (修改意见4 问题1).

审稿人要求: 仅在 t 与 t+h 之间该器官存在新的有效临床测量的锚点上,
重新计算 MAE / macro-MAE, 并报告每器官各时距"存在新测量"的比例.

判定规则: 器官 o 在锚点 (stay, τ) 的 horizon h 上"有新测量" ⟺
该 stay 的事件缓存中存在 var_id ∈ SOFA组成变量(o) 且 avail_rel_h ∈ (τ, τ+h].

器官 → SOFA 组成变量 (contracts.py var_id):
  resp  = [7, 8, 9]   (PaO2, pO2, FiO2)
  cv    = [15, 16]    (ABPm, NIBPm)  [升压药属治疗输入, 不计]
  renal = [1, 2]      (Cr, Cr serum)
  coag  = [5]         (PLT)
  liver = [3, 4]      (Bilirubin)
  cns   = [11, 12, 13] (GCS-E/V/M)

模型: PLF-OGT (traj_mae_ci_cache.npz), Transformer (transformertcr_trajectory_cache.npz),
persistence (= organ[:, 0]). 行序已验证与 rollout npz confirmation split 对齐.

输出: results/v4/frozen_measurement_conditioned.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

V12 = Path(r"F:/MIMIC3_1/V12")
V13 = Path(r"F:/MIMIC3_1/V13")
OUT = V13 / "results/v4/frozen_measurement_conditioned.json"

HORIZONS = [1, 3, 6, 12]
ORGANS = ["resp", "cv", "renal", "coag", "liver", "cns"]
ORGAN_VARS = {
    "resp": [7, 8, 9], "cv": [15, 16], "renal": [1, 2],
    "coag": [5], "liver": [3, 4], "cns": [11, 12, 13],
}
N_BOOT = 2000; BOOT_SEED = 42


def build_has_new(tau_arr, stay_arr, ev, off):
    """对每个样本 × 每个 horizon × 每个器官, 判定 (τ, τ+h] 内是否有新测量.

    返回 (N, 12, 6) bool 数组 (h_idx = h-1).
    """
    offsets = {int(s): (int(a), int(b)) for s, a, b in
               zip(off["event_stays"], off["event_start"], off["event_end"])}
    ev_var = ev["var_id"]; ev_time = ev["avail_rel_h"]

    N = len(tau_arr)
    has_new = np.zeros((N, 12, 6), dtype=bool)

    # 按 stay 分组样本
    stay_to_rows = {}
    for i in range(N):
        stay_to_rows.setdefault(int(stay_arr[i]), []).append(i)

    h_all = np.arange(1, 13)
    for sid, rows in stay_to_rows.items():
        if sid not in offsets:
            continue
        a, b = offsets[sid]
        vars_s = ev_var[a:b]; times_s = ev_time[a:b]
        rows = np.array(rows)
        taus = tau_arr[rows]
        for oi, organ in enumerate(ORGANS):
            vmask = np.isin(vars_s, ORGAN_VARS[organ])
            if not vmask.any():
                continue
            t_o = np.sort(times_s[vmask])  # 该器官组成变量的全部测量时刻
            if len(t_o) == 0:
                continue
            # 对每个样本: 新测量存在 ⟺ searchsorted(t_o, τ+h) > searchsorted(t_o, τ)
            # (即 (τ, τ+h] 区间内有 ≥1 个测量时刻)
            lo = np.searchsorted(t_o, taus, side="right")   # > τ 的第一个位置
            for hi_idx, h in enumerate(h_all):
                hi = np.searchsorted(t_o, taus + h, side="right")  # ≤ τ+h
                has_new[rows, hi_idx, oi] = hi > lo
    return has_new


def pooled_mae(pred, lab, msk):
    sel = msk > 0
    if sel.sum() == 0:
        return float("nan")
    return float(np.abs(pred[sel] - lab[sel]).mean())


def cluster_bootstrap_ci(stays, per_cell_abs_err, valid_mask, n_clusters_hint=None):
    """stay-cluster bootstrap 的 MAE CI. per_cell_abs_err: (N, 6) 某一 horizon."""
    uniq = np.unique(stays)
    idx_map = {s: np.where(stays == s)[0] for s in uniq}
    rng = np.random.RandomState(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        vm = valid_mask[idx]
        organ_mae = []
        for o in range(6):
            sel = vm[:, o] > 0
            organ_mae.append(per_cell_abs_err[idx][:, o][sel].mean() if sel.sum() else np.nan)
        boots.append(np.nanmean(organ_mae))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    print("=" * 70, flush=True)
    print("Measurement-conditioned evaluation (修改意见4 问题1)", flush=True)
    print("=" * 70, flush=True)

    # ── 加载数据 ──
    rt = np.load(V12 / "data/gold/labels/rollout_targets_gmu_v6.npz", allow_pickle=True)
    test_mask = rt["sample_split"].astype(str) == "confirmation"
    tau = rt["sample_tau"][test_mask].astype(np.float64)
    organ_lab_full = rt["organ"][test_mask]
    organ_mask_full = rt["organ_mask"][test_mask]

    cache_plf = np.load(V13 / "results/v4/traj_mae_ci_cache.npz", allow_pickle=True)
    cache_tr = np.load(V13 / "results/v4/transformertcr_trajectory_cache.npz", allow_pickle=True)
    pred_plf = cache_plf["pred"]; pred_tr = cache_tr["pred"]
    stays = cache_plf["stays"]
    assert (cache_plf["organ_lab"] == organ_lab_full).all(), "PLF cache 行序不一致"
    assert (cache_tr["organ_lab"] == organ_lab_full).all(), "TR cache 行序不一致"
    print(f"样本: {len(tau):,}, stays: {len(np.unique(stays)):,}", flush=True)

    # ── 新测量判定 ──
    print("\n[1/3] 计算 has_new_measurement (N×12×6)...", flush=True)
    ev = np.load(V12 / "data/cache/event_gmu.npz", allow_pickle=True)
    off = np.load(V12 / "data/cache/sample_offsets_gmu.npz", allow_pickle=True)
    has_new = build_has_new(tau, stays, ev, off)
    frac = has_new.mean(axis=0)  # (12, 6)
    print("  每器官各时距新测量比例:", flush=True)
    print(f"  {'organ':<8}" + "".join(f"{'h='+str(h):>8}" for h in HORIZONS), flush=True)
    for oi, o in enumerate(ORGANS):
        print(f"  {o:<8}" + "".join(f"{frac[h-1, oi]:>8.3f}" for h in HORIZONS), flush=True)

    # ── 条件化 MAE ──
    print("\n[2/3] 条件化 MAE (仅新测量格子)...", flush=True)
    models = {
        "PLF": pred_plf,
        "TR": pred_tr,
        "persistence": None,  # organ[:, 0]
    }
    results = {"fractions": {h: {o: float(frac[h-1, oi]) for oi, o in enumerate(ORGANS)}
                             for h in HORIZONS},
               "mae": {}, "bootstrap_ci_macro": {}}

    for h in HORIZONS:
        lab_now = organ_lab_full[:, 0, :]      # τ 时刻
        lab_h = organ_lab_full[:, h, :]        # τ+h  (organ index h = τ+h)
        msk_h = organ_mask_full[:, h, :]
        new_h = has_new[:, h-1, :]             # (N, 6)
        cond_mask = msk_h * new_h              # 条件化有效格子

        hr = {}
        for name, pred in models.items():
            p = lab_now if pred is None else pred[:, h-1, :]
            organ_mae = [pooled_mae(p[:, o], lab_h[:, o], cond_mask[:, o]) for o in range(6)]
            macro = float(np.nanmean(organ_mae))
            hr[name] = {"organ_mae": organ_mae, "macro_mae": macro}
            print(f"  h={h:>2} {name:<12} macro={macro:.4f}  per-organ="
                  f"{[round(x, 3) for x in organ_mae]}", flush=True)
        # SOFA 总分 (≥1 器官有新测量的锚点)
        any_new = new_h.any(axis=1) & (msk_h.sum(axis=1) > 0)
        sofa = {}
        for name, pred in models.items():
            p_sofa = lab_now.sum(1) if pred is None else pred[:, h-1, :].sum(1)
            l_sofa = lab_h.sum(1)
            sofa[name] = float(np.abs(p_sofa[any_new] - l_sofa[any_new]).mean())
        hr["sofa_total_mae_anyNew"] = sofa
        hr["n_any_new"] = int(any_new.sum())
        results["mae"][f"h{h}"] = hr

    # ── macro-MAE bootstrap CI (stay cluster) ──
    print(f"\n[3/3] macro-MAE cluster bootstrap CI (n_boot={N_BOOT})...", flush=True)
    for h in HORIZONS:
        lab_now = organ_lab_full[:, 0, :]
        lab_h = organ_lab_full[:, h, :]
        msk_h = organ_mask_full[:, h, :]
        cond_mask = (msk_h * has_new[:, h-1, :]) > 0
        for name, pred in models.items():
            p = lab_now if pred is None else pred[:, h-1, :]
            abs_err = np.abs(p - lab_h)
            ci = cluster_bootstrap_ci(stays, abs_err, cond_mask)
            results["bootstrap_ci_macro"][f"h{h}"] = results["bootstrap_ci_macro"].get(f"h{h}", {})
            results["bootstrap_ci_macro"][f"h{h}"][name] = ci
            print(f"  h={h:>2} {name:<12} macro CI = ({ci[0]:.4f}, {ci[1]:.4f})", flush=True)

    results["method"] = {
        "definition": "器官 o 在 (τ, τ+h] 有新测量 ⟺ 事件缓存中存在该器官 SOFA 组成变量"
                      " (resp=PaO2/FiO2, cv=MAP, renal=Cr, coag=PLT, liver=Bilirubin, cns=GCS)"
                      " 的 avail_rel_h ∈ (τ, τ+h]",
        "note": "MAE 仅在新测量且 organ_mask 有效的格子上聚合; macro = 6 器官 pooled MAE 均值;"
                " SOFA 总分在 ≥1 器官有新测量的锚点上计算",
        "caches": ["traj_mae_ci_cache.npz (PLF 3-seed ensemble TCR)",
                   "transformertcr_trajectory_cache.npz (TR)"],
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
