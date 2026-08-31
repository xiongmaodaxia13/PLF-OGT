#!/usr/bin/env python
"""Measurement-conditioned TCR vs care-off (修改意见4 问题1→问题3 证据链焊接).

用与问题 1 (eval_measurement_conditioned.py) 完全相同的新测量掩码,
重新比较 TCR 与 care-off (zero mode):
  a. 条件化 macro-MAE (TCR/care-off/Δ) × h∈{1,3,6,12}
  b. 逐器官条件化 ΔMAE
  c. ICU-stay 配对 cluster bootstrap CI (Δmacro-MAE)
  d. 治疗分层 (新启动/已在治/未暴露) 的条件化 CV ΔMAE + 事件数稳定性检查

预测缓存: traj_mae_ci_cache.npz (TCR) + careoff_zero_organ_cache.npz (zero mode).
输出: results/v4/frozen_mc_tcr_careoff.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, r"F:/MIMIC3_1/V13/scripts")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from eval_measurement_conditioned import build_has_new, ORGANS, ORGAN_VARS

V12 = Path(r"F:/MIMIC3_1/V12")
V13 = Path(r"F:/MIMIC3_1/V13")
OUT = V13 / "results/v4/frozen_mc_tcr_careoff.json"
HORIZONS = [1, 3, 6, 12]
N_BOOT = 2000; BOOT_SEED = 42
VASO = 0


def cluster_ci_delta(stays, abs_err_t, abs_err_c, valid_mask):
    """配对 stay-cluster bootstrap: Δmacro = mean(care_off MAE - tcr MAE) 的 CI."""
    sv = stays
    uniq = np.unique(sv)
    idx_map = {s: np.where(sv == s)[0] for s in uniq}
    rng = np.random.RandomState(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        smp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[s] for s in smp])
        vm = valid_mask[idx]
        organ_d = []
        for o in range(6):
            sel = vm[:, o] > 0
            if sel.sum() == 0:
                organ_d.append(np.nan)
            else:
                organ_d.append(abs_err_c[idx][:, o][sel].mean() - abs_err_t[idx][:, o][sel].mean())
        boots.append(np.nanmean(organ_d))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    # ── 预测缓存 (行序与 rollout confirmation 一致, 已验证) ──
    tcr_c = np.load(V13 / "results/v4/traj_mae_ci_cache.npz", allow_pickle=True)
    co_c = np.load(V13 / "results/v4/careoff_zero_organ_cache.npz", allow_pickle=True)
    pred_t = tcr_c["pred"]; pred_c = co_c["pred"]
    organ = tcr_c["organ_lab"]; omask = tcr_c["organ_mask"]; stays = tcr_c["stays"]
    assert (co_c["stays"] == stays).all(), "care-off 缓存行序不一致"

    rt = np.load(V12 / "data/gold/labels/rollout_targets_gmu_v6.npz", allow_pickle=True)
    tm = rt["sample_split"].astype(str) == "confirmation"
    tau = rt["sample_tau"][tm].astype(np.float64)
    fut_on = rt["future_act_on"][tm]
    print(f"样本: {len(stays):,}", flush=True)

    # ── 新测量掩码 (与问题 1 完全一致) ──
    print("[1/3] 计算 has_new_measurement ...", flush=True)
    ev = np.load(V12 / "data/cache/event_gmu.npz", allow_pickle=True)
    off = np.load(V12 / "data/cache/sample_offsets_gmu.npz", allow_pickle=True)
    has_new = build_has_new(tau, stays, ev, off)

    # ── 条件化 TCR vs care-off ──
    print("[2/3] 条件化 macro-MAE + bootstrap CI ...", flush=True)
    results = {"mae": {}}
    for h in HORIZONS:
        lab = organ[:, h, :]
        msk = omask[:, h, :]
        cond = (msk * has_new[:, h - 1, :]) > 0
        abs_t = np.abs(pred_t[:, h - 1, :] - lab)
        abs_c = np.abs(pred_c[:, h - 1, :] - lab)
        organ_d = []
        hr = {"n_cells": int(cond.sum())}
        hr["tcr_macro"] = float(np.nanmean([abs_t[:, o][cond[:, o]].mean() if cond[:, o].sum() else np.nan for o in range(6)]))
        hr["co_macro"] = float(np.nanmean([abs_c[:, o][cond[:, o]].mean() if cond[:, o].sum() else np.nan for o in range(6)]))
        for o in range(6):
            sel = cond[:, o]
            organ_d.append(float(abs_c[:, o][sel].mean() - abs_t[:, o][sel].mean()) if sel.sum() else float("nan"))
        hr["delta_macro"] = float(np.nanmean(organ_d))
        hr["organ_delta"] = dict(zip(ORGANS, organ_d))
        ci = cluster_ci_delta(stays, abs_t, abs_c, cond)
        hr["delta_macro_ci"] = ci
        results["mae"][f"h{h}"] = hr
        print(f"  h={h:>2}: TCR={hr['tcr_macro']:.4f}  care-off={hr['co_macro']:.4f}  "
              f"Δ(care-off−TCR)={hr['delta_macro']:+.4f} (CI {ci[0]:+.4f},{ci[1]:+.4f})  "
              f"CV Δ={organ_d[1]:+.4f}", flush=True)

    # ── 治疗分层 × 条件化 CV ──
    print("[3/3] 治疗分层 (新启动/已在治/未暴露) 的条件化 CV ΔMAE ...", flush=True)
    # hist_act_on 从 dataset 收集 (CPU)
    from v6.data.dataset import collate
    from v6.data.v4_dataset import PLFOGTV4Dataset
    from torch.utils.data import DataLoader
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    hist_on = np.concatenate([b["hist_act_on"].numpy() for b in loader])
    assert len(hist_on) == len(stays)

    hist_vaso = hist_on[:, VASO] > 0
    fut_vaso = fut_on[:, :6, VASO].any(axis=1)
    G = {
        "G1_新启动": (~hist_vaso) & fut_vaso,
        "G2_锚点时已在治": hist_vaso & fut_vaso,
        "G3_未暴露": (~hist_vaso) & (~fut_vaso),
    }
    # 判别事件数检查
    delta = rt["delta_sofa"][tm]; cmask = rt["class_mask"][tm]
    m6 = cmask[:, 5] > 0
    y6 = (delta[:, 5] >= 2)

    h = 6
    lab = organ[:, h, :]; msk = omask[:, h, :]
    cond_cv = (msk[:, 1] * has_new[:, 5, 1]) > 0   # CV 分量 (idx 1) 条件化格子
    abs_t_cv = np.abs(pred_t[:, 5, 1] - lab[:, 1])
    abs_c_cv = np.abs(pred_c[:, 5, 1] - lab[:, 1])
    strata = {}
    for gname, gm in G.items():
        sel = gm & cond_cv
        n = int(sel.sum())
        n_ev = int((y6 & m6 & gm).sum())
        row = {"n_cond_cells": n, "n_group_anchors": int(gm.sum()), "n_events_total": n_ev}
        if n >= 100:
            row["cv_delta_mae"] = float(abs_c_cv[sel].mean() - abs_t_cv[sel].mean())
        strata[gname] = row
        print(f"  {gname}: 条件化格子 n={n:,} (组锚点 {gm.sum():,}, 总事件 {n_ev})  "
              f"CV ΔMAE={row.get('cv_delta_mae', float('nan')):+.4f}", flush=True)
    results["cv_vaso_strata_conditional_h6"] = strata

    results["note"] = ("掩码与 eval_measurement_conditioned.py 完全一致 (器官 SOFA 组成变量在 (τ,τ+h] "
                       "存在新测量且结局有效); TCR/care-off 均为 3-seed ensemble organ 预测缓存 "
                       "(care-off = zero mode); Δ = care-off − TCR, 正值表示 TCR 更好; "
                       "CI = 配对 stay-cluster bootstrap 2000 次.")
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)

    # 标题决策依据
    d6 = results["mae"]["h6"]
    lo, hi = d6["delta_macro_ci"]
    robust = lo > 0
    print(f"\n=== 标题决策: 6h 条件化 Δmacro CI=({lo:+.4f},{hi:+.4f}) → "
          f"{'不含零(稳健,可用标题二)' if robust else '含零(用稳妥标题一)'} ===", flush=True)


if __name__ == "__main__":
    main()
