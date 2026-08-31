#!/usr/bin/env python
"""治疗暴露归一化的器官贡献分析 (修改意见4 问题2, 五点方案第4点).

问题: CV 分量的治疗信息增益最大, 究竟因为 (a) 升压药直接进入 CV SOFA 评分定义
(定义性耦合), 还是 (b) 升压药暴露频繁/剂量信息丰富 (临床信号利用)?

分析:
  A. 逐器官 ΔMAE (TCR vs care-off zero-mode; 验证与 frozen_organ_noncv 一致)
  B. 每器官"相关治疗 6h 窗口内暴露"的锚点比例 (锚点级, 非 stay 级)
  C. 频率归一化增益 = ΔMAE / 暴露比例
  D. CV 深化: 升压药暴露二分 × 升压药强度三分位 → CV ΔMAE 分层 + CI
  E. CNS 对照: 镇静暴露二分 → CNS ΔMAE
  F. 同分层下非 CV 器官 ΔMAE (检验分层效应的 CV 特异性)

输出: results/v4/frozen_exposure_normalized.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

V12 = Path(r"F:/MIMIC3_1/V12")
V13 = Path(r"F:/MIMIC3_1/V13")
OUT = V13 / "results/v4/frozen_exposure_normalized.json"
ZERO_CACHE = V13 / "results/v4/careoff_zero_organ_cache.npz"
H6 = 5
N_BOOT = 2000; BOOT_SEED = 42

ORGANS = ["resp", "cv", "renal", "coag", "liver", "cns"]
# 器官 → 相关治疗 class_id (1-based, ACTION_CLASSES 序)
ORGAN_TX = {
    "cv": [1, 2, 3, 4, 5],           # 升压/正性肌力/扩血管/β阻/抗心律失常
    "cns": [11, 12, 13, 14, 15],     # 镇静/镇痛/肌松/抗精神/抗惊厥
    "resp": [16, 22],                # 呼吸支持/支扩
    "renal": [9, 10],                # 利尿/CRRT
    "coag": [20],                    # 抗凝
    "liver": [],                     # 无直接肝支持类
}
TX_NAME = {1: "vasopressor", 2: "inotrope", 11: "sedative", 12: "analgesic_opioid"}


def cluster_ci(stays, vals_per_anchor_fn, mask):
    """stay-cluster bootstrap CI for a statistic computed on masked anchors."""
    sv = stays[mask]
    uniq = np.unique(sv)
    idx_map = {s: np.where(sv == s)[0] for s in uniq}
    rng = np.random.RandomState(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx_local = np.concatenate([idx_map[s] for s in sampled])
        try:
            boots.append(vals_per_anchor_fn(idx_local))
        except Exception:
            continue
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    # ── 数据 ──
    tcr = np.load(V13 / "results/v4/traj_mae_ci_cache.npz", allow_pickle=True)
    pred_t = tcr["pred"]
    organ = tcr["organ_lab"]; omask = tcr["organ_mask"]; stays = tcr["stays"]

    # care-off zero-mode organ 预测: 缓存或推理 (careoff_traj_cache 是 block mode, 弃用)
    if ZERO_CACHE.exists():
        zc = np.load(ZERO_CACHE, allow_pickle=True)
        pred_c = zc["pred"]
        assert (zc["stays"] == stays).all()
        print("care-off zero-mode organ 缓存命中", flush=True)
    else:
        print("推理 care-off (zero mode) 3-seed organ ...", flush=True)
        configure_cuda(); dev = DEVICE
        spec = load_v4_proxy_contract(V12 / "configs/v4/v4_proxy_contract.json")
        ds = PLFOGTV4Dataset("test")
        loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
        ens = []
        for seed in [42, 52, 62]:
            ck = torch.load(V12 / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
            m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
            m.load_state_dict(ck["model_state_dict"], strict=False)
            m.to(dev).eval()
            ps = []
            with torch.inference_mode():
                for b in loader:
                    b = {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = m(b, stage="conditioned", future_treatment_mode="zero")
                    ps.append(out["organ_future"][:, :12, :].float().cpu().numpy())
            ens.append(np.concatenate(ps))
            print(f"  zero-mode seed {seed}: done", flush=True)
            del m; torch.cuda.empty_cache()
        pred_c = np.mean(ens, axis=0)
        np.savez_compressed(ZERO_CACHE, pred=pred_c, stays=stays)
        print(f"缓存保存: {ZERO_CACHE}", flush=True)

    rt = np.load(V12 / "data/gold/labels/rollout_targets_gmu_v6.npz", allow_pickle=True)
    tm = rt["sample_split"].astype(str) == "confirmation"
    fut_on = rt["future_act_on"][tm]      # (N, 24, 23) — 用 [:, :6, :] 表示 6h 窗口
    fut_int = rt["future_act_int"][tm]
    print(f"样本: {len(stays):,}; future_act_on shape: {fut_on.shape}", flush=True)

    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid_traj = (m_now * m_6h).sum(axis=1) > 0

    # ── A. 逐器官 ΔMAE (验证 frozen_organ_noncv) ──
    print("\n=== A. 逐器官 6h ΔMAE (care-off − TCR) ===", flush=True)
    organ_delta = {}
    for oi, name in enumerate(ORGANS):
        m = valid_traj & (m_6h[:, oi] > 0)
        tcr_mae = float(np.abs(pred_t[m, H6, oi] - o_6h[m, oi]).mean())
        co_mae = float(np.abs(pred_c[m, H6, oi] - o_6h[m, oi]).mean())
        organ_delta[name] = {"tcr_mae": tcr_mae, "co_mae": co_mae, "delta": co_mae - tcr_mae,
                             "n": int(m.sum())}
        print(f"  {name:<7} TCR={tcr_mae:.4f}  care-off={co_mae:.4f}  Δ={co_mae-tcr_mae:+.4f}", flush=True)
    ref = {"Cv": 0.2512, "CNS": 0.0528, "Resp": 0.0064, "Renal": 0.0033,
           "Coag": -0.0012, "Hepatic": 0.0009}
    mapn = {"cv": "Cv", "cns": "CNS", "resp": "Resp", "renal": "Renal",
            "coag": "Coag", "liver": "Hepatic"}
    ok = all(abs(organ_delta[k]["delta"] - ref[mapn[k]]) < 0.002 for k in organ_delta)
    print(f"  与 frozen_organ_noncv 一致性: {'PASS' if ok else 'FAIL'}", flush=True)

    # ── B. 每器官相关治疗 6h 窗口暴露比例 ──
    print("\n=== B. 相关治疗 6h 窗口暴露比例 (锚点级) ===", flush=True)
    win_on = fut_on[:, :H6 + 1, :] > 0          # (N, 6, 23) h1..h6 任一在治
    win_int = fut_int[:, :H6 + 1, :]
    exposure = {}
    for name, tx_ids in ORGAN_TX.items():
        if not tx_ids:
            exposure[name] = float("nan")
            print(f"  {name:<7} (无相关治疗类)", flush=True)
            continue
        ids0 = [i - 1 for i in tx_ids]
        exp = win_on[:, :, ids0].any(axis=(1, 2))
        exposure[name] = float(exp.mean())
        print(f"  {name:<7} 暴露率={exp.mean()*100:.1f}%  (类: {tx_ids})", flush=True)

    # ── C. 频率归一化增益 ──
    print("\n=== C. 频率归一化增益 = ΔMAE / 暴露率 ===", flush=True)
    norm_gain = {}
    for name in ORGAN_NAMES if (ORGAN_NAMES := ORGANS) else []:
        d = organ_delta[name]["delta"]; e = exposure[name]
        norm_gain[name] = d / e if e and not np.isnan(e) else float("nan")
        print(f"  {name:<7} Δ={d:+.4f} / 暴露={e*100:.1f}% → 归一化增益={norm_gain[name]:+.4f}", flush=True)

    # ── D. CV 深化: 升压药暴露 × 强度分层 ──
    print("\n=== D. CV 分量: 升压药暴露/强度分层 ===", flush=True)
    vaso_exp = win_on[:, :, 0]                   # vasopressor class_id=1 → idx 0
    vaso_any = vaso_exp.any(axis=1)
    vaso_int_mean = np.where(vaso_exp, win_int[:, :, 0], np.nan)
    vaso_int_mean = np.nanmean(vaso_int_mean, axis=1)  # 在治时刻均强度 (跳过 off 时刻)
    oi_cv = ORGANS.index("cv")
    m_cv = valid_traj & (m_6h[:, oi_cv] > 0)

    def cv_delta(idx_local, mask_full):
        """在 mask 的第 idx_local 局部索引上算 CV ΔMAE."""
        mm = mask_full
        # idx_local 是相对 stays[mm] 的
        it = pred_t[mm, H6, oi_cv][idx_local]; ic = pred_c[mm, H6, oi_cv][idx_local]
        lab = o_6h[mm, oi_cv][idx_local]
        return float((np.abs(ic - lab) - np.abs(it - lab)).mean())

    d_results = {}
    for label, sel in [("vaso_on", vaso_any), ("vaso_off", ~vaso_any)]:
        mm = m_cv & sel
        if mm.sum() < 50:
            continue
        d_pt = cv_delta(np.arange(mm.sum()), mm)
        # bootstrap
        sv = stays[mm]
        uniq = np.unique(sv); idx_map = {s: np.where(sv == s)[0] for s in uniq}
        rng = np.random.RandomState(BOOT_SEED)
        boots = []
        for _ in range(N_BOOT):
            smp = rng.choice(uniq, size=len(uniq), replace=True)
            il = np.concatenate([idx_map[s] for s in smp])
            it = pred_t[mm, H6, oi_cv][il]; ic = pred_c[mm, H6, oi_cv][il]
            lab = o_6h[mm, oi_cv][il]
            boots.append(float((np.abs(ic - lab) - np.abs(it - lab)).mean()))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        d_results[label] = {"n": int(mm.sum()), "delta": d_pt, "ci": ci}
        print(f"  {label:<9} n={mm.sum():>7,}  CV ΔMAE={d_pt:+.4f} (CI {ci[0]:+.4f}, {ci[1]:+.4f})", flush=True)

    # 强度三分位 (仅 vaso_any 锚点)
    sel = vaso_any & m_cv
    if sel.sum() > 150:
        vm = vaso_int_mean[sel]
        q1, q2 = np.nanpercentile(vm, [33.3, 66.7])
        for label, lo, hi in [("int_low", None, q1), ("int_mid", q1, q2), ("int_high", q2, None)]:
            if lo is None:
                s2 = sel & (vaso_int_mean <= q1)
            elif hi is None:
                s2 = sel & (vaso_int_mean > q2)
            else:
                s2 = sel & (vaso_int_mean > q1) & (vaso_int_mean <= q2)
            if s2.sum() < 50:
                continue
            d_pt = cv_delta(np.arange(s2.sum()), s2)
            d_results[label] = {"n": int(s2.sum()), "delta": d_pt,
                                "int_range": [float(lo) if lo is not None else float("nan"),
                                              float(hi) if hi is not None else float("nan")]}
            print(f"  {label:<9} n={s2.sum():>7,}  CV ΔMAE={d_pt:+.4f}", flush=True)
        print(f"  (强度三分位切点: q33={q1:.3f}, q66={q2:.3f}, log1p 尺度)", flush=True)

    # ── E. CNS: 镇静暴露分层 ──
    print("\n=== E. CNS 分量: 镇静暴露分层 ===", flush=True)
    sed_exp = win_on[:, :, 10]                   # sedative class_id=11 → idx 10
    sed_any = sed_exp.any(axis=1)
    oi_cns = ORGANS.index("cns")
    m_cns = valid_traj & (m_6h[:, oi_cns] > 0)
    e_results = {}
    for label, sel in [("sed_on", sed_any), ("sed_off", ~sed_any)]:
        mm = m_cns & sel
        if mm.sum() < 50:
            continue
        it = pred_t[mm, H6, oi_cns]; ic = pred_c[mm, H6, oi_cns]; lab = o_6h[mm, oi_cns]
        d_pt = float((np.abs(ic - lab) - np.abs(it - lab)).mean())
        e_results[label] = {"n": int(mm.sum()), "delta": d_pt}
        print(f"  {label:<9} n={mm.sum():>7,}  CNS ΔMAE={d_pt:+.4f}", flush=True)

    # ── F. 同分层下其他器官 (CV 特异性检验) ──
    print("\n=== F. 升压药分层下的其他器官 ΔMAE (特异性) ===", flush=True)
    f_results = {}
    for oi, name in enumerate(ORGANS):
        if name == "cv":
            continue
        row = {}
        for label, sel in [("vaso_on", vaso_any), ("vaso_off", ~vaso_any)]:
            mm = valid_traj & (m_6h[:, oi] > 0) & sel
            if mm.sum() < 50:
                continue
            it = pred_t[mm, H6, oi]; ic = pred_c[mm, H6, oi]; lab = o_6h[mm, oi]
            row[label] = float((np.abs(ic - lab) - np.abs(it - lab)).mean())
        f_results[name] = row
        on = row.get("vaso_on", float("nan")); off = row.get("vaso_off", float("nan"))
        print(f"  {name:<7} vaso_on Δ={on:+.4f}  vaso_off Δ={off:+.4f}", flush=True)

    out = {
        "organ_delta_mae": organ_delta,
        "exposure_rate_6h_window": exposure,
        "normalized_gain": norm_gain,
        "cv_vaso_strata": d_results,
        "cns_sed_strata": e_results,
        "other_organs_vaso_strata": f_results,
        "interpretation_frame": {
            "若归一化后 CV 仍显著最高": "定义性恢复是主要成分 (与机制归因定位一致)",
            "若归一化后差距缩小": "暴露频率/信息量解释占更大比重 (支持临床信号利用)",
            "若 vaso_on 组 CV ΔMAE >> vaso_off 组": "增益集中于实际暴露升压药的锚点, "
            "治疗信息利用与定义耦合均在暴露处发生",
        },
        "note": "TCR=traj_mae_ci_cache; care-off=careoff_zero_organ_cache (zero mode, "
                "与 frozen_organ_noncv 的 Cv Δ=+0.251 同口径); 暴露=6h 窗口内任一时刻在治; "
                "强度=log1p 等效剂量窗口均值; CI=stay-cluster bootstrap 2000 次.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
