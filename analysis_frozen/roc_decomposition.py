#!/usr/bin/env python
"""ROC 提升的来源拆解 (修改意见4 问题2 反驳证据).

审稿人: TCR−care-off 的判别提升主要是 CV 标签重构.
反驳设计: 按升压药暴露动态分组, 在"从未接触升压药"的锚点组 (恶化完全由生理
驱动, CV 由 MAP 定义) 上检验治疗信息的判别增益:

  G1 新启动: 锚点时 off & 未来 6h 窗口 on   (治疗→定义直接通路, 双方都预期增益大)
  G2 持续:   锚点时 on  & 未来窗口 on       (CV 分数依赖当时剂量)
  G3 从未接触: off → off                    (生理驱动; **核心反驳组**)

每组内: n, 总恶化率(ΔSOFA≥2), non-CV 恶化率(Δnoncv≥2), TCR/care-off AUROC,
平均预测概率差. G3 的 ΔAUROC 配 stay-cluster bootstrap CI.

另: 新启动组 (G1) 的 non-CV 恶化率 vs G3 —— 检验"新启动升压药是全局恶化前兆".

输出: results/v4/frozen_roc_decomposition.json
"""
from __future__ import annotations
import os, sys, json
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
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results/v4/frozen_roc_decomposition.json")
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42
VASO = 0          # class_id 1 → idx 0


def move(b, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("ROC 提升来源拆解: 升压药暴露动态分组 (G1新启动/G2持续/G3从未接触)", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    # ── 收集标签 + 锚点时治疗状态 ──
    hist_on_l, cmask_l, stay_l = [], [], []
    for b in loader:
        hist_on_l.append(b["hist_act_on"].numpy())
        cmask_l.append(b["class_mask"].numpy())
        stay_l.append(b["stay_id"].numpy())
    hist_on = np.concatenate(hist_on_l)   # (N, 23) 锚点时在治
    cmask = np.concatenate(cmask_l)
    stays = np.concatenate(stay_l)

    rt = np.load(REPO / "data/gold/labels/rollout_targets_gmu_v6.npz", allow_pickle=True)
    tm = rt["sample_split"].astype(str) == "confirmation"
    delta = rt["delta_sofa"][tm]          # (N, 24)
    delta_ncv = rt["delta_noncv"][tm]
    fut_on = rt["future_act_on"][tm]      # (N, 24, 23)
    assert delta.shape[0] == len(stays)

    hist_vaso = hist_on[:, VASO] > 0
    fut_vaso = fut_on[:, :H6 + 1, VASO].any(axis=1)

    G = {
        "G1_新启动": (~hist_vaso) & fut_vaso,
        "G2_持续": hist_vaso & fut_vaso,
        "G3_从未接触": (~hist_vaso) & (~fut_vaso),
    }

    m6 = cmask[:, H6] > 0
    y_tot = (delta[:, H6] >= 2).astype(float)
    y_ncv = (delta_ncv[:, H6] >= 2).astype(float)

    # ── 推理: TCR (actual) + care-off (zero) 3-seed ensemble logits ──
    ens = {"actual": [], "zero": []}
    for mode in ["actual", "zero"]:
        for seed in SEEDS:
            ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
            m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
            m.load_state_dict(ck["model_state_dict"], strict=False)
            m.to(dev).eval()
            ls = []
            with torch.inference_mode():
                for b in loader:
                    b = move(b, dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = m(b, stage="conditioned", future_treatment_mode=mode)
                    ls.append(out["class_logits"].float().cpu().numpy())
            ens[mode].append(np.concatenate(ls))
            print(f"  {mode} seed {seed}: done", flush=True)
            del m; torch.cuda.empty_cache()
    tcr_logits = np.mean(ens["actual"], axis=0)
    co_logits = np.mean(ens["zero"], axis=0)
    tcr_p = softmax_np(tcr_logits[:, H6, :])[:, 0]
    co_p = softmax_np(co_logits[:, H6, :])[:, 0]

    # 验证口径: 全体 AUROC 应≈ 0.875 / 0.794
    v = m6
    print(f"\n口径验证: TCR AUROC={roc_auc_score(y_tot[v], tcr_p[v]):.4f} (应≈0.875)  "
          f"care-off={roc_auc_score(y_tot[v], co_p[v]):.4f} (应≈0.794)", flush=True)

    # ── 分组分析 ──
    print(f"\n{'='*70}", flush=True)
    results = {}
    for gname, gmask in G.items():
        sel = gmask & m6
        n = int(sel.sum())
        ev = float(y_tot[sel].mean())
        ev_ncv = float(y_ncv[sel].mean())
        row = {"n": n, "prev_total": ev, "prev_noncv": ev_ncv,
               "tcr_p_mean": float(tcr_p[sel].mean()), "co_p_mean": float(co_p[sel].mean()),
               "delta_p": float(tcr_p[sel].mean() - co_p[sel].mean())}
        yy = y_tot[sel]
        if len(set(yy)) > 1 and yy.sum() >= 5:
            row["tcr_auroc"] = float(roc_auc_score(yy, tcr_p[sel]))
            row["co_auroc"] = float(roc_auc_score(yy, co_p[sel]))
            row["delta_auroc"] = row["tcr_auroc"] - row["co_auroc"]
            try:
                row["tcr_auprc"] = float(average_precision_score(yy, tcr_p[sel]))
                row["co_auprc"] = float(average_precision_score(yy, co_p[sel]))
                row["delta_auprc"] = row["tcr_auprc"] - row["co_auprc"]
            except Exception:
                pass
        results[gname] = row
        print(f"\n[{gname}] n={n:,}  总恶化率={ev*100:.1f}%  non-CV恶化率={ev_ncv*100:.1f}%", flush=True)
        print(f"  TCR p̄={row['tcr_p_mean']:.4f}  care-off p̄={row['co_p_mean']:.4f}  Δp={row['delta_p']:+.4f}", flush=True)
        if "delta_auroc" in row:
            print(f"  AUROC: TCR={row['tcr_auroc']:.4f}  care-off={row['co_auroc']:.4f}  "
                  f"Δ={row['delta_auroc']:+.4f}   AUPRC Δ={row.get('delta_auprc', float('nan')):+.4f}", flush=True)

    # ── G3 ΔAUROC bootstrap CI (核心反驳证据) ──
    print(f"\n=== G3 (从未接触升压药) ΔAUROC bootstrap CI (n_boot={N_BOOT}) ===", flush=True)
    g3 = G["G3_从未接触"] & m6
    yy = y_tot[g3]; tp = tcr_p[g3]; cp = co_p[g3]; sv = stays[g3]
    uniq = np.unique(sv)
    idx_map = {s: np.where(sv == s)[0] for s in uniq}
    rng = np.random.RandomState(BOOT_SEED)
    b_dauroc, b_dauprc = [], []
    pt = roc_auc_score(yy, tp) - roc_auc_score(yy, cp)
    try:
        pp = average_precision_score(yy, tp) - average_precision_score(yy, cp)
    except Exception:
        pp = float("nan")
    for _ in range(N_BOOT):
        smp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[s] for s in smp])
        yb = yy[idx]
        if len(set(yb)) < 2 or yb.sum() < 3:
            continue
        try:
            b_dauroc.append(roc_auc_score(yb, tp[idx]) - roc_auc_score(yb, cp[idx]))
            b_dauprc.append(average_precision_score(yb, tp[idx]) - average_precision_score(yb, cp[idx]))
        except Exception:
            continue
    g3_ci = {
        "delta_auroc": float(pt),
        "delta_auroc_ci": [float(np.percentile(b_dauroc, 2.5)), float(np.percentile(b_dauroc, 97.5))],
        "delta_auprc": float(pp),
        "delta_auprc_ci": [float(np.percentile(b_dauprc, 2.5)), float(np.percentile(b_dauprc, 97.5))],
        "n_boot": len(b_dauroc),
    }
    results["G3_bootstrap_ci"] = g3_ci
    print(f"  ΔAUROC = {pt:+.4f} (95% CI {g3_ci['delta_auroc_ci'][0]:+.4f} ~ {g3_ci['delta_auroc_ci'][1]:+.4f})", flush=True)
    print(f"  ΔAUPRC = {pp:+.4f} (95% CI {g3_ci['delta_auprc_ci'][0]:+.4f} ~ {g3_ci['delta_auprc_ci'][1]:+.4f})", flush=True)
    ci = g3_ci["delta_auroc_ci"]
    print(f"  CI 含零: {'是' if ci[0] <= 0 <= ci[1] else '否'}", flush=True)

    # ── 附加: G1 内 non-CV 恶化富集 (新启动升压药 = 全局恶化前兆) ──
    g1 = G["G1_新启动"] & m6
    g3m = G["G3_从未接触"] & m6
    print(f"\n=== 新启动升压药的恶化前兆检验 ===", flush=True)
    print(f"  G1 总恶化率 {y_tot[g1].mean()*100:.1f}% vs G3 {y_tot[g3m].mean()*100:.1f}%  "
          f"(比值 {y_tot[g1].mean()/max(y_tot[g3m].mean(),1e-9):.2f})", flush=True)
    print(f"  G1 non-CV恶化率 {y_ncv[g1].mean()*100:.1f}% vs G3 {y_ncv[g3m].mean()*100:.1f}%  "
          f"(比值 {y_ncv[g1].mean()/max(y_ncv[g3m].mean(),1e-9):.2f})", flush=True)
    results["precursor_check"] = {
        "G1_prev_total": float(y_tot[g1].mean()), "G3_prev_total": float(y_tot[g3m].mean()),
        "G1_prev_noncv": float(y_ncv[g1].mean()), "G3_prev_noncv": float(y_ncv[g3m].mean()),
    }

    out = {"groups": results,
           "note": "分组按升压药 (class_id=1) 锚点时在治 × 未来6h窗口暴露; 判别结局为 6h 总SOFA "
                   "恶化 ΔSOFA≥2; TCR/care-off 均为 3-seed ensemble logits (zero mode); "
                   "G3 (从未接触) 的恶化完全由生理与其他治疗定义路径驱动, 其 ΔROC 即治疗信息对"
                   "非升压药定义恶化的判别贡献. CI = stay-cluster bootstrap 2000."}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
