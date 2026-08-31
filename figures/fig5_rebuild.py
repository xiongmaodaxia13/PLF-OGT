#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 5 重构 — 跨队列独立再开发重现部分方向性结果 (2026-08-29 定稿方案).

图 5｜跨队列独立再开发重现部分方向性结果
Panel a PLF-Transformer 轨迹代价 forest / b 循环分量模型排序 dot / c 治疗信息增益 forest / d 残差患者对应 forest

颜色: a/c = 金 #F1A93B (PLF/治疗信息); d = 蓝紫 #8891DB (residual); b = 图 2 三模型色。
cohort 用 marker 形状区分: GMUICU=●, MIMIC-IV=■; Panel a 敏感性组(|ΔSOFA|>=1)用空心。

数据锁定 (容差 0.001), 全部来自 frozen 文件:
- a GMUICU: results/v4/frozen_changed_state_ci.json (all + changed_ge1 的 plf_minus_tr)
- a MIMIC:  results_mimic/frozen_mimic_tr_retrained.json bootstrap_ci  【注意: results_mimic/
  frozen_changed_state_ci.json 是已撤回的旧 Transformer 基线(tr 1.222), 禁止使用】
- b GMUICU: traj 缓存重算 (终点有效口径, 同图 2); b MIMIC: 正文 P107 锁定常量 (无独立 frozen json)
- c GMUICU: results/v4/ALL_RESULTS_SUMMARY.json ot_paired (=表2注 +0.372)
- c MIMIC:  results_mimic/mimic_3seed_corrected.json careoff_paired.6h
- d GMUICU: results/v4/frozen_patient_specific_r_v2.json bootstrap_seed42.shuffled
- d MIMIC:  results_mimic/frozen_mimic_patient_specific.json bootstrap_seed42
输出: fig5.pdf / fig5.png(300dpi) / fig5_caption_cn.txt / fig5_data_used.csv
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig2_rebuild import (COL_STATIC, COL_TRANS, COL_PLF, style_axis, panel_label,
                          save_figure, write_safe)  # style block 经 import 生效

R_PURPLE = "#8891DB"   # 图 4 残差状态色
C_TEXT_D = "#333333"   # cohort 图例 marker 色

# ══════════ 锁定值 (跨队列对照表 / P106-P108) ══════════
EXPECTED = {
    # Panel a: (点, lo, hi)
    "gmu_all": (0.276, 0.235, 0.317), "mimic_all": (0.087, 0.078, 0.096),
    "gmu_chg": (0.188, 0.148, 0.231), "mimic_chg": (0.041, 0.031, 0.052),
    # Panel b: (static, TR, PLF) 6h 循环分量 MAE
    "gmu_cv": (0.431, 0.152, 0.177), "mimic_cv": (0.280, 0.246, 0.236),
    # Panel c: ΔAUPRC (TCR - care-off)
    "gmu_c": (0.372, 0.349, 0.394), "mimic_c": (0.092, 0.076, 0.108),
    # Panel d: ΔAUPRC (自身 - 跨患者替换)
    "gmu_d": (0.144, 0.119, 0.171), "mimic_d": (0.073, 0.060, 0.086),
}


def load_data(v4=Path(r"F:/MIMIC3_1/V13/results/v4"),
              rm=Path(r"F:/MIMIC3_1/V13/results_mimic")):
    # Panel a
    g = json.load(open(v4 / "frozen_changed_state_ci.json", encoding="utf-8"))
    a_gmu_all = g["all"]["plf_minus_tr"]
    a_gmu_chg = g["changed_ge1"]["plf_minus_tr"]
    m = json.load(open(rm / "frozen_mimic_tr_retrained.json", encoding="utf-8"))["bootstrap_ci"]
    a_mimic_all = m["all"]["plf_minus_tr"]
    a_mimic_chg = m["changed_ge1"]["plf_minus_tr"]
    # Panel b GMUICU: 缓存重算 (终点有效口径, organ=1 循环, h=6)
    c = np.load(v4 / "traj_mae_ci_cache.npz", allow_pickle=True)
    t = np.load(v4 / "transformertcr_trajectory_cache.npz", allow_pickle=True)
    organ, omask = c["organ_lab"] if "organ_lab" in c else c["organ"], c["organ_mask"]
    msk = omask[:, 6, 1] > 0
    cv_gmu = (
        float(np.abs(organ[msk, 0, 1] - organ[msk, 6, 1]).mean()),
        float(np.abs(t["pred"][msk, 5, 1] - organ[msk, 6, 1]).mean()),
        float(np.abs(c["pred"][msk, 5, 1] - organ[msk, 6, 1]).mean()),
    )
    # Panel c
    ot = json.load(open(v4 / "ALL_RESULTS_SUMMARY.json", encoding="utf-8"))["ot_paired"]
    mc6 = json.load(open(rm / "mimic_3seed_corrected.json", encoding="utf-8"))["careoff_paired"]["6h"]
    # Panel d
    pd_g = json.load(open(v4 / "frozen_patient_specific_r_v2.json",
                          encoding="utf-8"))["bootstrap_seed42"]["shuffled"]
    pd_m = json.load(open(rm / "frozen_mimic_patient_specific.json",
                          encoding="utf-8"))["bootstrap_seed42"]["shuffled"]
    return dict(
        gmu_all=(a_gmu_all["point"], a_gmu_all["ci_lo"], a_gmu_all["ci_hi"]),
        mimic_all=(a_mimic_all["point"], a_mimic_all["ci"][0], a_mimic_all["ci"][1]),
        gmu_chg=(a_gmu_chg["point"], a_gmu_chg["ci_lo"], a_gmu_chg["ci_hi"]),
        mimic_chg=(a_mimic_chg["point"], a_mimic_chg["ci"][0], a_mimic_chg["ci"][1]),
        gmu_cv=cv_gmu,
        mimic_cv=EXPECTED["mimic_cv"],  # 正文 P107 锁定 (无独立 frozen json)
        gmu_c=(ot["delta_auprc"], ot["delta_auprc_lo"], ot["delta_auprc_hi"]),
        mimic_c=(mc6["delta_auprc"], mc6["delta_auprc_ci"][0], mc6["delta_auprc_ci"][1]),
        gmu_d=(pd_g["delta_auprc_point"], pd_g["delta_auprc_ci"][0], pd_g["delta_auprc_ci"][1]),
        mimic_d=(pd_m["delta_auprc_point"], pd_m["delta_auprc_ci"][0], pd_m["delta_auprc_ci"][1]),
    )


def assert_all(D):
    for k, exp in EXPECTED.items():
        got = D[k]
        for gv, ev in zip(got, exp):
            assert abs(gv - ev) <= 0.001, f"{k}: {gv} vs {ev}"
    print("断言全部通过: 绘图值与跨队列对照表、正文 P106-P108 及 frozen 文件一致")


# ══════════ 绘图 ══════════
def forest(ax, rows, color, xlim, xticks, xlab, hollow=()):
    """rows = [(y, (point, lo, hi), cohort_marker, label), ...]"""
    ax.axvline(0, color="#9AA0A6", linewidth=0.7, linestyle="--", zorder=1)
    for y, (pt, lo, hi), mk, hollow_flag in rows:
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="none",
                    ecolor=color, elinewidth=1.1, capsize=2.2, zorder=2)
        ax.plot([pt], [y], mk, ms=5,
                mfc=("white" if hollow_flag else color), mec=color,
                mew=(0.9 if hollow_flag else 0.5), zorder=3)
        ax.text(hi + (xlim[1] - xlim[0]) * 0.025, y, f"+{pt:.3f}",
                ha="left", va="center", fontsize=5.8)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlab)


def main():
    D = load_data()
    assert_all(D)

    fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.93, bottom=0.145,
                          wspace=0.32, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Panel a: PLF−TR 轨迹代价, 主分析实心 / |ΔSOFA|≥1 空心 ──
    style_axis(ax_a, ygrid=False)
    ax_a.xaxis.grid(True, color="#E9ECEF", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    rows_a = [
        (3.45, D["gmu_all"], "o", False), (2.95, D["mimic_all"], "s", False),
        (1.45, D["gmu_chg"], "o", True), (0.95, D["mimic_chg"], "s", True),
    ]
    forest(ax_a, rows_a, COL_PLF, (-0.03, 0.42), [0, 0.1, 0.2, 0.3, 0.4],
           "ΔMAE（PLF-OGT - Transformer）")
    ax_b_line_y = 2.2
    ax_a.axhline(ax_b_line_y, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax_a.text(-0.025, 3.82, "全部锚点", ha="left", va="center", fontsize=6.5, color=C_TEXT_D)
    ax_a.text(-0.025, 1.82, "|ΔSOFA|≥1 锚点", ha="left", va="center", fontsize=6.5, color=C_TEXT_D)
    ax_a.set_yticks([3.45, 2.95, 1.45, 0.95])
    ax_a.set_yticklabels(["GMUICU", "MIMIC-IV", "GMUICU", "MIMIC-IV"])
    ax_a.set_ylim(0.4, 4.1)
    ax_a.text(0.985, 0.965, "正值表示 PLF-OGT 误差更高", transform=ax_a.transAxes,
              ha="right", va="top", fontsize=6, color="#555555")

    # ── Panel b: 循环分量模型排序 (两队列 × 三模型 dot plot) ──
    style_axis(ax_b, ygrid=True)
    model_cols = [COL_STATIC, COL_TRANS, COL_PLF]
    for gi, (cv_g, cv_m) in enumerate(zip(D["gmu_cv"], D["mimic_cv"])):
        for xi, v in [(0, cv_g), (1, cv_m)]:
            ax_b.plot(xi + (gi - 1) * 0.16, v, "o", ms=6, mfc=model_cols[gi],
                      mec=model_cols[gi], zorder=3)
            ax_b.text(xi + (gi - 1) * 0.16, v + 0.012, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=5.8)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["GMUICU", "MIMIC-IV"])
    ax_b.set_xlim(-0.5, 1.5)
    ax_b.set_ylim(0, 0.50)
    ax_b.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_b.set_ylabel("6 h 循环分量 MAE")
    handles_b = [plt.Line2D([0], [0], marker="o", color="none", mfc=c, mec=c, ms=5)
                 for c in model_cols]
    ax_b.legend(handles_b, ["静态状态延续对照", "Transformer", "PLF-OGT"],
                loc="upper right", frameon=False, fontsize=5.5,
                handletextpad=0.15, borderaxespad=0.1, labelspacing=0.25)

    # ── Panel c: 治疗路径信息增益 ──
    style_axis(ax_c, ygrid=False)
    ax_c.xaxis.grid(True, color="#E9ECEF", linewidth=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    forest(ax_c, [(1.42, D["gmu_c"], "o", False), (0.72, D["mimic_c"], "s", False)],
           COL_PLF, (-0.02, 0.46), [0, 0.1, 0.2, 0.3, 0.4],
           "ΔAUPRC（TCR - care-off）")
    ax_c.set_yticks([1.42, 0.72])
    ax_c.set_yticklabels(["GMUICU", "MIMIC-IV"])
    ax_c.set_ylim(0.2, 1.9)

    # ── Panel d: 残差患者对应关系 ──
    style_axis(ax_d, ygrid=False)
    ax_d.xaxis.grid(True, color="#E9ECEF", linewidth=0.4, zorder=0)
    ax_d.set_axisbelow(True)
    forest(ax_d, [(1.42, D["gmu_d"], "o", False), (0.72, D["mimic_d"], "s", False)],
           R_PURPLE, (-0.01, 0.21), [0, 0.05, 0.10, 0.15, 0.20],
           "ΔAUPRC（自身状态 - 跨患者替换）")
    ax_d.set_yticks([1.42, 0.72])
    ax_d.set_yticklabels(["GMUICU", "MIMIC-IV"])
    ax_d.set_ylim(0.2, 1.9)

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        panel_label(fig, ax, letter)

    # 底部 cohort 图例 (marker 形状)
    handles = [plt.Line2D([0], [0], marker="o", color="none", mfc=C_TEXT_D,
                          mec=C_TEXT_D, ms=5.5),
               plt.Line2D([0], [0], marker="s", color="none", mfc=C_TEXT_D,
                          mec=C_TEXT_D, ms=5.5)]
    fig.legend(handles, ["GMUICU", "MIMIC-IV"], loc="lower center", ncol=2,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.015),
               handlelength=1.0, columnspacing=3.0)

    save_figure(fig, str(Path(__file__).parent / "fig5"))
    plt.close(fig)

    # ── 数据存档 ──
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Panel", "项目", "点估计", "95% CI 下限", "95% CI 上限", "来源"])
    for k, lbl in [("gmu_all", "a GMUICU 全部锚点 ΔMAE"), ("mimic_all", "a MIMIC-IV 全部锚点 ΔMAE"),
                   ("gmu_chg", "a GMUICU |ΔSOFA|≥1 ΔMAE"), ("mimic_chg", "a MIMIC-IV |ΔSOFA|≥1 ΔMAE"),
                   ("gmu_c", "c GMUICU ΔAUPRC(TCR-care-off)"), ("mimic_c", "c MIMIC-IV ΔAUPRC(TCR-care-off)"),
                   ("gmu_d", "d GMUICU ΔAUPRC(自身-跨患者)"), ("mimic_d", "d MIMIC-IV ΔAUPRC(自身-跨患者)")]:
        w.writerow([lbl.split(" ")[0], lbl.split(" ", 1)[1], round(D[k][0], 4),
                    round(D[k][1], 4), round(D[k][2], 4),
                    "frozen json (见脚本 docstring)"])
    for coh, key in [("GMUICU", "gmu_cv"), ("MIMIC-IV", "mimic_cv")]:
        src = "traj 缓存重算(终点有效口径)" if coh == "GMUICU" else "正文 P107 锁定"
        w.writerow(["b", f"{coh} 循环MAE 静态/Transformer/PLF",
                    "/".join(f"{v:.3f}" for v in D[key]), "", "", src])
    write_safe(Path(__file__).parent / "fig5_data_used.csv", buf.getvalue().encode("utf-8-sig"))

    # ── 中文图注 ──
    caption = (
        "图 5｜跨队列独立再开发重现部分方向性结果。\n"
        "a，PLF-OGT 相对标准 Transformer 的 6 h 总 SOFA 轨迹 MAE 配对差异。正值表示 PLF-OGT 误差更高；"
        "在全部可评价锚点（实心标记）与 |ΔSOFA|≥1 锚点（空心标记）的敏感性分析中，两个队列的差异方向一致且均为正，"
        "但 MIMIC-IV 中差异幅度较小。\n"
        "b，两队列 6 h 循环分量 MAE。GMUICU 和 MIMIC-IV 中，两种学习模型均优于静态状态延续对照；"
        "但 Transformer 与 PLF-OGT 之间的相对排序存在队列差异。\n"
        "c，TCR 相对 care-off 的 6 h AUPRC 配对差异。两个队列均显示锚点后实际治疗路径提供附加判别信息，"
        "但效应大小不同。\n"
        "d，当前患者自身残差状态相对跨患者替换的 6 h AUPRC 配对差异。两个队列中差异均为正，"
        "支持残差状态对正确患者对应关系的依赖得到方向性重现。"
        "a、c 和 d 中误差线表示患者级配对 cluster bootstrap 95% CI；b 为点估计。"
        "MIMIC-IV 中所有模型均使用本地数据独立重新开发，因此本图反映预设模型与分析流程在另一数据库中的重新建立，"
        "而非 GMUICU 冻结模型权重的直接外部验证。"
    )
    write_safe(Path(__file__).parent / "fig5_caption_cn.txt", caption.encode("utf-8"))

    print("输出: fig5.pdf / fig5.png / fig5_caption_cn.txt / fig5_data_used.csv")


if __name__ == "__main__":
    main()
