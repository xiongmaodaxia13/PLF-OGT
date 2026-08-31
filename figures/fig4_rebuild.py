#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 4 重构 — 概念-残差潜在状态核查 (2026-08-29 定稿方案, 适配版配色).

图 4｜概念—残差潜在状态核查揭示互补预测信息与语义边界
Panel a 概念锚定必要性 / b S/R 互补信息 / c 残差患者对应关系 / d 重叠与跨训练稳定性

配色 (适配版, 与图 2/3 体系衔接): 金 #F1A93B = 完整状态 S+R (PLF-OGT 身份色);
青绿 #00A087 = 概念状态 S; 蓝紫 #8891DB = 残差状态 R; 灰 #A8ACB9 = 移除/负控。

数据锁定 (容差 0.001), 全部来自 frozen 文件:
- Panel a 左: full_proxy_3seed.json (0.068±0.010) + no_anchor_3seed.json (0.702±0.005, 重训 3-seed)
- Panel a 右: bidirectional_probe.json random_mapping_control (0.094/1.086, 开发阶段负控)
- Panel b: Supplementary_Data_1.xlsx「逐次训练与Shapley」3-seed 中位数 = 正文 P100
- Panel c: frozen_patient_specific_r_v2.json summary 均值 (3-seed; matched/mean/query_only/shuffled)
- Panel d1: bidirectional_probe.json max_r2; d2: ALL_RESULTS_SUMMARY.json hungarian improvement
口径注意: 0.584/0.567 为逐次口径, 与表 2 ensemble 0.610 不可直接比较 (表 4 注)。
输出: fig4.pdf / fig4.png(300dpi) / fig4_caption_cn.txt / fig4_data_used.csv
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig2_rebuild import (COL_PLF, COL_STATIC, style_axis, panel_label,
                          save_figure, write_safe)  # style block 经 import 生效
from fig3_rebuild import lighten, GRAY_EDGE, C_CONNECT

S_GREEN = "#00A087"                    # 概念状态 S (用户方案保留)
R_PURPLE = "#8891DB"                   # 残差状态 R (避开图 2 Transformer 深蓝)
R_MID = lighten(R_PURPLE, 0.30)        # Panel c 人群平均
R_LIGHT = lighten(R_PURPLE, 0.58)      # Panel c 清除残差
GRAY = COL_STATIC                      # 移除/负控 = 既定参照灰

# ══════════ 锁定值 (正文 P97-P102 / 表 4) ══════════
EXPECTED = {
    "a_full": 0.068, "a_noanchor": 0.702,     # 重训 3-seed 均值
    "a_correct": 0.094, "a_random": 1.086,    # 开发阶段负控
    "b": [0.584, 0.427, 0.518, 0.236],        # 3-seed 中位数 (S+R/S/R/均移除)
    "c_auprc": [0.567, 0.466, 0.423, 0.410],  # 3-seed 均值 (自身/人群平均/清除/跨患者)
    "c_macro": [0.315, 0.756, 0.746, 0.928],
    "d_r2": (0.876, 0.561),                   # R→S, S→R
    "d_match": (0.041, 0.019),                # 42vs52, 42vs62
}
COND_CN = ["自身状态", "人群平均", "清除残差", "跨患者替换"]


def load_data(base=Path(r"F:/MIMIC3_1/V13/results/v4"),
              xlsx=Path(r"F:/MIMIC3_1/V13/manuscript/Supplementary_Data_1.xlsx")):
    full = json.load(open(base / "full_proxy_3seed.json", encoding="utf-8"))
    noanc = json.load(open(base / "no_anchor_3seed.json", encoding="utf-8"))
    probe = json.load(open(base / "bidirectional_probe.json", encoding="utf-8"))
    ps = json.load(open(base / "frozen_patient_specific_r_v2.json", encoding="utf-8"))["summary"]
    summ = json.load(open(base / "ALL_RESULTS_SUMMARY.json", encoding="utf-8"))
    hung = summ["input_mask_hung"]["hungarian"]
    # Panel b: xlsx 逐次训练 3-seed 中位数
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb["逐次训练与Shapley"]
    rows = list(ws.iter_rows(min_row=2, max_row=5, values_only=True))
    keymap = {"S+R 均开放": 0, "仅语义开放": 1, "仅残差开放": 2, "S+R 均关闭": 3}
    seeds = [[], [], [], []]
    for r in rows:
        if r[0] in keymap:
            seeds[keymap[r[0]]] = [float(v) for v in r[1:4]]
    wb.close()
    b_med = [median(s) for s in seeds]
    c = {k: ps[k] for k in ["matched", "mean", "query_only", "shuffled"]}
    return dict(
        a_full=full["mean"], a_noanchor=noanc["mean"],
        a_correct=probe["random_mapping_control"]["correct_mae"],
        a_random=probe["random_mapping_control"]["random_mae"],
        b=b_med,
        c_auprc=[c[k]["auprc_mean"] for k in ["matched", "mean", "query_only", "shuffled"]],
        c_macro=[c[k]["macro_mae_mean"] for k in ["matched", "mean", "query_only", "shuffled"]],
        d_r2=(probe["r_to_s_probe"]["max_r2"], probe["s_to_r_probe"]["max_r2"]),
        d_match=(hung["42_vs_52"]["improvement"], hung["42_vs_62"]["improvement"]),
        b_seeds=seeds,
    )


def assert_all(D):
    tol = 0.001
    for k in ["a_full", "a_noanchor", "a_correct", "a_random"]:
        assert abs(D[k] - EXPECTED[k]) <= tol, f"{k}: {D[k]}"
    for got, exp in zip(D["b"], EXPECTED["b"]):
        assert abs(got - exp) <= tol, f"b: {got} vs {exp}"
    for key in ["c_auprc", "c_macro"]:
        for got, exp in zip(D[key], EXPECTED[key]):
            assert abs(got - exp) <= tol, f"{key}: {got} vs {exp}"
    for got, exp in zip(D["d_r2"], EXPECTED["d_r2"]):
        assert abs(got - exp) <= tol, f"d_r2: {got} vs {exp}"
    for got, exp in zip(D["d_match"], EXPECTED["d_match"]):
        assert abs(got - exp) <= tol, f"d_match: {got} vs {exp}"
    print("断言全部通过: 绘图值与正文 P97-P102、表 4 及 frozen 文件一致")


# ══════════ 绘图 ══════════
def main():
    D = load_data()
    assert_all(D)

    fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.93, bottom=0.145,
                          wspace=0.32, hspace=0.42)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.85)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    gs_c = gs[1, 0].subgridspec(2, 1, hspace=0.62)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.75)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    # ── Panel a: 双单元哑铃 (y 0-1.20 统一, 线性轴) ──
    for ax, good, bad, ttl in [
        (ax_a1, D["a_full"], D["a_noanchor"], "移除概念锚定（重新训练）"),
        (ax_a2, D["a_correct"], D["a_random"], "随机代理重映射（开发阶段负控）"),
    ]:
        style_axis(ax, ygrid=True)
        ax.plot([0, 1], [good, bad], color=C_CONNECT, linewidth=1.0, zorder=1)
        ax.plot([0], [good], "o", ms=5.5, mfc=S_GREEN, mec=S_GREEN, zorder=2)
        ax.plot([1], [bad], "o", ms=5.5, mfc=GRAY, mec=GRAY_EDGE, mew=0.8, zorder=2)
        for x_, v in [(0, good), (1, bad)]:
            ax.text(x_, v + 0.045, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["完整锚定", "移除锚定"] if ax is ax_a1 else ["正确映射", "随机重映射"])
        ax.set_ylim(0, 1.20)
        ax.set_yticks([0, 0.4, 0.8, 1.2])
        ax.set_title(ttl, fontsize=6.5, color="#555555", pad=2.5)
    ax_a1.set_ylabel("代理恢复 MAE")
    ax_a1.text(0.03, 0.97, "11/12 个主要代理恢复 MAE <0.10\n胆红素为主要例外",
               transform=ax_a1.transAxes, ha="left", va="top",
               fontsize=5.5, color="#555555")

    # ── Panel b: 固定模型信息移除, 4 柱 ──
    style_axis(ax_b, ygrid=True)
    xb = np.arange(4)
    hb = D["b"]
    bcols = [COL_PLF, S_GREEN, R_PURPLE, GRAY]
    bedges = [COL_PLF, S_GREEN, R_PURPLE, GRAY_EDGE]
    ax_b.bar(xb, hb, width=0.52, color=bcols, edgecolor=bedges, linewidth=0.5)
    for x_, v in zip(xb, hb):
        ax_b.text(x_, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["S+R", "仅 S", "仅 R", "均移除"])
    ax_b.set_ylim(0, 0.65)
    ax_b.set_yticks([0, 0.2, 0.4, 0.6])
    ax_b.set_ylabel("6 h AUPRC")
    ax_b.text(0.98, 0.955, "固定模型信息移除", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=6, color="#555555")

    # ── Panel c: 残差患者对应 (上下两单元横向条, 共享条件轴) ──
    ypos = np.arange(4)[::-1]  # 自身在上 → 跨患者在下
    ccols = [R_PURPLE, R_MID, R_LIGHT, GRAY]
    cedges = [R_PURPLE, R_PURPLE, R_PURPLE, GRAY_EDGE]
    for ax, vals, xlim, xticks, xlab, ylab_on in [
        (ax_c1, D["c_auprc"], (0, 0.68), [0, 0.2, 0.4, 0.6], "6 h AUPRC", True),
        (ax_c2, D["c_macro"], (0, 1.10), [0, 0.5, 1.0], "6 h 器官 macro-MAE", False),
    ]:
        style_axis(ax, ygrid=False)
        ax.xaxis.grid(True, color="#E9ECEF", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.barh(ypos, vals, height=0.55, color=ccols, edgecolor=cedges, linewidth=0.5)
        off = 0.012 if xlim[1] < 1 else 0.02
        for y, v in zip(ypos, vals):
            ax.text(v + off, y, f"{v:.3f}", ha="left", va="center", fontsize=5.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(COND_CN if ylab_on else [""] * 4)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlab)

    # ── Panel d: 重叠 + 跨训练稳定性 (双单元横向条) ──
    for ax, vals, cols, edges, ylabs, xlim, xticks, xlab, ttl in [
        (ax_d1, list(D["d_r2"]), [R_PURPLE, S_GREEN], [R_PURPLE, S_GREEN],
         ["R → S", "S → R"], (0, 1.0), [0, 0.5, 1.0], "最大线性探针 R²", "S/R 信息重叠"),
        (ax_d2, list(D["d_match"]), [R_PURPLE, R_PURPLE], [R_PURPLE, R_PURPLE],
         ["比较 1", "比较 2"], (0, 0.05), [0, 0.02, 0.04], "重新匹配距离改善", "跨训练槽稳定性"),
    ]:
        style_axis(ax, ygrid=False)
        ax.xaxis.grid(True, color="#E9ECEF", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        yd = [1, 0]
        ax.barh(yd, vals, height=0.5, color=cols, edgecolor=edges, linewidth=0.5)
        off = 0.015 if xlim[1] <= 0.05 else 0.02
        for y, v in zip(yd, vals):
            ax.text(v + off, y, f"{v:.3f}", ha="left", va="center", fontsize=5.8)
        ax.set_yticks(yd)
        ax.set_yticklabels(ylabs)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlab)
        ax.set_title(ttl, fontsize=6.5, color="#555555", pad=2.5)

    for ax, letter in zip([ax_a1, ax_b, ax_c1, ax_d1], ["a", "b", "c", "d"]):
        panel_label(fig, ax, letter)

    # 底部四项语义图例
    handles = [plt.Line2D([0], [0], marker="s", color="none", mfc=c, mec=e, ms=6)
               for c, e in [(S_GREEN, S_GREEN), (R_PURPLE, R_PURPLE),
                            (COL_PLF, COL_PLF), (GRAY, GRAY_EDGE)]]
    fig.legend(handles, ["概念状态 S", "残差状态 R", "完整状态 S+R", "移除或负控"],
               loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.015), handlelength=1.2, columnspacing=2.0)

    save_figure(fig, str(Path(__file__).parent / "fig4"))
    plt.close(fig)

    # ── 数据存档 ──
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Panel", "项目", "数值", "来源"])
    w.writerow(["a 锚定必要性", "完整锚定代理MAE(3-seed均值)", round(D["a_full"], 4), "full_proxy_3seed.json"])
    w.writerow(["a 锚定必要性", "移除锚定(重新训练)代理MAE", round(D["a_noanchor"], 4), "no_anchor_3seed.json"])
    w.writerow(["a 锚定必要性", "正确映射代理MAE(开发阶段)", round(D["a_correct"], 4), "bidirectional_probe.json"])
    w.writerow(["a 锚定必要性", "随机重映射代理MAE(开发阶段)", round(D["a_random"], 4), "bidirectional_probe.json"])
    for lbl, v in zip(["S+R", "仅S", "仅R", "均移除"], D["b"]):
        w.writerow(["b S/R信息移除(3-seed中位数)", lbl, round(v, 4),
                    "Supplementary_Data_1.xlsx 逐次训练与Shapley"])
    for cond, a_, m_ in zip(COND_CN, D["c_auprc"], D["c_macro"]):
        w.writerow(["c 患者特异性(3-seed均值)", f"{cond} AUPRC/macro-MAE",
                    f"{round(a_,4)} / {round(m_,4)}", "frozen_patient_specific_r_v2.json"])
    w.writerow(["d 信息重叠", "R→S / S→R 最大线性探针R²",
                f"{D['d_r2'][0]:.4f} / {D['d_r2'][1]:.4f}", "bidirectional_probe.json"])
    w.writerow(["d 跨训练槽稳定性", "比较1(42vs52) / 比较2(42vs62) 重新匹配改善",
                f"{D['d_match'][0]:.4f} / {D['d_match'][1]:.4f}", "ALL_RESULTS_SUMMARY.json hungarian"])
    write_safe(Path(__file__).parent / "fig4_data_used.csv", buf.getvalue().encode("utf-8-sig"))

    # ── 中文图注 ──
    caption = (
        "图 4｜概念—残差潜在状态核查揭示互补预测信息与语义边界。\n"
        "a，概念状态的代理锚定核查。左：在完整独立测试集的三次独立训练中，移除概念锚定（重新训练）后"
        "代理恢复 MAE 由 0.068 增至 0.702；右：开发阶段随机重映射负控中，随机改变概念—代理对应关系"
        "使恢复 MAE 由 0.094 增至 1.086。12 个主要代理中 11 个的恢复 MAE<0.10（胆红素 0.357 为主要例外），"
        "支持概念状态的可核查性依赖预定义临床代理结构。\n"
        "b，固定已训练模型的信息移除分析（三次训练逐次估计的中位数）：同时保留概念与残差状态时 6 h AUPRC 最高，"
        "仅保留其中一类时下降，两类均移除时进一步下降，支持两类状态提供互补预测信息。"
        "该逐次口径与表 2 的 3-seed ensemble 主分析（0.610）不可直接比较，口径对照见表 4 注。\n"
        "c，残差状态患者对应关系负控（三次训练均值；自身状态对应 matched，跨患者替换对应 shuffled）。"
        "使用患者自身残差状态时判别与轨迹表现最佳；人群平均状态、清除残差内容或跨患者替换均导致性能下降，"
        "其中跨患者替换损害最明显。固定模型配对 bootstrap 中 matched 相对 shuffled 的 ΔAUPRC 为"
        "+0.144（95% CI +0.119 至 +0.171）。该口径与表 2 ensemble 主分析不可直接比较。\n"
        "d，潜在状态重叠与跨训练稳定性。概念与残差状态之间存在较高的双向线性可预测性（最大线性探针 R²）；"
        "允许不同独立训练之间重新匹配残差槽位后，对应关系的改善仍有限（0.041 与 0.019），"
        "提示患者特异信息并不等同于单个残差槽具有稳定、可识别的临床语义。"
        "胆红素代理恢复和灌注/休克概念证据富集为主要失败边界，完整结果见补充表 S5。"
    )
    write_safe(Path(__file__).parent / "fig4_caption_cn.txt", caption.encode("utf-8"))

    print("输出: fig4.pdf / fig4.png / fig4_caption_cn.txt / fig4_data_used.csv")


if __name__ == "__main__":
    main()
