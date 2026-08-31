#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 3 重构 — 治疗路径信息的附加贡献主要集中于循环状态 (2026-08-29 定稿方案).

图 3｜锚点后实际治疗路径提供的附加信息主要集中于循环状态
Panel a 总体增益(轨迹+判别双单元) / b 六器官ΔMAE(核心) / c 升压药暴露动态 / d non-CV 边界

口径: V13 正文 zero 模式 care-off (与图 2 / 正文 / 红字版一致)。
数据锁定 (容差 0.001):
- Panel a: allhorizon_ci.json TCR.6h + OLP.6h (care-off 原始输出=OLP); macro 由 npz 重算
- Panel b: traj_mae_ci_cache.npz (TCR) + careoff_zero_organ_cache.npz (zero 口径) = S4 Panel A 源
- Panel c: frozen_roc_decomposition.json G1/G2/G3 = 正文 P91
- Panel d: frozen_noncv_ci.json = S4 Panel B 源
注意: careoff_allhorizon_ci.json / careoff_macro_ci.json / careoff_traj_ci.json 为 carry 口径 (V15), 本图不使用。
输出: fig3.pdf / fig3.png(300dpi) / fig3_caption_cn.txt / fig3_data_used.csv
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

from fig2_rebuild import (COL_PLF, COL_TEXT, COL_GRID, COL_HIGHLIGHT, COL_STATIC,
                          ORGANS_CN, style_axis, panel_label,
                          save_figure, write_safe)  # style block 经 import 生效


# ══════════ 颜色: TCR 主色 = 图 2 PLF-OGT 金橙; care-off = 其浅色实色版 ══════════
def lighten(hexcol: str, f: float) -> str:
    """同色系混白: f=0 原色, f=1 纯白。实色而非透明度, 印刷/PDF 更稳。"""
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(255 * (c + (1 - c) * f)) for c in (r, g, b))


CAREOFF = lighten(COL_PLF, 0.55)   # care-off / 排除循环 的浅色版
C_MID = lighten(COL_PLF, 0.30)     # Panel c 已在治 (≈70% 主色强度)
C_LIGHT = lighten(COL_PLF, 0.58)   # Panel c 未暴露 (≈40% 主色强度, 加主色描边保可见)
C_CONNECT = "#C9CDD1"              # Panel a 两点连线
GRAY_EDGE = "#8A9099"              # 金+灰版中灰色标记的描边

# 两套配色: goldgray = 定稿 (金=治疗信息在场 / 灰=缺席); mono = 纯同色深浅备选
PALETTES = {
    "mono": dict(
        co_fill=CAREOFF, co_edge=COL_PLF,
        c_colors=[COL_PLF, C_MID, C_LIGHT], c_edges=[COL_PLF] * 3,
        b_mode="all_amber",
        d_colors=[COL_PLF, CAREOFF], d_edges=[COL_PLF] * 2),
    "goldgray": dict(
        co_fill=COL_STATIC, co_edge=GRAY_EDGE,
        c_colors=[COL_PLF, C_MID, COL_STATIC], c_edges=[COL_PLF, COL_PLF, GRAY_EDGE],
        b_mode="cv_focus",
        d_colors=[COL_PLF, COL_STATIC], d_edges=[COL_PLF, GRAY_EDGE]),
}

# ══════════ 锁定值 (V13 zero 口径) ══════════
EXPECTED = {
    # Panel a: 正文 P90
    "macro": (0.357, 0.305),          # (care-off, TCR)
    "auprc": (0.238, 0.610),
    "auroc": (0.794, 0.875),          # 图注用
    "sofa_mae": (1.199, 1.002),       # 图注用 (常量, 不重算)
    # Panel b: S4 Panel A (docx 表12)
    "b_tr": [0.815, 0.177, 0.341, 0.131, 0.217, 0.147],
    "b_co": [0.822, 0.428, 0.345, 0.130, 0.218, 0.199],
    "b_delta": [0.006, 0.251, 0.003, -0.001, 0.001, 0.053],  # 未舍入差值
    # Panel c: 正文 P91
    "c_delta": [0.318, 0.121, 0.013],
    "c_n": [9685, 7610, 47456],
    # Panel d: S4 Panel B (docx 表13)
    "d": (0.081, 0.013),
}
CV_SHARE = 0.801  # 循环占六器官 MAE 改善总和比例 (正文 P119, 未舍入口径)


def load_data(base=Path(r"F:/MIMIC3_1/V13/results/v4")):
    """全部绘图值从 frozen 文件读取/重算, 断言区逐项核对."""
    # Panel b + macro: npz 重算 (终点有效口径, 与图 2 Panel b 同法)
    c = np.load(base / "traj_mae_ci_cache.npz", allow_pickle=True)
    z = np.load(base / "careoff_zero_organ_cache.npz", allow_pickle=True)
    organ, omask = c["organ_lab"] if "organ_lab" in c else c["organ"], c["organ_mask"]
    h = 6
    msk = omask[:, h, :]
    preds = {"tr": c["pred"][:, h - 1, :], "co": z["pred"][:, h - 1, :]}
    per_organ = {}
    for k, pr in preds.items():
        per_organ[k] = [
            float(np.abs(pr[msk[:, o] > 0, o] - organ[msk[:, o] > 0, h, o]).mean())
            for o in range(6)
        ]
    macro = {k: float(np.mean(v)) for k, v in per_organ.items()}
    # Panel a 判别: allhorizon_ci.json (care-off 原始输出 = OLP)
    ah = json.load(open(base / "allhorizon_ci.json", encoding="utf-8"))
    disc = {"co": ah["OLP"]["6h"], "tr": ah["TCR"]["6h"]}
    # Panel c: frozen_roc_decomposition.json
    rd = json.load(open(base / "frozen_roc_decomposition.json", encoding="utf-8"))["groups"]
    groups = []
    for gk in ["G1_新启动", "G2_持续", "G3_从未接触"]:
        g = rd[gk]
        groups.append({"n": g["n"], "delta": g["delta_auroc"]})
    # Panel d: frozen_noncv_ci.json
    nc = json.load(open(base / "frozen_noncv_ci.json", encoding="utf-8"))
    d_total = nc["total_check"]["delta_auroc_total"]
    d_noncv = nc["non_cv"]["delta_auroc"]
    return per_organ, macro, disc, groups, d_total, d_noncv


def assert_all(per_organ, macro, disc, groups, d_total, d_noncv):
    tol = 0.001
    assert abs(macro["co"] - EXPECTED["macro"][0]) <= tol, macro
    assert abs(macro["tr"] - EXPECTED["macro"][1]) <= tol, macro
    assert abs(disc["co"]["auprc"] - EXPECTED["auprc"][0]) <= tol, disc["co"]["auprc"]
    assert abs(disc["tr"]["auprc"] - EXPECTED["auprc"][1]) <= tol, disc["tr"]["auprc"]
    assert abs(disc["co"]["auroc"] - EXPECTED["auroc"][0]) <= tol, disc["co"]["auroc"]
    assert abs(disc["tr"]["auroc"] - EXPECTED["auroc"][1]) <= tol, disc["tr"]["auroc"]
    for k in ("tr", "co"):
        for got, exp in zip(per_organ[k], EXPECTED[f"b_{k}"]):
            assert abs(got - exp) <= tol, f"b_{k}: {got} vs {exp}"
    for got, exp in zip([co - tr for co, tr in zip(per_organ["co"], per_organ["tr"])],
                        EXPECTED["b_delta"]):
        assert abs(got - exp) <= tol, f"b_delta: {got} vs {exp}"
    for g, dn, dd in zip(groups, EXPECTED["c_n"], EXPECTED["c_delta"]):
        assert g["n"] == dn, g
        assert abs(g["delta"] - dd) <= tol, g
    assert abs(d_total - EXPECTED["d"][0]) <= tol, d_total
    assert abs(d_noncv - EXPECTED["d"][1]) <= tol, d_noncv
    print("断言全部通过: 绘图值与正文 P90/P91/P119、S4 Panel A/B 及 frozen 文件一致 (zero 口径)")


# ══════════ 绘图 ══════════
def main(palette: str = "goldgray", stem: str = "fig3"):
    per_organ, macro, disc, groups, d_total, d_noncv = load_data()
    assert_all(per_organ, macro, disc, groups, d_total, d_noncv)
    pal = PALETTES[palette]

    fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.93, bottom=0.145,
                          wspace=0.32, hspace=0.42)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.85)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Panel a: 双单元哑铃 (左 轨迹终点 / 右 判别终点), y 轴均从 0 起 ──
    for ax, co, tr, ylim, ylab, ttl, voff in [
        (ax_a1, macro["co"], macro["tr"], 0.45, "6 h 器官 macro-MAE", "主要轨迹终点", 0.020),
        (ax_a2, disc["co"]["auprc"], disc["tr"]["auprc"], 0.70, "6 h 总 SOFA 恶化 AUPRC", "主要判别终点", 0.030),
    ]:
        style_axis(ax, ygrid=True)
        ax.plot([0, 1], [co, tr], color=C_CONNECT, linewidth=1.0, zorder=1)
        ax.plot([0], [co], "o", ms=5.5, mfc=pal["co_fill"], mec=pal["co_edge"], mew=0.8, zorder=2)
        ax.plot([1], [tr], "o", ms=5.5, mfc=COL_PLF, mec=COL_PLF, zorder=2)
        for x_, v in [(0, co), (1, tr)]:
            ax.text(x_, v + voff, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["care-off", "TCR"])
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=6.5, color="#555555", pad=2.5)

    # ── Panel b: 水平棒棒糖, Δ = care-off - TCR, 正值 = TCR 更低 ──
    style_axis(ax_b, ygrid=False)
    ax_b.xaxis.grid(True, color=COL_GRID, linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    deltas = [co - tr for co, tr in zip(per_organ["co"], per_organ["tr"])]
    ypos = np.arange(6)[::-1]  # 呼吸在上, 保持解剖序 (与图 2b 对应)
    ax_b.axhspan(ypos[1] - 0.42, ypos[1] + 0.42, color=COL_HIGHLIGHT, zorder=0)
    for y, d, organ in zip(ypos, deltas, ORGANS_CN):
        col = COL_PLF if (pal["b_mode"] == "all_amber" or organ == "循环") else COL_STATIC
        ax_b.hlines(y, 0, d, color=col, linewidth=1.6, zorder=2)
        ax_b.plot([d], [y], "o", ms=4.5, mfc=col, mec=col, zorder=3)
        if d >= 0:
            ax_b.text(d + 0.008, y, f"{d:+.3f}", ha="left", va="center", fontsize=5.8)
        else:
            ax_b.text(d - 0.008, y, f"{d:+.3f}", ha="right", va="center", fontsize=5.8)
    ax_b.axvline(0, color="#9AA0A6", linewidth=0.7, zorder=1)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(ORGANS_CN)
    ax_b.set_ylim(-0.55, 5.55)
    ax_b.set_xlim(-0.03, 0.30)
    ax_b.set_xticks([0.0, 0.1, 0.2])
    ax_b.set_xlabel("6 h MAE 改善（care-off - TCR）")
    ax_b.text(0.98, 0.955, "正值表示 TCR 误差更低", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=6, color="#555555")

    # ── Panel c: 升压药暴露动态, 同色系深浅 ──
    style_axis(ax_c, ygrid=True)
    xc = np.arange(3)
    hc = [g["delta"] for g in groups]
    ax_c.bar(xc, hc, width=0.52, color=pal["c_colors"],
             edgecolor=pal["c_edges"], linewidth=0.5)
    for x_, v in zip(xc, hc):
        ax_c.text(x_, v + 0.008, f"{v:+.3f}", ha="center", va="bottom", fontsize=6)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([f"锚点后新启动\nn={g['n']:,}" for g in groups])
    ax_c.set_ylim(0, 0.35)
    ax_c.set_yticks([0, 0.1, 0.2, 0.3])
    ax_c.set_ylabel("ΔAUROC（TCR - care-off）")

    # ── Panel d: 排除循环边界 ──
    style_axis(ax_d, ygrid=True)
    xd = np.arange(2)
    hd = [d_total, d_noncv]
    ax_d.bar(xd, hd, width=0.42, color=pal["d_colors"],
             edgecolor=pal["d_edges"], linewidth=0.5)
    for x_, v in zip(xd, hd):
        ax_d.text(x_, v + 0.003, f"{v:+.3f}", ha="center", va="bottom", fontsize=6)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["总 SOFA", "排除循环分量"])
    ax_d.set_ylim(0, 0.10)
    ax_d.set_yticks([0, 0.05, 0.10])
    ax_d.set_ylabel("ΔAUROC（TCR - care-off）")

    for ax, letter in zip([ax_a1, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        panel_label(fig, ax, letter)

    # 底部两项图例 (替代图 2 三模型图例)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", mfc=COL_PLF, mec=COL_PLF, ms=5.5),
        plt.Line2D([0], [0], marker="o", color="none", mfc=pal["co_fill"],
                   mec=pal["co_edge"], ms=5.5),
    ]
    fig.legend(handles, ["TCR（治疗条件重放）", "care-off（关闭治疗条件更新）"],
               loc="lower center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.015), handlelength=1.4, columnspacing=3.0)

    save_figure(fig, str(Path(__file__).parent / stem))
    plt.close(fig)
    if stem != "fig3":
        print(f"输出: {stem}.pdf / {stem}.png (对比版, 不写图注/CSV)")
        return

    # ── 数据存档 ──
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Panel", "项目", "care-off", "TCR", "差值/说明"])
    w.writerow(["a 器官macro-MAE", "6h 终点有效口径", round(macro["co"], 4), round(macro["tr"], 4),
                f"Δ={macro['tr']-macro['co']:+.4f} (TCR 更低)"])
    w.writerow(["a 总SOFA恶化AUPRC", "6h", round(disc["co"]["auprc"], 4), round(disc["tr"]["auprc"], 4),
                f"ΔAUPRC={disc['tr']['auprc']-disc['co']['auprc']:+.4f}"])
    w.writerow(["a 总SOFA MAE(图注)", "6h 正文P90", EXPECTED["sofa_mae"][0], EXPECTED["sofa_mae"][1],
                "常量(正文锁定)"])
    w.writerow(["a AUROC(图注)", "6h", round(disc["co"]["auroc"], 4), round(disc["tr"]["auroc"], 4),
                f"ΔAUROC={d_total:+.4f}"])
    for o, tr_, co_, d_ in zip(ORGANS_CN, per_organ["tr"], per_organ["co"], deltas):
        w.writerow(["b 六器官ΔMAE=S4 Panel A", o, round(co_, 4), round(tr_, 4), f"Δ={d_:+.4f}"])
    w.writerow(["b 循环份额", "六器官MAE改善总和", "", "", f"{CV_SHARE*100:.1f}% (正文P119)"])
    for lbl, g in zip(["锚点后新启动", "锚点时已在治", "未暴露"], groups):
        w.writerow(["c 升压药暴露分层ΔAUROC", f"{lbl} n={g['n']:,}", "", "", f"{g['delta']:+.4f}"])
    w.writerow(["d 判别增益边界", "总 SOFA 恶化", "", "", f"{d_total:+.4f}"])
    w.writerow(["d 判别增益边界", "排除循环分量后恶化", "", "", f"{d_noncv:+.4f}"])
    write_safe(Path(__file__).parent / "fig3_data_used.csv", buf.getvalue().encode("utf-8-sig"))

    # ── 中文图注 ──
    caption = (
        "图 3｜锚点后实际治疗路径提供的附加信息主要集中于循环状态。\n"
        "a，在同一已训练 PLF-OGT 中比较完整治疗条件轨迹重放（TCR）与关闭治疗条件状态更新的 care-off。"
        "TCR 的 6 h 器官 macro-MAE 低于 care-off，而 6 h 总 SOFA 恶化 AUPRC 较高；"
        "总 SOFA MAE（1.199 对 1.002）和 AUROC（0.794 对 0.875）的结果方向一致。\n"
        "b，TCR 相对 care-off 的六器官 6 h MAE 改善，定义为 MAE(care-off)-MAE(TCR)，"
        "正值表示加入锚点后实际治疗路径后轨迹误差降低。改善主要集中于循环分量，其次为中枢神经分量；"
        "循环分量约占六器官 MAE 改善总和的 80.1%（差值基于未舍入原始值计算）。\n"
        "c，按升压药暴露动态分层的 6 h ΔAUROC（TCR - care-off）。TCR 相对 care-off 的判别增益"
        "在锚点后新启动升压药组最大，锚点时已在治组次之，未暴露组较小。"
        "该分析描述同一预测窗内治疗状态与模型信息增益的关联，不用于推断治疗因果效应或确定时间先后。\n"
        "d，排除循环 SOFA 分量后，TCR 相对 care-off 的恶化判别增益明显缩小，"
        "提示治疗路径信息的主要贡献与循环支持状态密切相关。"
        "图中为点估计；相应患者级配对 bootstrap 置信区间和完整敏感性分析见补充表 S4。"
    )
    write_safe(Path(__file__).parent / "fig3_caption_cn.txt", caption.encode("utf-8"))

    print("输出: fig3.pdf / fig3.png / fig3_caption_cn.txt / fig3_data_used.csv")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "goldgray"
    out_stem = sys.argv[2] if len(sys.argv) > 2 else "fig3"
    main(palette=mode, stem=out_stem)