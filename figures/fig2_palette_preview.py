#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 2 配色候选预览 — 三套方案渲染到真实 Panel a/b 数据上供用户选择.

行 = 方案 (A 玫瑰编辑风 / B 青蓝琥珀 / C 庄重高对比), 列 = Panel a 折线 + Panel b 分组柱.
输出: fig2_palette_preview.png (不改动正式 fig2 系列, 选定后再改 fig2_rebuild.py)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig2_rebuild import (HORIZONS, ORGANS_CN, TOTAL, load_data, style_axis,
                          COL_GRID)

CANDS = [
    ("方案①｜博客三色-8：深蓝-金橙-灰（首推）",
     "静态 #A8ACB9　Transformer #2B6688　PLF-OGT #F1A93B",
     ["#A8ACB9", "#2B6688", "#F1A93B"]),
    ("方案②｜博客双色-8 红/青 + 三色-8 的灰",
     "静态 #A8ACB9　Transformer #339DB5　PLF-OGT #C9352B",
     ["#A8ACB9", "#339DB5", "#C9352B"]),
    ("方案③｜博客三色-4：淡雅灰-蓝-金",
     "静态 #828D93　Transformer #78C2E0　PLF-OGT #EDB176",
     ["#828D93", "#78C2E0", "#EDB176"]),
]
HIGHLIGHT = "#EFF2F4"


def main():
    panel_b, _ = load_data()
    fig, axes = plt.subplots(3, 2, figsize=(150 / 25.4, 210 / 25.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.075,
                        wspace=0.40, hspace=0.62)
    x = np.arange(6)
    bw = 0.26
    for r, (title, hexes, cols) in enumerate(CANDS):
        axl, axr = axes[r]
        # 左: Panel a 折线
        style_axis(axl, ygrid=True)
        for key, col in zip(["static", "tr", "plf"], cols):
            axl.plot(HORIZONS, TOTAL[key], "-o", color=col, linewidth=1.6,
                     markersize=4.0)
        axl.set_xticks(HORIZONS)
        axl.set_ylim(0, 1.15)
        axl.set_ylabel("总 SOFA MAE")
        axl.set_title(f"{title}\n{hexes}", loc="left", fontsize=9,
                      fontweight="bold", pad=10)
        # 右: Panel b 分组柱 (无数值标签, 只看配色)
        style_axis(axr, ygrid=True)
        for gi, (key, col) in enumerate(zip(["static", "tr", "plf"], cols)):
            axr.bar(x + (gi - 1) * bw, panel_b[key], width=bw, color=col,
                    edgecolor=col, linewidth=0.3)
        axr.axvspan(x[1] - 0.45, x[1] + 0.45, color=HIGHLIGHT, zorder=0)
        axr.set_xticks(x)
        axr.set_xticklabels(ORGANS_CN)
        axr.set_ylim(0, 0.95)
        axr.set_ylabel("6 h 器官分量 MAE")

    fig.text(0.5, 0.965, "Figure 2 配色候选（取自论文绘图配色博客，渲染于真实数据）",
             ha="center", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.012,
             "各方案内颜色含义固定：灰 = 静态状态延续对照 · 第二色 = Transformer · 第三色 = PLF-OGT",
             ha="center", fontsize=7.5, color="#555555")
    out = str(__import__("pathlib").Path(__file__).parent / "fig2_palette_preview.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("输出:", out)


if __name__ == "__main__":
    main()
