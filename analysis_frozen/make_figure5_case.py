#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图 5 ｜ 病例级解释（桑基图 + 三部分分解表）.

2 例 × 2 条件 = 4 格，每格上排桑基图(当前 SOFA → [concept|residual|baseline] → 预测 SOFA)
下排彩色三部分分解表(6 器官 × 当前/6h预测/可解释/未阐释/基线/真实).
  - 行 = 患者：上 stay_00328(改善)，下 stay_01864(恶化)
  - 列 = 条件：左 TCR(按医嘱)，右 OLP(无干预)

桑基图几何移植自 V12/scripts/render_v4_reasoning_report.py::sankey_chart().
数据源：从 *_research.html 解析(含真实结局)；reasoning_cases.json 已过期故不用.
输出：F:/MIMIC3_1/V13/figures/图5_病例级解释.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import numpy as np
import re
from pathlib import Path as PPath

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ORGANS = ["呼吸", "循环", "肾脏", "凝血", "肝脏", "神经"]
REPORT_DIR = PPath(r"F:/MIMIC3_1/V12/results/clinical_reports_v4")
OUT = PPath(r"F:/MIMIC3_1/V13/figures/图5_病例级解释.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 桑基源色（与报告一致）
C_CONCEPT = "#1b5e20"   # 可解释（绿系）
C_RESID = "#7b1fa2"     # 未阐释（紫）
C_BASE = "#9e9e9e"      # 先验基线（灰）
C_LEFT = "#64b5f6"      # 当前 SOFA（蓝）
# 表格单元格语义色
CELL_PUSH = "#ffcdd2"   # 推恶化（红底）
CELL_SUPP = "#c8e6c9"   # 抑恶化（绿底）
CELL_ZERO = "#ffffff"
CELL_WARN = "#fff3e0"   # 预测恶化格
CELL_GOOD = "#e8f5e9"   # 预测改善格
TXT_PUSH = "#c62828"
TXT_SUPP = "#2e7d32"
WORSEN_THRESH = 1.0
IMPROVE_THRESH = -1.0


def _f(s):
    return float(s.replace("+", ""))


def parse_case(stay, tau):
    """返回两条件的逐器官行、总分、headline."""
    raw = (REPORT_DIR / f"stay_{stay}_tau_{tau}_research.html").read_text(encoding="utf-8")

    def organ_rows(marker):
        i = raw.find(marker)
        blk = raw[i:i + 4000]
        pat = (r"<td class='label'>([^<]+)</td><td>([\d.]+)</td>"
               r"<td[^>]*>([\d.-]+) \(([\d.+-]+)\)</td>"
               r"<td[^>]*>([\d.+-]+)</td><td[^>]*>([\d.+-]+)</td>"
               r"<td[^>]*>([\d.+-]+)</td><td>([\d.+-]+)</td>")
        rows = []
        for m in re.finditer(pat, blk):
            o, cur, pred, d, c, r, n, real = m.groups()
            rows.append({"organ": o.strip(), "cur": _f(cur), "pred": _f(pred),
                         "delta": _f(d), "concept": _f(c), "resid": _f(r),
                         "nat": _f(n), "real_delta": _f(real)})
            if len(rows) >= 7:
                break
        return rows

    a1 = organ_rows("A-1.")   # TCR
    b1 = organ_rows("B-1.")   # OLP
    return {
        "stay": stay, "tau": tau,
        "organs_tcr": a1[:6], "total_tcr": a1[6],
        "organs_olp": b1[:6], "total_olp": b1[6],
    }


# ---------- 桑基图（移植 sankey_chart）----------
def draw_sankey(ax, cur_sofa, full_delta, shap_s, shap_r, v_empty, title, title_color):
    """当前 SOFA → [concept|residual|baseline] → 预测 SOFA, 流量守恒."""
    end = cur_sofa + full_delta
    nodes = [("可解释 concept", shap_s, C_CONCEPT),
             ("未阐释 residual", shap_r, C_RESID),
             ("先验基线 natural", v_empty, C_BASE)]
    w, h = 460, 250
    pad_l, pad_r, pad_t, pad_b = 24, 24, 26, 50
    col_w = 22
    x_left = pad_l
    x_mid = w / 2 - col_w / 2
    x_right = w - pad_r - col_w
    gap = 8

    left_pts = cur_sofa
    right_pts = end
    mid_total_pts = sum(abs(v) for _, v, _ in nodes) + gap * (len(nodes) - 1) / 10.0
    max_pts = max(left_pts, right_pts, mid_total_pts, 1.0)
    avail_h = h - pad_t - pad_b
    px_per_pt = min(avail_h / max_pts, 16.0)

    left_top = pad_t + (avail_h - left_pts * px_per_pt) / 2
    left_bot = left_top + left_pts * px_per_pt
    right_top = pad_t + (avail_h - right_pts * px_per_pt) / 2
    right_bot = right_top + right_pts * px_per_pt

    mid_nodes_h = sum(abs(v) for _, v, _ in nodes) * px_per_pt + gap * (len(nodes) - 1)
    mid_y = pad_t + (avail_h - mid_nodes_h) / 2
    mid_flows = []
    for name, val, col in nodes:
        ht = abs(val) * px_per_pt
        mid_flows.append((name, val, col, mid_y, mid_y + ht))
        mid_y += ht + gap

    def flow_band(x1, y1a, y1b, x2, y2a, y2b, color):
        mx = (x1 + x2) / 2
        verts = [(x1, y1a), (mx, y1a), (mx, y2a), (x2, y2a),
                 (x2, y2b), (mx, y2b), (mx, y1b), (x1, y1b), (x1, y1a)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                               alpha=0.35, edgecolor="none", lw=0))

    total_abs = sum(abs(v) for _, v, _ in nodes) or 1.0
    cum = left_top
    left_avail = left_bot - left_top
    for name, val, col, y_top, y_bot in mid_flows:
        frac = abs(val) / total_abs
        seg_h = frac * left_avail
        flow_band(x_left + col_w, cum, cum + seg_h, x_mid, y_top, y_bot, col)
        cum += seg_h
    cum = right_top
    right_avail = right_bot - right_top
    for name, val, col, y_top, y_bot in mid_flows:
        frac = abs(val) / total_abs
        seg_h = frac * right_avail
        flow_band(x_mid + col_w, y_top, y_bot, x_right, cum, cum + seg_h, col)
        cum += seg_h

    # 节点矩形 + 标注（SVG 坐标，y 轴反转）
    ax.add_patch(Rectangle((x_left, left_top), col_w, max(left_bot - left_top, 2),
                           facecolor=C_LEFT, edgecolor="none"))
    ax.text(x_left + col_w / 2, left_top - 8, f"{cur_sofa:.0f}", ha="center",
            va="bottom", fontsize=12, weight="bold", color="#1565c0")
    ax.text(x_left + col_w / 2, h - 26, "当前", ha="center", va="top", fontsize=9, color="#555")
    ax.text(x_left + col_w / 2, h - 14, "SOFA", ha="center", va="top", fontsize=9, color="#555")

    for name, val, col, y_top, y_bot in mid_flows:
        ax.add_patch(Rectangle((x_mid, y_top), col_w, max(y_bot - y_top, 2),
                               facecolor=col, edgecolor="none"))
        sign = "+" if val >= 0 else ""
        ax.text(x_mid + col_w + 4, (y_top + y_bot) / 2, f"{name} {sign}{val:.1f}",
                ha="left", va="center", fontsize=8.5, color=col, weight="bold")
    ax.text(x_mid + col_w / 2, pad_t - 8, "贡献来源", ha="center", va="bottom",
            fontsize=9, color="#888")

    end_col = "#c62828" if full_delta >= WORSEN_THRESH else (
        "#43a047" if full_delta <= IMPROVE_THRESH else "#fb8c00")
    ax.add_patch(Rectangle((x_right, right_top), col_w, max(right_bot - right_top, 2),
                           facecolor=end_col, edgecolor="none"))
    ax.text(x_right + col_w / 2, right_top - 8, f"{end:.1f}", ha="center",
            va="bottom", fontsize=12, weight="bold", color=end_col)
    ax.text(x_right + col_w / 2, h - 26, "6h 后", ha="center", va="top", fontsize=9, color="#555")
    ax.text(x_right + col_w / 2, h - 14, "预测", ha="center", va="top", fontsize=9, color="#555")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # 反转 y 匹配 SVG
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", loc="left", color=title_color, pad=4)


# ---------- 彩色三部分分解表 ----------
def _cell_color(v, push_thresh=0.05):
    if v > push_thresh:
        return CELL_PUSH, TXT_PUSH
    if v < -push_thresh:
        return CELL_SUPP, TXT_SUPP
    return CELL_ZERO, "#555"


def draw_table(ax, organs, total, title, title_color):
    cols = ["器官", "当前", "6h预测(Δ)", "可解释", "未阐释", "基线", "真实"]
    cell_text = []
    cell_colors = []
    for r in organs:
        pred_str = f"{r['pred']:.1f} ({r['delta']:+.1f})"
        real_str = f"{r['real_delta']:+.1f}"
        c1, t1 = _cell_color(r["delta"])
        c2, t2 = _cell_color(r["concept"])
        c3, t3 = _cell_color(r["resid"])
        c4, t4 = _cell_color(r["nat"])
        c5, t5 = _cell_color(r["real_delta"])
        row = [r["organ"], f"{r['cur']:.0f}", pred_str,
               f"{r['concept']:+.1f}", f"{r['resid']:+.1f}",
               f"{r['nat']:+.1f}", real_str]
        cell_text.append(row)
        cell_colors.append(["#f5f5f5", "#ffffff", c1, c2, c3, c4, c5])
    # 总分行
    tt = total
    c_tot_pred = CELL_WARN if tt["delta"] > 0 else (CELL_GOOD if tt["delta"] < 0 else CELL_ZERO)
    cell_text.append(["总分", f"{tt['cur']:.0f}", f"{tt['pred']:.1f} ({tt['delta']:+.1f})",
                      f"{tt['concept']:+.1f}", f"{tt['resid']:+.1f}",
                      f"{tt['nat']:+.1f}", f"{tt['real_delta']:+.1f}"])
    cell_colors.append(["#e8f5e9", "#e8f5e9", c_tot_pred] +
                       [_cell_color(tt[k])[0] for k in ["concept", "resid", "nat", "real_delta"]])

    tbl = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center",
                   colColours=["#37474f"] * len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.32)

    # 表头样式
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_text_props(color="white", weight="bold", fontsize=8.2)
        c.set_edgecolor("#37474f")
    # 单元格底色 + 文字色
    n_rows = len(cell_text)
    for i in range(n_rows):
        for j in range(len(cols)):
            c = tbl[i + 1, j]
            c.set_facecolor(cell_colors[i][j])
            c.set_edgecolor("#e0e0e0")
            v = cell_text[i][j]
            # 数字格按符号着文字
            if j >= 2 and j != 2:
                try:
                    num = _f(v)
                    c.set_text_props(color=TXT_PUSH if num > 0.05 else (TXT_SUPP if num < -0.05 else "#555"),
                                     weight="bold" if i == n_rows - 1 else "normal")
                except ValueError:
                    pass
            if i == n_rows - 1:
                c.set_text_props(weight="bold")
            if j == 0:
                c.set_text_props(ha="left")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", loc="left", color=title_color, pad=6)
    # 图注
    ax.text(0.5, -0.02, "红=推恶化　绿=抑恶化（与报告同配色）　|　三列之和=变化量（守恒）",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#777", style="italic")


def main():
    cases = [parse_case("00328", "849"), parse_case("01864", "306")]
    patient_meta = [
        ("stay_00328 ｜ 改善（真实 6h ΔSOFA -3.0）｜ 方向一致（命中）", "#137333"),
        ("stay_01864 ｜ 恶化（真实 6h ΔSOFA +2.0）｜ 方向不一致（失误）", "#D93025"),
    ]
    cond_meta = [
        ("TCR ｜ 按目前医嘱执行", "#1b5e20"),
        ("OLP ｜ 无进一步干预", "#e65100"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(9.2, 12.4),
                             gridspec_kw={"height_ratios": [1.15, 1.0, 1.15, 1.0],
                                          "hspace": 0.32, "wspace": 0.08})

    for pi, case in enumerate(cases):
        total = case["total_tcr"] if pi == 0 else case["total_tcr"]
        for ci, (cond_label, cond_col) in enumerate(cond_meta):
            sankey_ax = axes[pi * 2, ci]
            table_ax = axes[pi * 2 + 1, ci]
            if ci == 0:
                tot = case["total_tcr"]; organs = case["organs_tcr"]
            else:
                tot = case["total_olp"]; organs = case["organs_olp"]
            sk_title = f"{'a' if pi==0 else 'c'}.{1 if ci==0 else 2}  {cond_label}"
            tb_title = f"{'b' if pi==0 else 'd'}.{1 if ci==0 else 2}  {cond_label}"
            draw_sankey(sankey_ax, tot["cur"], tot["delta"], tot["concept"],
                        tot["resid"], tot["nat"], sk_title, cond_col)
            draw_table(table_ax, organs, tot, tb_title, cond_col)

    # 患者行标签（左侧）
    for pi, (lbl, col) in enumerate(patient_meta):
        ymid = (pi * 2 + 0.5) / 4 + 0.02
        fig.text(0.012, 0.96 - ymid - 0.46 * pi, lbl, rotation=90, va="center", ha="center",
                 fontsize=10.5, weight="bold", color=col)

    fig.suptitle("图 5｜病例级解释：桑基图 + 三部分分解表（上：stay_00328 改善；下：stay_01864 恶化）",
                 fontsize=13.5, weight="bold", y=0.995)
    plt.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.02)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"保存: {OUT}")
    for c in cases:
        tt, to = c["total_tcr"], c["total_olp"]
        print(f"  stay {c['stay']}: TCR Δ={tt['delta']:+.1f} (S{tt['concept']:+.1f}/R{tt['resid']:+.1f}/N{tt['nat']:+.1f}) | "
              f"OLP Δ={to['delta']:+.1f} (S{to['concept']:+.1f}/R{to['resid']:+.1f}/N{to['nat']:+.1f}) | "
              f"real Δ={tt['real_delta']:+.1f}")


if __name__ == "__main__":
    main()
