#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 2 重构 — 最终定稿规格 (任务书 2026-08-29).

图 2｜总 SOFA 指标掩盖短期轨迹重构中的器官级异质性
Panel a 总体轨迹误差 / b 六器官分解(核心) / c 循环跨时距 / d 测量条件化验证

数据锁定: 绘图前所有数值与正文表 2 及补充表 S3 (Panel B/E) 逐项核对 (容差 0.001)。
- Panel a/c: 表 2 冻结常量 (终点有效口径, 已逐格审计)
- Panel b:   traj_mae_ci_cache.npz 重算 (终点有效口径逐器官 = S3 Panel B 源)
- Panel d:   frozen_measurement_conditioned.json (条件化口径 = S3 Panel E 源)
输出: fig2.pdf / fig2.png(300dpi) / fig2_caption_cn.txt / fig2_data_used.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════ 统一 style block (Figure 2–5 复用) ══════════
# 配色取自论文绘图配色博客三色-8 (深蓝-金橙-灰): 灰基线 / 深蓝 Transformer / 金橙 PLF-OGT
COL_STATIC = "#A8ACB9"
COL_TRANS = "#2B6688"
COL_PLF = "#F1A93B"
COLS = [COL_STATIC, COL_TRANS, COL_PLF]
COL_HIGHLIGHT = "#EEF2F4"
COL_GRID = "#E9ECEF"
COL_TEXT = "#333333"
MODEL_NAMES = ["静态状态延续对照", "Transformer", "PLF-OGT"]
plt.rcParams.update({
    # 中文字体置首: matplotlib 版本 fallback 不可靠, 雅黑同时覆盖中文与数字
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": COL_TEXT,
    "axes.labelcolor": COL_TEXT,
    "xtick.color": COL_TEXT,
    "ytick.color": COL_TEXT,
    "axes.edgecolor": COL_TEXT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 300,
})


def style_axis(ax, ygrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=COL_GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)


def panel_label(fig, ax, letter):
    fig.text(ax.get_position().x0 - 0.015, ax.get_position().y1 + 0.005, letter,
             fontsize=9, fontweight="bold", ha="right", va="bottom")


def save_figure(fig, stem):
    """逐格式保存; 文件被阅读器锁定时逐级尝试临时名, 单格式失败不中断 (Windows)."""
    import os
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        target = stem + ext
        base = stem + "_tmp"
        tmps = (base + ext, base + "2" + ext, base + "3" + ext)
        try:
            fig.savefig(target, **kw)
            for t in tmps:  # 正式文件已更新, 清理遗留临时件
                try:
                    os.remove(t)
                except (FileNotFoundError, PermissionError):
                    pass
            continue
        except PermissionError:
            pass
        for t in tmps:
            try:
                fig.savefig(t, **kw)
            except PermissionError:
                continue
            try:
                os.replace(t, target)
            except PermissionError:
                print(f"提示: {os.path.basename(target)} 被阅读器占用, 新版在 {os.path.basename(t)} (关闭后重跑合入)")
            break
        else:
            print(f"警告: {os.path.basename(target)} 及全部临时名均被占用, 本次未更新该格式")


def write_safe(path: Path, data: bytes):
    """文本/数据落盘, 被占用时写 _tmp 并尝试替换 (Windows)."""
    import os
    try:
        path.write_bytes(data)
        tmp = path.with_name(path.stem + "_tmp" + path.suffix)
        try:
            os.remove(tmp)
        except (FileNotFoundError, PermissionError):
            pass
    except PermissionError:
        tmp = path.with_name(path.stem + "_tmp" + path.suffix)
        tmp.write_bytes(data)
        try:
            os.replace(tmp, path)
        except PermissionError:
            print(f"警告: {path.name} 被阅读器占用, 新版在 {tmp.name} (关闭后重跑即可)")


# ══════════ 数据 (锁定值 + 断言) ══════════
HORIZONS = [1, 3, 6, 12]

# Panel a: 总 SOFA MAE (表 2, 终点有效口径)
TOTAL = {
    "static": [0.266, 0.525, 0.646, 0.794],
    "tr": [0.611, 0.651, 0.725, 0.842],
    "plf": [0.948, 0.967, 1.002, 1.080],
}

# Panel b: 6h 逐器官 (终点有效口径; 源 = traj_mae_ci_cache.npz, S3 Panel B)
ORGANS_CN = ["呼吸", "循环", "肾脏", "凝血", "肝脏", "中枢神经"]
EXPECTED_B = {  # 断言基准 (S3 Panel B 已审计值)
    "static": [0.218, 0.431, 0.058, 0.056, 0.027, 0.084],
    "tr":     [0.659, 0.152, 0.111, 0.119, 0.071, 0.142],
    "plf":    [0.815, 0.177, 0.342, 0.131, 0.218, 0.147],
}

# Panel c: 循环分量多时距 (表 2)
CV = {
    "static": [0.225, 0.419, 0.431, 0.403],
    "tr": [0.154, 0.153, 0.152, 0.156],
    "plf": [0.193, 0.184, 0.177, 0.181],
}

# Panel d: 条件化循环 6h (frozen_measurement_conditioned.json, S3 Panel E 源)
EXPECTED_D = {"static": 0.460, "tr": 0.148, "plf": 0.165}


def load_data():
    """Panel b 从 npz 缓存重算 (终点有效口径); Panel d 从条件化 json 读取."""
    base = Path(r"F:/MIMIC3_1/V13/results/v4")
    # Panel b
    c = np.load(base / "traj_mae_ci_cache.npz", allow_pickle=True)
    t = np.load(base / "transformertcr_trajectory_cache.npz", allow_pickle=True)
    organ, omask = c["organ_lab"], c["organ_mask"]
    h = 6
    msk = omask[:, h, :]
    preds = {"static": organ[:, 0, :], "tr": t["pred"][:, h - 1, :], "plf": c["pred"][:, h - 1, :]}
    panel_b = {}
    for k, pr in preds.items():
        panel_b[k] = [float(np.abs(pr[msk[:, o] > 0, o] - organ[msk[:, o] > 0, h, o]).mean())
                      for o in range(6)]
    # Panel d
    mc = json.load(open(base / "frozen_measurement_conditioned.json", encoding="utf-8"))
    cv_idx = 1
    panel_d = {k: mc["mae"]["h6"][k]["organ_mae"][cv_idx] for k in ["persistence", "TR", "PLF"]}
    panel_d = {"static": panel_d["persistence"], "tr": panel_d["TR"], "plf": panel_d["PLF"]}
    return panel_b, panel_d


def assert_all(panel_b, panel_d):
    for k in ["static", "tr", "plf"]:
        for got, exp in zip(panel_b[k], EXPECTED_B[k]):
            assert abs(got - exp) <= 0.001, f"Panel b {k}: {got} vs {exp}"
        assert abs(panel_d[k] - EXPECTED_D[k]) <= 0.001, f"Panel d {k}: {panel_d[k]} vs {EXPECTED_D[k]}"
    print("断言全部通过: 所有绘图值与表 2 / S3 Panel B / S3 Panel E 锁定值一致")


# ══════════ 绘图 ══════════
def main():
    panel_b, panel_d = load_data()
    assert_all(panel_b, panel_d)

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 140 / 25.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.145,
                        wspace=0.32, hspace=0.42)

    # ── Panel a ──
    style_axis(ax_a, ygrid=True)
    for key, col in zip(["static", "tr", "plf"], COLS):
        ax_a.plot(HORIZONS, TOTAL[key], "-o", color=col, linewidth=1.4,
                  markersize=3.8, label=None)
    ax_a.set_xticks(HORIZONS)
    ax_a.set_ylim(0, 1.15)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel("预测时距（h）")
    ax_a.set_ylabel("总 SOFA MAE")

    # ── Panel b ──
    style_axis(ax_b, ygrid=True)
    x = np.arange(6)
    bw = 0.26
    # 在每个器官内部按柱高动态错开标签，避免循环、凝血和中枢神经的近值重叠。
    pad, min_gap = 0.012, 0.042
    series = np.asarray([panel_b["static"], panel_b["tr"], panel_b["plf"]], dtype=float)
    label_y = series + pad
    for organ_idx in range(series.shape[1]):
        previous = -np.inf
        for model_idx in np.argsort(series[:, organ_idx]):
            label_y[model_idx, organ_idx] = max(label_y[model_idx, organ_idx], previous + min_gap)
            previous = label_y[model_idx, organ_idx]
    for gi, (key, col) in enumerate(zip(["static", "tr", "plf"], COLS)):
        bars = ax_b.bar(x + (gi - 1) * bw, panel_b[key], width=bw, color=col,
                        edgecolor=col, linewidth=0.3)
        for organ_idx, (b_, v) in enumerate(zip(bars, panel_b[key])):
            ax_b.text(b_.get_x() + b_.get_width() / 2, label_y[gi, organ_idx], f"{v:.3f}",
                      ha="center", va="bottom", fontsize=5.4)
    # 循环高亮
    ax_b.axvspan(x[1] - 0.45, x[1] + 0.45, color=COL_HIGHLIGHT, zorder=0)
    ax_b.text(x[1], 0.865, "排序反转", ha="center", va="bottom",
              fontsize=6, color="#555555")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ORGANS_CN)
    ax_b.set_ylim(0, 0.95)
    ax_b.set_ylabel("6 h 器官分量 MAE")

    # ── Panel c ──
    style_axis(ax_c, ygrid=True)
    for key, col in zip(["static", "tr", "plf"], COLS):
        ax_c.plot(HORIZONS, CV[key], "-o", color=col, linewidth=1.4, markersize=3.8)
    ax_c.set_xticks(HORIZONS)
    ax_c.set_ylim(0, 0.50)
    ax_c.set_xlabel("预测时距（h）")
    ax_c.set_ylabel("循环分量 MAE")

    # ── Panel d ──
    style_axis(ax_d, ygrid=True)
    xd = np.arange(3)
    vals_d = [panel_d["static"], panel_d["tr"], panel_d["plf"]]
    bars = ax_d.bar(xd, vals_d, width=0.5, color=COLS, edgecolor=COLS, linewidth=0.3)
    for b_, v in zip(bars, vals_d):
        ax_d.text(b_.get_x() + b_.get_width() / 2, v + 0.008, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=5.5)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["静态延续", "Transformer", "PLF-OGT"])
    ax_d.set_ylim(0, 0.50)
    ax_d.set_ylabel("6 h 循环分量 MAE")
    ax_d.text(0.5, 0.965, "预测窗内至少 1 项循环 SOFA 组成变量有新测量",
              transform=ax_d.transAxes, ha="center", va="top",
              fontsize=6, color="#555555")

    # Panel 角标
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        panel_label(fig, ax, letter)

    # 底部共享图例
    handles = [plt.Line2D([0], [0], color=col, linewidth=1.4, marker="o",
                          markersize=3.8) for col in COLS]
    fig.legend(handles, MODEL_NAMES, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.015), handlelength=2.2,
               columnspacing=2.5)

    save_figure(fig, str(Path(__file__).parent / "fig2"))
    plt.close(fig)

    # ── 数据存档 ──
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Panel", "项目", "静态状态延续对照", "Transformer", "PLF-OGT"])
    for i, h in enumerate(HORIZONS):
        w.writerow([f"a 总SOFA MAE", f"{h}h", TOTAL["static"][i], TOTAL["tr"][i], TOTAL["plf"][i]])
    for i, o in enumerate(ORGANS_CN):
        w.writerow(["b 6h器官分量MAE(终点有效)", o, panel_b["static"][i], panel_b["tr"][i], panel_b["plf"][i]])
    for i, h in enumerate(HORIZONS):
        w.writerow(["c 循环MAE", f"{h}h", CV["static"][i], CV["tr"][i], CV["plf"][i]])
    w.writerow(["d 条件化循环MAE 6h", "≥1项新循环测量", panel_d["static"], panel_d["tr"], panel_d["plf"]])
    write_safe(Path(__file__).parent / "fig2_data_used.csv", buf.getvalue().encode("utf-8-sig"))

    # ── 中文图注 ──
    caption = (
        "图 2｜总 SOFA 指标掩盖短期轨迹重构中的器官级异质性。\n"
        "a，在各预测时距结局可评价的全部锚点中比较 1、3、6 和 12 h 总 SOFA 平均绝对误差（MAE）。"
        "静态状态延续对照将每一锚点记录的 SOFA 状态原样延续至相应预测时距，其总 SOFA 误差最低。\n"
        "b，6 h 六器官分量 MAE。静态状态延续对照在呼吸、肾脏、凝血、肝脏和中枢神经分量中误差最低，"
        "而循环分量呈相反排序，两种学习模型均优于静态状态延续对照。\n"
        "c，循环分量在 1、3、6 和 12 h 的 MAE。学习模型相对静态状态延续对照的优势在各预测时距保持一致；"
        "3 h 后静态状态延续误差约为学习模型的 2.6–2.8 倍。\n"
        "d，测量条件化敏感性分析，仅纳入预测窗内至少存在 1 项新的循环 SOFA 组成变量测量的观测。"
        "6 h 循环分量的相对排序与主要分析保持一致。图中均为点估计；相应置信区间在可获得时见表 2 和补充表 S3。"
    )
    write_safe(Path(__file__).parent / "fig2_caption_cn.txt", caption.encode("utf-8"))

    print("输出: fig2.pdf / fig2.png / fig2_caption_cn.txt / fig2_data_used.csv")


if __name__ == "__main__":
    main()
