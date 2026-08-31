#!/usr/bin/env python
"""Figure 3 frozen 版: Panel A 六器官MAE + Panel B non-CV增益.

从 frozen_organ_noncv.json 直接生成, 与正文/S11 完全一致.
"""
import matplotlib
matplotlib.use("Agg")
import json, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

RD = Path(r"F:/MIMIC3_1/V13/results/v4")
OUT = Path(r"F:/MIMIC3_1/V13/figures/图3_器官级范围_frozen.png")

org = json.load(open(str(RD / "frozen_organ_noncv.json")))

ORGAN_CN = ["呼吸", "心血管", "肾脏", "凝血", "肝脏", "CNS"]
ORGAN_KEY = ["Resp", "Cv", "Renal", "Coag", "Hepatic", "CNS"]
COLORS_TCR = "#2E74B5"
COLORS_CO = "#D93025"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 2]})

# ============================================================
# Panel A: 六器官 MAE (TCR vs care-off)
# ============================================================
tcr_maes = [org["organ_mae"][k]["tcr_mae"] for k in ORGAN_KEY]
co_maes = [org["organ_mae"][k]["co_mae"] for k in ORGAN_KEY]
x = np.arange(len(ORGAN_CN))
w = 0.35

bars1 = ax_a.bar(x - w/2, tcr_maes, w, label="TCR", color=COLORS_TCR, edgecolor="white")
bars2 = ax_a.bar(x + w/2, co_maes, w, label="care-off", color=COLORS_CO, edgecolor="white")

# 标注 Δ
for i, k in enumerate(ORGAN_KEY):
    delta = org["organ_mae"][k]["delta"]
    if abs(delta) > 0.01:
        ax_a.annotate(f"Δ+{delta:.3f}" if delta > 0 else f"Δ{delta:.3f}",
                      xy=(i, max(tcr_maes[i], co_maes[i]) + 0.02),
                      ha="center", fontsize=8, color="#D93025" if delta > 0 else "#5F6368")

ax_a.set_xticks(x)
ax_a.set_xticklabels(ORGAN_CN, fontsize=10)
ax_a.set_ylabel("6h SOFA MAE", fontsize=11)
ax_a.set_title("a  逐器官轨迹 MAE（TCR vs care-off）", fontsize=12, weight="bold", loc="left")
ax_a.legend(fontsize=9, loc="upper right")
ax_a.set_ylim(0, max(co_maes) * 1.25)
ax_a.grid(axis="y", alpha=0.3, linestyle="--")

# ============================================================
# Panel B: total vs non-CV 消融增益 (ΔAUROC + ΔBrier)
# ============================================================
nv = org["non_cv"]
tc = org["total_cls"]

categories = ["总 SOFA", "non-CV SOFA"]
delta_auroc = [tc["delta_auroc"], nv["delta_auroc"]]
delta_brier = [None, nv["delta_brier"]]  # total ΔBrier 没有在 frozen 里算

x_b = np.arange(len(categories))
w_b = 0.35

bars_auc = ax_b.bar(x_b - w_b/2, delta_auroc, w_b, label="ΔAUROC (TCR−care-off)",
                     color="#2E74B5", edgecolor="white")
# non-CV ΔBrier
brier_val = nv["delta_brier"]
ax_b.bar(x_b[1] + w_b/2, brier_val, w_b, label="ΔBrier (non-CV, care-off−TCR)",
         color="#F9AB00", edgecolor="white")

# 标注数值
for i, v in enumerate(delta_auroc):
    ax_b.annotate(f"{v:+.3f}", xy=(x_b[i] - w_b/2, v + 0.002),
                  ha="center", fontsize=9, weight="bold")
ax_b.annotate(f"{brier_val:+.3f}", xy=(x_b[1] + w_b/2, brier_val + 0.002),
              ha="center", fontsize=9, weight="bold", color="#F9AB00")

ax_b.axhline(0, color="#5F6368", linewidth=0.8, linestyle="-")
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(categories, fontsize=10)
ax_b.set_ylabel("消融差值", fontsize=11)
ax_b.set_title("b  总 SOFA 与 non-CV 消融增益", fontsize=12, weight="bold", loc="left")
ax_b.legend(fontsize=8, loc="upper right")
ax_b.set_ylim(-0.04, 0.10)
ax_b.grid(axis="y", alpha=0.3, linestyle="--")

fig.suptitle("图 3｜事实治疗路径信息的功能利用与器官级范围", fontsize=13, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(str(OUT), dpi=200, bbox_inches="tight")
print(f"保存: {OUT}")
