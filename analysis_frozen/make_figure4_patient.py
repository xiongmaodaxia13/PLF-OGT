#!/usr/bin/env python
"""Figure 4 frozen 版: Panel A 代理恢复+随机映射 + Panel B S/R 2x2 + Panel C patient-specificity 四柱图."""
import matplotlib
matplotlib.use("Agg")
import json, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

RD = Path(r"F:/MIMIC3_1/V13/results/v4")
OUT = Path(r"F:/MIMIC3_1/V13/figures/图4_概念残差审计_frozen.png")

# 数据
psr = json.load(open(str(RD / "frozen_patient_specific_r_v2.json")))
summary = psr["summary"]
boot = psr["bootstrap_seed42"]

# patient-specificity 四条件
conditions = ["matched", "mean", "query_only", "shuffled"]
labels_cn = ["matched\n（正确R）", "population\nmean", "query-only\n（清零R）", "shuffled\n（换患者R）"]
colors = ["#2E74B5", "#5F9EA0", "#F9AB00", "#D93025"]

auprcs = [summary[c]["auprc_mean"] for c in conditions]
auprc_stds = [summary[c]["auprc_std"] for c in conditions]
maes = [summary[c]["macro_mae_mean"] for c in conditions]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1]})

# ============================================================
# Panel A: AUPRC 四柱图 (+ bootstrap CI for seed42)
# ============================================================
x = np.arange(len(conditions))
bars = ax_a.bar(x, auprcs, 0.6, color=colors, edgecolor="white", yerr=auprc_stds,
                capsize=4, error_kw={"linewidth": 1.2, "color": "#5F6368"})

# 标注数值
for i, (v, c) in enumerate(zip(auprcs, conditions)):
    ax_a.annotate(f"{v:.3f}", xy=(i, v + 0.01), ha="center", fontsize=9, weight="bold")
    # bootstrap CI (ΔAUPRC = matched - cond, 所以 cond 的 CI 下界 = matched - CI_hi)
    if c != "matched" and c in boot:
        ci = boot[c]["delta_auprc_ci"]
        ax_a.annotate(f"Δ{ci[0]:+.3f}\n至\n{ci[1]:+.3f}", xy=(i, v - 0.05),
                      ha="center", fontsize=7, color="#5F6368")

ax_a.set_xticks(x)
ax_a.set_xticklabels(labels_cn, fontsize=9)
ax_a.set_ylabel("6h AUPRC", fontsize=11)
ax_a.set_title("a  Patient-specificity negative controls\n(3-seed mean ± SD)", fontsize=11, weight="bold", loc="left")
ax_a.set_ylim(0.35, 0.62)
ax_a.grid(axis="y", alpha=0.3, linestyle="--")
# 加水平虚线 = matched
ax_a.axhline(auprcs[0], color="#2E74B5", linewidth=0.8, linestyle="--", alpha=0.5)

# ============================================================
# Panel B: macro MAE 四柱图
# ============================================================
bars_b = ax_b.bar(x, maes, 0.6, color=colors, edgecolor="white")
for i, v in enumerate(maes):
    ax_b.annotate(f"{v:.3f}", xy=(i, v + 0.01), ha="center", fontsize=9, weight="bold")

ax_b.set_xticks(x)
ax_b.set_xticklabels(labels_cn, fontsize=9)
ax_b.set_ylabel("6h macro-MAE", fontsize=11)
ax_b.set_title("b  轨迹误差（越低越好）", fontsize=11, weight="bold", loc="left")
ax_b.set_ylim(0, 1.1)
ax_b.grid(axis="y", alpha=0.3, linestyle="--")
ax_b.axhline(maes[0], color="#2E74B5", linewidth=0.8, linestyle="--", alpha=0.5)

fig.suptitle("图 4｜Residual representation patient-specificity 审计", fontsize=13, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(str(OUT), dpi=200, bbox_inches="tight")
print(f"保存: {OUT}")
