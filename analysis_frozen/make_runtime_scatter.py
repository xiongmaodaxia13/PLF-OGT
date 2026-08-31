#!/usr/bin/env python
"""#7: runtime vs AUROC 散点图 (陆老师建议, 如实呈现).

横轴: 单 batch 推理时间 (ms, log 轴, 因 PLF/TR 差 2.82x)
纵轴: 6h/各时距 TCR AUROC
每个模型 4 个点 (1/3/6/12h), 两模型两色.

输出: manuscripts/gmuicu/figures/Fig_runtime_vs_auroc.png
"""
import matplotlib
matplotlib.use("Agg")
import json, matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/manuscripts/gmuicu/figures/Fig_runtime_vs_auroc.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

rt = json.load(open(REPO / "results/v4/runtime_benchmark.json"))
plf_ms = rt["PLF-OGT"]["ms_per_batch"]
tr_ms = rt["Transformer"]["ms_per_batch"]
plf_params = rt["PLF-OGT"]["params"]
tr_params = rt["Transformer"]["params"]

plf = json.load(open(REPO / "results/v4/allhorizon_ci.json"))["TCR"]
tr = json.load(open(REPO / "results/v4/baselines_multihorizon_tcr.json"))["transformer_tcr"]

horizons = ["1h", "3h", "6h", "12h"]
plf_auc = [plf[h]["auroc"] for h in horizons]
tr_auc = [tr[h]["auroc"] for h in horizons]

fig, ax = plt.subplots(figsize=(8.5, 6))

# 散点 (同模型4点在同一 runtime 垂线)
ax.scatter([plf_ms]*4, plf_auc, s=140, c="#D93025", marker="o",
           edgecolors="white", linewidth=1.5, zorder=3, label=f"PLF-OGT ({plf_params/1e6:.2f}M 参数)")
ax.scatter([tr_ms]*4, tr_auc, s=140, c="#1A73E8", marker="s",
           edgecolors="white", linewidth=1.5, zorder=3, label=f"Transformer ({tr_params/1e6:.2f}M 参数)")

# 标注每个点的时距
for h, a in zip(horizons, plf_auc):
    ax.annotate(h, (plf_ms, a), textcoords="offset points", xytext=(12, 4),
                fontsize=9, color="#D93025")
for h, a in zip(horizons, tr_auc):
    ax.annotate(h, (tr_ms, a), textcoords="offset points", xytext=(12, 4),
                fontsize=9, color="#1A73E8")

# 连线同模型不同时距 (展示趋势)
ax.plot([plf_ms]*4, plf_auc, color="#D93025", alpha=0.3, lw=1.2, zorder=2)
ax.plot([tr_ms]*4, tr_auc, color="#1A73E8", alpha=0.3, lw=1.2, zorder=2)

ax.set_xscale("log")
ax.set_xlabel("单 batch 推理时间 (ms, 对数轴)", fontsize=12)
ax.set_ylabel("TCR AUROC", fontsize=12)
ax.set_title("推理速度与判别性能：PLF-OGT vs Transformer（各预测时距）", fontsize=13, weight="bold")

# 标注速度比
speed_ratio = plf_ms / tr_ms
ax.text(0.5, 0.97, f"PLF-OGT 比 Transformer 慢 {speed_ratio:.2f}×  |  同一 RTX 4090 D, batch=512, bf16",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FEF7E0", ec="#F9AB00", lw=1))

ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
ax.set_ylim(0.80, 0.94)

# 表注信息
fig.text(0.5, 0.01,
         "硬件：NVIDIA RTX 4090 D (46 GB)  |  框架：PyTorch 2.10 + CUDA 12.8  |  语言：Python  |  "
         "每模型 30 次推理取中位数",
         ha="center", fontsize=8.5, color="#5F6368")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"保存: {OUT}")
