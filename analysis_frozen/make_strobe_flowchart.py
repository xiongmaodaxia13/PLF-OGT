#!/usr/bin/env python
"""补充图 S1 ｜ STROBE/TRIPOD 队列筛选流程图（精修版）.

GMUICU + MIMIC-IV 双列对照, 标注每步人数与排除原因.
数字均与 gmuicu_pipeline / mimic 构建代码核对一致.
输出: F:/MIMIC3_1/V13/figures/补充图S1_队列流程.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"F:/MIMIC3_1/V13/figures/补充图S1_队列流程.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 双列等高布局：两列内容垂直对齐到统一坐标 (xlim 0-10, ylim 0-10)
fig, (ax_g, ax_m) = plt.subplots(1, 2, figsize=(13.5, 8.6))
for ax in (ax_g, ax_m):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")


def box(ax, x, y, w, h, text, color="#E8F0FE", edge="#3367D6", fs=10.5, bold=False):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.08", lw=1.5, fc=color, ec=edge)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.45)


def excl(ax, x, y, text, color="#FCE8E6", edge="#D93025"):
    box(ax, x, y, 4.0, 0.95, text, color=color, edge=edge, fs=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#5F6368"))


# ===== GMUICU 开发队列 (左) =====
ax_g.set_title("GMUICU 开发队列", fontsize=13.5, weight="bold", pad=10)
box(ax_g, 5, 9.35, 6.2, 0.85, "ICU 转入记录筛选\nn = 3,902（transfer 表含 ICU 转入）", bold=True)
excl(ax_g, 5, 8.20, "排除：无 ICU 入科时间\n（无转入/入院时间）")
box(ax_g, 5, 7.05, 6.2, 0.85, "可定位 ICU 时间窗\nn = 3,887")
excl(ax_g, 5, 5.90, "排除：ICU 住院 < 6 h\n（MIN_ICU_LOS_H）")
box(ax_g, 5, 4.75, 6.2, 0.85, "cohort_master 纳入\nn = 3,850", color="#E6F4EA", edge="#137333", bold=True)
excl(ax_g, 5, 3.60, "排除：14 例缺足够逐小时\n观察事件（V4 锚点构建）")
box(ax_g, 5, 2.45, 6.2, 0.85, "V4 建模队列\nn = 3,836", color="#FEF7E0", edge="#F9AB00", bold=True)
ax_g.text(5, 1.10,
          "患者级 70:20:10 划分\n训练 2,684 / 验证 766 / 测试 386\n638,032 锚点（测试 67,665）",
          ha="center", va="center", fontsize=10.5, style="italic", linespacing=1.5)
for y1, y2 in [(8.92, 8.68), (7.77, 7.53), (6.62, 5.18), (4.32, 4.08), (3.17, 2.88)]:
    arrow(ax_g, 5, y1, 5, y2)

# ===== MIMIC-IV 独立复现队列 (右) =====
ax_m.set_title("MIMIC-IV 独立复现队列", fontsize=13.5, weight="bold", pad=10)
box(ax_m, 5, 9.35, 6.2, 0.85, "MIMIC-IV ICU stay 筛选\nn = 26,845（成人 ICU stay，LOS ≥ 6 h）", bold=True)
excl(ax_m, 5, 7.60, "排除：545 例\n缺足够逐小时观察事件 /\n器官掩码全无效 / 变量映射缺失")
box(ax_m, 5, 5.95, 6.2, 0.85, "V4 建模队列\nn = 26,300", color="#FEF7E0", edge="#F9AB00", bold=True)
# 用一个与 GMUICU 列底部对称的"划分"框，填补 MIMIC 列中下部空隙并保持两列视觉平衡
box(ax_m, 5, 4.30, 6.2, 1.50,
    "subject_id 级 70:20:10 划分\n训练 18,410 / 验证 5,260 / 测试 2,630\n423,174 锚点（测试 43,498）｜ 6 h 恶化率 3.1%",
    color="#E8F0FE", edge="#3367D6", fs=10.5, bold=True)
box(ax_m, 5, 2.45, 6.2, 1.10,
    "独立再开发\n相同变量定义、架构、训练流程\n（非固定权重迁移；本地重估标准化与权重）",
    color="#F1F3F4", edge="#5F6368", fs=10, bold=False)
for y1, y2 in [(8.92, 8.10), (7.10, 6.40), (5.50, 5.08)]:
    arrow(ax_m, 5, y1, 5, y2)

fig.suptitle("补充图 S1｜队列筛选与建模纳入流程（STROBE/TRIPOD 式）",
             fontsize=14.5, weight="bold", y=0.985)
plt.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"保存: {OUT}")
