# -*- coding: utf-8 -*-
"""
Publication-Quality Figure 6 (F6) for Journal of Hydrology R3.

Scientific Question:
"In cross-calibration paradigm comparison, does parameter-information organization
preferentially align with the same parameter coordinate rather than alternative coordinates?"

Layout Architecture:
- Left Column: (b) Ensemble same-coordinate specificity across n=35 models with permutation null inset (Hero panel)
- Right Top: (a) Parameter-level specificity in representative model structures (5 vertically stacked hydrological skeletons)
- Right Bottom: (c) Same coordinate among top-ranked matches (Top-1, Top-2, Top-3 observed vs random expectation on 270 eligible parameters)

Output files:
- project/benchmark/manuscript/r3/figures/F6_coordinate_specificity_main.png (600 DPI, PNG)
- project/benchmark/manuscript/r3/figures/F6.png (600 DPI, PNG)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import blended_transform_factory as blend
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Paths
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUT_PNG_MAIN = FIGURES / "F6_coordinate_specificity_main.png"
OUTPUT_PNG_STD = FIGURES / "F6.png"

# ============================================================ 1. 全局画布与排版参数
W, H = 1448, 1086          # 参考画布（px），所有位置基于此坐标系
SCALE = 1.0                # 1.0 -> 14.48 in 宽, 600 DPI 高清导出
DPI = 600


def pt(x):                 # 字体 / 线宽随 SCALE 缩放
    return x * SCALE


def area(d):               # scatter 直径 d pt
    return (d * SCALE) ** 2


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": pt(1.0),
    "xtick.major.width": pt(1.0),
    "xtick.major.size": pt(4.5),
    "xtick.major.pad": pt(4),
    "xtick.labelsize": pt(12.5),
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

TEAL = "#12776b"
TEAL_EDGE = "#0b4f47"
TEAL_TXT = "#247b70"       # 调低饱和度
PURPLE = "#7c5a96"         # 调低饱和度
STEM = "#9d9d9d"
BAND = "#edf6f4"           # 更浅的 IQR 阴影带
GREY_DOT = "#8d8d8d"
CORAL = "#c2593f"
CORAL_DARK = "#9c3b24"

# 5 档特异性标记颜色：浅 -> 深
LEVELS = ["#e0f0ed", "#a6d5cd", "#57ae9f", "#1a8474", "#0b4a42"]


def get_level(adv: float) -> int:
    if adv < 0.30:
        return 0
    elif adv < 0.60:
        return 1
    elif adv < 0.85:
        return 2
    elif adv < 1.10:
        return 3
    else:
        return 4


# ============================================================ 2. 数据读取与审计验证
print("Loading R3 canonical data tables...")
df_cross = pd.read_csv(TABLES / "R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv")
df_cross_info = df_cross[df_cross["space"] == "information_cluster"]

df_model = pd.read_csv(TABLES / "R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv")
df_model_info = df_model[(df_model["space"] == "information_cluster") & (df_model["population"] == "all36")]

from dmotpy.models.core import STATE_INFO

# 组织 35 个模型的真实 A_m，严格按 1S 到 6S 分组
ENSEMBLE = {
    "1S": [],
    "2S": [],
    "3S": [],
    "4S": [],
    "5S": [],
    "6S": [],
}

for idx, r in df_model_info.iterrows():
    m = r["model"]
    if m == "collie1":
        continue
    s_cnt = STATE_INFO.get(m, 0)
    stratum = f"{s_cnt}S"
    am = float(r["diagonal_advantage_median"])
    if stratum in ENSEMBLE:
        ENSEMBLE[stratum].append((m, am))

GROUP_COLOR = {g: (TEAL_TXT if g in ("1S", "3S", "5S") else PURPLE) for g in ENSEMBLE}

# 提取 5 个代表模型的参数特异性
rep_models = ["ihacres", "topmodel", "vic", "xinanjiang", "hbv96"]
adv_by_model: dict[str, dict[str, float]] = {}

for m in rep_models:
    sub = df_cross_info[df_cross_info["model"] == m]
    params = sub.drop_duplicates("ic_parameter").sort_values("ic_parameter_index")["ic_parameter"].tolist()
    c_mat = sub.pivot(index="ic_parameter", columns="dpl_parameter", values="correspondence").loc[params, params].to_numpy()
    m_adv = {}
    for i, p in enumerate(params):
        diag = c_mat[i, i]
        others = np.delete(c_mat[i], i)
        m_adv[p] = float(diag - np.nanmedian(others))
    adv_by_model[m] = m_adv


def rank_rows(x: np.ndarray) -> np.ndarray:
    return np.vstack([rankdata(row, method="average") for row in np.asarray(x, float)])


def cross_matrix(ic: np.ndarray, dpl: np.ndarray) -> np.ndarray:
    ir = rank_rows(ic)
    dr = rank_rows(dpl)
    ir = ir - ir.mean(axis=1, keepdims=True)
    dr = dr - dr.mean(axis=1, keepdims=True)
    denom = np.sqrt((ir * ir).sum(axis=1)[:, None] * (dr * dr).sum(axis=1)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (ir @ dr.T) / denom
    return out


# 计算真实 1,000 次置换零分布
print("Computing 1,000-draw parameter label permutation null...")
df_long = pd.read_csv(TABLES / "R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
df_long_info = df_long[df_long["space"] == "information_cluster"]
models_35 = sorted([m for m in df_long_info["model"].unique() if m != "collie1"])

ic_by_model = {}
dpl_by_model = {}
for m in models_35:
    sub_ic = df_long_info[(df_long_info["model"] == m) & (df_long_info["method"] == "IC")]
    p_ic = sorted(sub_ic["parameter_index"].unique())
    f_ic = sorted(sub_ic["feature_index"].unique())
    ic_by_model[m] = sub_ic.pivot(index="parameter_index", columns="feature_index", values="rho").reindex(index=p_ic, columns=f_ic).to_numpy(float)

    sub_dpl = df_long_info[(df_long_info["model"] == m) & (df_long_info["method"] == "dPL")]
    p_dpl = sorted(sub_dpl["parameter_index"].unique())
    f_dpl = sorted(sub_dpl["feature_index"].unique())
    dpl_by_model[m] = sub_dpl.pivot(index="parameter_index", columns="feature_index", values="rho").reindex(index=p_dpl, columns=f_dpl).to_numpy(float)

rng = np.random.default_rng(20260902)
n_perm = 1000
NULL_MEDIANS = np.empty(n_perm, dtype=float)
for b in range(n_perm):
    model_advs = []
    for m in models_35:
        dpl_p = dpl_by_model[m]
        ic_p = ic_by_model[m]
        perm = rng.permutation(dpl_p.shape[0])
        c_mat = cross_matrix(ic_p, dpl_p[perm])
        diag = np.diag(c_mat)
        advs = []
        for i in range(c_mat.shape[0]):
            others = np.delete(c_mat[i], i)
            advs.append(diag[i] - np.nanmedian(others))
        model_advs.append(np.nanmedian(advs))
    NULL_MEDIANS[b] = float(np.nanmedian(model_advs))

# Top-k 统计数据（270 个有效参数坐标）
TOPK = dict(k=[1, 2, 3], random=[0.12963, 0.25926, 0.38889], hits=[155, 192, 227], total=270)
TOPK_XLABEL = "Cumulative share of same-coordinate matches"
STEM_BASE = 0.0
PANEL = {"ensemble": "(b)", "schematic": "(a)", "topk": "(c)"}

# ============================================================ 3. 画布与通用排版工具
fig = plt.figure(figsize=(W / 100 * SCALE, H / 100 * SCALE))
cv = fig.add_axes([0, 0, 1, 1], zorder=0)
cv.set_xlim(0, W)
cv.set_ylim(H, 0)                       # y 轴向下，严格对齐像素坐标
cv.axis("off")


def px_axes(x0, y0, x1, y1):
    """在像素矩形 (x0, y0)-(x1, y1) 处新建数据坐标轴。"""
    ax = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], zorder=1)
    ax.patch.set_alpha(0)
    return ax


def txt(x, y, s, size, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return cv.text(x, y, s, fontsize=pt(size), **kw)


def panel_title(x, y_base, letter, rest, size_rest, size_letter=18.5):
    """面板标题：编号加粗，紧随说明文字。"""
    t = txt(x, y_base, letter, size_letter, fontweight="bold", va="baseline")
    cv.annotate(" " + rest, xy=(1, y_base), xycoords=(t, "data"), ha="left", va="baseline",
                fontsize=pt(size_rest), fontweight="bold")


# 三大主面板外边框
for x0, y0, x1, y1 in [(3, 3, 776, 1083), (785, 3, 1445, 723), (785, 733, 1445, 1083)]:
    cv.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#3a3a3a", lw=pt(1.0)))

# ============================================================ 4. (b) Ensemble lollipop (Hero Panel)
GAP = 0.50
names, vals, ys, groups = [], [], [], []
y = 0.0
for g, items in ENSEMBLE.items():
    y_start = y
    for name, v in sorted(items, key=lambda t: t[1]):   # 组内按 A_m 升序排列
        names.append(name)
        vals.append(v)
        ys.append(y)
        y += 1
    groups.append((g, y_start, y - 1, len(items)))
    y += GAP
vals, ys = np.asarray(vals), np.asarray(ys)

n_models = vals.size
med = np.median(vals)
q1, q3 = np.percentile(vals, [25, 75])
n_pos = int((vals > 0).sum())
p_perm = (1 + (NULL_MEDIANS >= med).sum()) / (1 + NULL_MEDIANS.size)

axb = px_axes(223, 95, 752, 990)
axb.axvspan(q1, q3, color=BAND, lw=0, zorder=0)
axb.axvline(0.0, color="#888888", lw=pt(0.9), ls="-", zorder=1)           # 0 基准线清晰保留
axb.axvline(med, color="#444444", lw=pt(1.0), ls=(0, (4, 3)), zorder=1)
axb.hlines(ys, STEM_BASE, vals, color=STEM, lw=pt(1.4), zorder=2)
axb.scatter(vals, ys, s=area(7.8), color=TEAL, ec=TEAL_EDGE, lw=pt(0.6), zorder=3, clip_on=False)
axb.set_xlim(-0.2, 1.55)                                                   # 保留少量负值空间 (-0.2 to 1.55)
axb.set_ylim(ys[-1] + 1.2, ys[0] - 0.85)
axb.set_xticks(np.arange(-0.2, 1.41, 0.2))
axb.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axb.set_yticks([])
axb.spines[["top", "right"]].set_visible(False)
axb.set_xlabel(r"Same-coordinate advantage over alternative coordinates, $A_m$", fontsize=pt(14.0), labelpad=pt(8))

# 左侧：模型名（右对齐并留出 11 px 边距）、分组括号、分组标签（使用混合坐标）
tb = blend(cv.transData, axb.transData)
for name, yy in zip(names, ys):
    cv.text(212, yy, name, transform=tb, fontsize=pt(12.0), ha="right", va="center", color="#1e293b")
for g, y0, y1, n in groups:
    c = GROUP_COLOR[g]
    cv.plot([122, 106, 106, 122], [y0 - 0.35, y0 - 0.35, y1 + 0.35, y1 + 0.35],
            transform=tb, color=c, lw=pt(1.0))                             # 分组线调细
    ym = 0.5 * (y0 + y1)
    cv.text(52, ym - 0.35, g, transform=tb, color="#475569", fontsize=pt(16.0), fontweight="bold",
            ha="center", va="center")
    cv.text(52, ym + 0.45, f"($n$ = {n})", transform=tb, color="#64748b", fontsize=pt(11.5),
            ha="center", va="center")

txt(52, 78, "Store\ncomplexity", 11.5, ha="center", color="#475569", linespacing=1.05)
txt(172, 76, "Model", 12.5, ha="center", fontweight="bold")
panel_title(13, 33, PANEL["ensemble"],
            rf"Ensemble same-coordinate specificity ($\mathbfit{{n}}$ = {n_models} models)", 18.5)

p_txt = r"$p$ < 0.001" if p_perm < 1e-3 else f"$p$ = {p_perm:.3f}"
stats = "\n".join([f"{n_pos}/{n_models} models > 0",
                   f"Median $A_m$ = {med:.3f}",
                   f"IQR [{q1:.3f}, {q3:.3f}]",
                   "Permutation " + p_txt])
txt(602, 82, stats, 12.0, va="top", linespacing=1.25)

# 零分布插图：上移至 hbv96 高度并放大尺寸，增强可读性
axi = px_axes(550, 680, 745, 850)
axi.patch.set_alpha(1)
axi.patch.set_facecolor("white")
counts, _, _ = axi.hist(NULL_MEDIANS, bins=np.arange(-0.3, 0.3001, 0.025),
                        color="#b8b8b8", ec="white", lw=pt(0.4))
axi.axvline(0, color="#333333", lw=pt(0.9), ls=(0, (3, 2)))
axi.axvline(med, color=TEAL, lw=pt(1.6))
axi.set_xlim(-0.35, 0.75)
axi.set_ylim(0, counts.max() * 1.45)
axi.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
axi.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axi.set_yticks([])
axi.tick_params(labelsize=pt(9.0), length=pt(3.0), pad=pt(2))
for s in axi.spines.values():
    s.set_linewidth(pt(0.8))
axi.text(0.5, 0.94, "Permutation null", transform=axi.transAxes, ha="center", va="top", fontsize=pt(10.5), fontweight="bold")
axi.text(med - 0.03, counts.max() * 0.88, f"Observed\n$A_{{\\mathrm{{diag}}}} = {med:.3f}$\n($p < 0.001$)",
         ha="right", va="center", fontsize=pt(9.0), color=TEAL, fontweight="bold")

# ============================================================ 5. (a) 结构示意（5 个模型竖向单列）
LINE_KW = dict(color="#111111", lw=pt(1.0), zorder=2)


def box(x0, y0, x1, y1, label, level=None, where="bottom"):
    """画过程框；level 为 0-4 的标记深浅档，where 为标记位置（bottom / right）。"""
    cv.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec="#333333",
                           lw=pt(1.0), zorder=3))
    dy = -3 if (level is not None and where == "bottom") else 0
    txt((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, 11.0, ha="center", linespacing=1.0, zorder=4)
    if level is not None:
        mx, my = ((x0 + x1) / 2, y1) if where == "bottom" else (x1, (y0 + y1) / 2)
        cv.add_patch(Circle((mx, my), 7.0, fc=LEVELS[level], ec="#3a3a3a", lw=pt(0.8), zorder=5))
    return (y0 + y1) / 2


def arrow(x0, y0, x1, y1):
    cv.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=pt(1.0),
                                mutation_scale=pt(10.5), shrinkA=0, shrinkB=0))


def line(xs, yy):
    cv.plot(xs, yy, **LINE_KW)


def ep(x, y_top, length=28):
    arrow(x, y_top, x, y_top - length)
    txt(x + 6, y_top - length + 4, r"$E_p$", 13.0)


def split(x_from, y_from, x_node, y_branches, x_to):
    """主干 -> 分叉节点 -> 各分支框左边。"""
    line([x_from, x_node], [y_from, y_from])
    line([x_node, x_node], [min(y_branches), max(y_branches)])
    for yb in y_branches:
        arrow(x_node, yb, x_to, yb)


def merge(x_from, y_branches, x_node, y_out, x_to):
    """各分支框右边 -> 汇合节点 -> 下游。"""
    for yb in y_branches:
        line([x_from, x_node], [yb, yb])
    line([x_node, x_node], [min(y_branches), max(y_branches)])
    arrow(x_node, y_out, x_to, y_out)


def model_label(y_name, name, spec):
    txt(797, y_name, name, 14.0, fontweight="bold")
    txt(797, y_name + 25, spec, 14.0, color=TEAL_TXT)


panel_title(797, 31, PANEL["schematic"],
            "Parameter-level specificity in representative model structures", 15.0)
for yy in (168, 288, 408, 540):
    cv.plot([795, 1437], [yy, yy], color="#cfcfcf", lw=pt(0.8), zorder=1)

# IHACRES (1S, 6p)
y = 110
model_label(88, "IHACRES", "(1S, 6p)")
txt(924, y, "$P$", 14.0)
arrow(945, y, 1005, y)
box(1005, 89, 1125, 131, "Moisture\ndeficit", get_level(adv_by_model["ihacres"]["d"]))
ep(1077, 89)
yq = box(1193, 66, 1300, 100, "Quick route", get_level(adv_by_model["ihacres"]["tau_q"]), "right")
ysl = box(1193, 117, 1300, 151, "Slow route", get_level(adv_by_model["ihacres"]["tau_s"]), "right")
split(1125, y, 1155, [yq, ysl], 1193)
merge(1300, [yq, ysl], 1350, y, 1392)
txt(1403, y, "$Q$", 14.0)

# TOPMODEL (2S, 7p)
y = 230
model_label(214, "TOPMODEL", "(2S, 7p)")
txt(924, y, "$P$", 14.0)
arrow(945, y, 973, y)
box(973, 208, 1073, 252, "Topographic\nsplit", get_level(adv_by_model["topmodel"]["chi"]))
arrow(1073, y, 1105, y)
box(1105, 213, 1200, 247, "Unsaturated", get_level(adv_by_model["topmodel"]["suzmax"]))
ep(1160, 213)
arrow(1200, y, 1242, y)
box(1242, 207, 1352, 253, "Saturated\ndeficit", get_level(adv_by_model["topmodel"]["f"]))
arrow(1352, y, 1392, y)
txt(1403, y, "$Q$", 14.0)

# VIC (3S, 10p)
y = 354
model_label(333, "VIC", "(3S, 10p)")
txt(924, y, "$P$", 14.0)
arrow(945, y, 971, y)
box(971, 334, 1052, 375, "Canopy", get_level(adv_by_model["vic"]["ibar"]))
ep(1020, 334)
arrow(1052, y, 1085, y)
box(1085, 334, 1183, 375, "Upper\nsoil", get_level(adv_by_model["vic"]["fsm"]))
ep(1132, 334)
arrow(1183, y, 1217, y)
box(1217, 331, 1348, 377, "Deep soil /\ngroundwater", get_level(adv_by_model["vic"]["c2"]))
arrow(1348, y, 1390, y)
txt(1401, y, "$Q$", 14.0)

# Xinanjiang (4S, 12p)
y = 478
model_label(457, "Xinanjiang", "(4S, 12p)")
txt(918, y, "$P$", 14.0)
arrow(937, y, 971, y)
box(971, 459, 1065, 497, "Tension\nwater", get_level(adv_by_model["xinanjiang"]["stot"]))
ep(1022, 459)
arrow(1065, y, 1094, y)
box(1094, 461, 1188, 495, "Free water", get_level(adv_by_model["xinanjiang"]["ex"]))
yi = box(1240, 431, 1337, 465, "Interflow", get_level(adv_by_model["xinanjiang"]["ci"]), "right")
yg = box(1240, 484, 1337, 518, "GW route", get_level(adv_by_model["xinanjiang"]["kg"]), "right")
split(1188, y, 1210, [yi, yg], 1240)
merge(1337, [yi, yg], 1380, y, 1400)
txt(1408, y, "$Q$", 14.0)

# HBV96 (5S, 15p)
y = 609
model_label(587, "HBV96", "(5S, 15p)")
txt(890, y, "$P$, $T$", 14.0)
arrow(925, y, 948, y)
box(948, 590, 1020, 628, "Snow", get_level(adv_by_model["hbv96"]["tt"]))
arrow(1020, y, 1043, y)
box(1043, 589, 1127, 629, "Soil\nmoisture", get_level(adv_by_model["hbv96"]["lp"]))
ep(1087, 589)
yu = box(1157, 563, 1262, 597, "Upper zone", get_level(adv_by_model["hbv96"]["k0"]), "right")
yl = box(1157, 615, 1262, 649, "Lower zone", get_level(adv_by_model["hbv96"]["k1"]), "right")
split(1127, y, 1138, [yu, yl], 1157)
box(1298, 589, 1390, 629, "MAXBAS", get_level(adv_by_model["hbv96"]["perc"]))
merge(1262, [yu, yl], 1285, y, 1298)
arrow(1390, y, 1410, y)
txt(1416, y, "$Q$", 14.0)

# 底部单行统一说明与色标图例（精简为 Darker fill = larger same-coordinate advantage）
for i, c in enumerate(LEVELS):
    cv.add_patch(Circle((830 + 35 * i, 695), 9.0, fc=c, ec="#3a3a3a", lw=pt(0.8)))
txt(993, 695, "Darker fill = larger same-coordinate advantage", 11.5)

# ============================================================ 6. (c) Top-k 哑铃图
panel_title(797, 763, PANEL["topk"], r"Same coordinate among top-ranked matches", 18.0)
cv.scatter([1173], [767], s=area(9.5), color=GREY_DOT, ec="#666666", lw=pt(0.6))
txt(1188, 767, "Random expectation", 11.0)
cv.scatter([1350], [767], s=area(9.5), color=TEAL, ec=TEAL_EDGE, lw=pt(0.6))
txt(1365, 767, "Observed", 11.0)

axc = px_axes(870, 802, 1365, 985)     # 右侧留出充足标注空间
k = np.asarray(TOPK["k"])
rnd = np.asarray(TOPK["random"])
hits, tot = np.asarray(TOPK["hits"]), TOPK["total"]
obs = hits / tot
axc.hlines(k, rnd, obs, color="#444444", lw=pt(1.6), zorder=1)
axc.scatter(rnd, k, s=area(11.0), color=GREY_DOT, ec="#666666", lw=pt(0.6), zorder=3, clip_on=False)
axc.scatter(obs, k, s=area(11.0), color=TEAL, ec=TEAL_EDGE, lw=pt(0.6), zorder=3, clip_on=False)
for kk, r, o, h in zip(k, rnd, obs, hits):
    axc.annotate(f"{r:.1%}", (r, kk), xytext=(pt(6), pt(7)), textcoords="offset points",
                 ha="left", va="bottom", fontsize=pt(12.5))
    a = axc.annotate(f"{o:.1%}", (o, kk), xytext=(pt(10), 0), textcoords="offset points",
                     ha="left", va="center", fontsize=pt(13.5), color=TEAL, fontweight="bold")
    axc.annotate(f" ({h}/{tot})", xy=(1, 0.5), xycoords=a, ha="left", va="center",
                 fontsize=pt(13.5))
axc.set_xlim(0, 1)
axc.set_ylim(3.6, 0.63)
axc.set_xticks(np.arange(0, 1.01, 0.2))
axc.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
axc.set_yticks([])
axc.spines[["top", "right"]].set_visible(False)
axc.set_xlabel(TOPK_XLABEL, fontsize=pt(14.0), labelpad=pt(8))
tc = blend(cv.transData, axc.transData)
for kk in k:
    cv.text(805, kk, f"Top {kk}", transform=tc, fontsize=pt(15.0),
            ha="left", va="center")

# ============================================================ 7. 输出保存
if __name__ == "__main__":
    plt.savefig(OUTPUT_PNG_MAIN, dpi=DPI)
    plt.savefig(OUTPUT_PNG_STD, dpi=DPI)
    print("Publication-quality Figure 6 successfully generated and saved to:")
    print(f"  - {OUTPUT_PNG_MAIN}")
    print(f"  - {OUTPUT_PNG_STD}")
    print(f"Summary statistics: median={med:.3f}, IQR=[{q1:.3f}, {q3:.3f}], p={p_perm:.4f}, n_pos={n_pos}/{n_models}")
