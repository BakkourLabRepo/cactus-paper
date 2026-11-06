"""fig_ms_model.py — manuscript figure: the choice model, and what learning feeds into it.

  A  schematic of the model of record: value map, magnitude reliance, dimension weights -> logistic
     choice, with the dimension learning task's inputs (K_s, DeltaValue slope) on top.
  B  gain |W_s| vs overall value knowledge K_s
  C  contribution profile, observed vs lambda = 1 / lambda = 0 / model of record
  D  the same profile by tertile of the DeltaValue slope, with an inset of each tertile's fitted
     learning psychometric

C and D: x is the points a dimension is won by; y starts at 0.25 behind a break mark
(cactus_paper.visualization.plotting.profile_axes).

Inputs (run first if stale):
  pixi run profiles  ->  results/fits/readout_profile.json (C), slope_readout_profile.json (D)

Run: pixi run python scripts/exp2/figures/fig_ms_model.py  (any cwd; a few seconds)
"""

import json
import shutil
from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from cactus_paper.design import load_design
from cactus_paper.paths import FITS, MOR_IDATA, ROOT
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import (
    DIM_COLORS,
    POINTS,
    POLE_MAG,
    POLE_TALLY,
    boot_line,
    panel_label,
    profile_axes,
    style_ms,
    tertile_profile,
)

OUT = figpath("fig_ms_model.png")
LINE, DOT = style_ms()
FS_PANEL, FS_TICK = 24, 13.5
GREYS = ["#bdbdbd", "#737373", "#252525"]  # DeltaValue slope tertiles
K_COL = "#08306b"  # value knowledge: dark end of the K Blues ramp (fig3_poles)

D = load_design()
nS, K = D["nS"], np.asarray(D["K"], float)
po = az.from_netcdf(MOR_IDATA).posterior
gain = np.abs(po["W"].values.reshape(-1, nS, 3).mean(0)).mean(1)

fig = plt.figure(figsize=(9.6, 8.6))
outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.36)
top = outer[0].subgridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.2)
bot = outer[1].subgridspec(1, 2, wspace=0.42)
axA, axB = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
axC, axD = fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])

# ── A: the model schematic ─────────────────────────────────────────────────────────────
a = axA
a.set(xlim=(0, 100), ylim=(0, 100))
a.axis("off")
INK, MUTED, BOXFC = "#222222", "#6f6f6f", "#fbfbfa"
FS_T, FS_S = 11, 10  # box titles, box sublabels


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    ec: str = "#444444",
    fc: str = BOXFC,
    lw: float = 1.4,
) -> None:
    a.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            boxstyle="round,pad=0.4,rounding_size=2.2",
            zorder=2,
        )
    )


def arrow(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    col: str = "#555555",
    lw: float = 1.5,
) -> None:
    a.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=col,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


XS, MID_Y, MID_H = (15, 50, 85), 52, 34
MID_W = (27, 37.5, 27)  # the middle title is the long one

# dimension learning task, with the three things the model takes from it
box(50, 91, 97, 14, ec="#9a9a9a", fc="#f2f2f2")
a.text(
    50,
    95.0,
    "Dimension learning task",
    ha="center",
    va="center",
    fontsize=FS_T,
    fontweight="bold",
    color=INK,
)
for x, lab, col in [
    (XS[0], "taught categories", MUTED),
    (XS[1], r"$\Delta$Value slope", GREYS[2]),
    (XS[2], r"value knowledge $K_s$", K_COL),
]:
    a.text(x, 87.6, lab, ha="center", va="center", fontsize=FS_S + 0.5, color=col)
    arrow(
        (x, 83.4),
        (x, MID_Y + MID_H / 2 + 1.2),
        col=col,
        lw=1.4 if col == MUTED else 1.9,
    )
# second DeltaValue-slope arrow, into the weights box
arrow(
    (XS[1] + 4, 83.4),
    (XS[2] - MID_W[2] / 2 + 2, MID_Y + MID_H / 2 + 1.2),
    col=GREYS[2],
    lw=1.9,
)

for x, w in zip(XS, MID_W):
    box(x, MID_Y, w, MID_H)
# no arrows between the three boxes: all three feed choice in parallel
TY, SY = MID_Y + 12.6, MID_Y - 13.2  # title / sublabel rows inside the three boxes
GY0, GY1 = MID_Y - 8.0, MID_Y + 6.0  # glyph band

# 1. value map: five taught levels, each band sloped (graded, gamma_d > 0)
a.text(
    XS[0],
    TY,
    "Value map",
    ha="center",
    va="center",
    fontsize=FS_T,
    fontweight="bold",
    color=INK,
)
gx0, gw = XS[0] - 9.5, 19.0
step = (GY1 - GY0) / 5
for k in range(
    5
):  # grey = the taught staircase, black = the graded map read inside each band
    yc = GY0 + (k + 0.5) * step
    xl, xr = gx0 + k * gw / 5, gx0 + (k + 1) * gw / 5
    a.plot([xl, xr], [yc, yc], color="#b5b5b5", lw=2.0, solid_capstyle="butt", zorder=3)
    if k:  # riser of the taught staircase
        a.plot([xl, xl], [yc - step, yc], color="#b5b5b5", lw=2.0, zorder=3)

# black = the fitted map: graded through each band rather than flat within it
a.plot(
    [gx0 + 0.4, gx0 + gw - 0.4],
    [GY0 + 0.15 * step, GY1 - 0.15 * step],
    color=INK,
    lw=2.2,
    solid_capstyle="round",
    zorder=4,
)
a.plot([gx0, gx0 + gw], [GY0 - 0.6] * 2, color="#9a9a9a", lw=0.9, zorder=3)
a.text(
    XS[0],
    SY,
    r"graded, $\gamma_d > 0$",
    ha="center",
    va="center",
    fontsize=FS_S,
    color=MUTED,
)

# 2. magnitude reliance: the contribution profile in miniature (panel C's grammar)
a.text(
    XS[1],
    TY,
    r"Magnitude reliance $\lambda_s$",
    ha="center",
    va="center",
    fontsize=FS_T - 0.5,
    fontweight="bold",
    color=INK,
)
gx0, gw = XS[1] - 5.5, 19.0
u = np.linspace(0.25, 1.0, 20)
gxy = lambda uu, cc: (
    gx0 + (uu - 0.25) / 0.75 * gw,
    GY0 + (cc - 0.25) / 0.75 * (GY1 - GY0),
)
a.plot(*gxy(u, np.ones_like(u)), color=POLE_TALLY, ls=(0, (3, 2)), lw=1.9, zorder=4)
a.plot(*gxy(u, u), color=POLE_MAG, ls=(0, (3, 2)), lw=1.9, zorder=4)
a.plot(*gxy(u, 0.45 + 0.55 * u), color=INK, lw=2.0, zorder=5)
a.plot([gx0, gx0 + gw], [GY0 - 0.6] * 2, color="#9a9a9a", lw=0.9, zorder=3)
a.plot([gx0, gx0], [GY0 - 0.6, GY1 + 1], color="#9a9a9a", lw=0.9, zorder=3)
a.text(
    gx0 - 1.0,
    GY1,
    "direction",
    ha="right",
    va="center",
    fontsize=FS_S - 1.5,
    color=POLE_TALLY,
)
a.text(
    gx0 - 1.0, GY0, "size", ha="right", va="center", fontsize=FS_S - 1.5, color=POLE_MAG
)
a.text(
    gx0 + gw / 2,
    SY,
    "points won by",
    ha="center",
    va="center",
    fontsize=FS_S - 1,
    color=MUTED,
)

# 3. dimension weights: one bar per dimension (Fig 1's dimension colors)
a.text(
    XS[2],
    TY,
    r"Weights $w_{sd}$",
    ha="center",
    va="center",
    fontsize=FS_T,
    fontweight="bold",
    color=INK,
)
for i, (d, h) in enumerate(zip(("color", "shape", "line"), (0.55, 1.0, 0.32))):
    a.add_patch(
        Rectangle(
            (XS[2] - 7.8 + i * 5.6, GY0),
            4.4,
            h * (GY1 - GY0),
            facecolor=DIM_COLORS[d],
            edgecolor="none",
            zorder=4,
        )
    )
a.plot([XS[2] - 9.5, XS[2] + 9.5], [GY0 - 0.6] * 2, color="#9a9a9a", lw=0.9, zorder=3)
a.text(XS[2], SY, r"gain $|W_s|$", ha="center", va="center", fontsize=FS_S, color=MUTED)

# choice
box(50, 12.5, 97, 15, ec="#111111", fc="#f4f4f4", lw=1.6)
for x in XS:
    arrow((x, MID_Y - MID_H / 2 - 1.0), (x, 20.6))
a.text(
    50,
    16.8,
    "Choice",
    ha="center",
    va="center",
    fontsize=FS_T,
    fontweight="bold",
    color=INK,
)
a.text(
    50,
    9.0,
    r"$P(\mathrm{right}) = \mathrm{logit}^{-1}(a_s + \Sigma_d\, w_{sd}\,"
    r"[(1-\lambda_s)\,\mathrm{sign}(\Delta k_d) + \lambda_s\,\Delta u_d])$",
    ha="center",
    va="center",
    fontsize=FS_S + 0.5,
    color="#333333",
)

# ── B: gain vs value knowledge ─────────────────────────────────────────────────────────
gr = np.linspace(K.min(), K.max(), 60)
mu, lo, hi = boot_line(K, gain, gr, seed=1)
axB.fill_between(gr, lo, hi, color="0.75", alpha=0.35, lw=0)
axB.plot(gr, mu, color="k", lw=2, zorder=4)
axB.scatter(K, gain, s=22, c="#333333", alpha=0.45, edgecolor="none")
axB.set(xlabel="value knowledge  (z)", ylabel=r"gain  $|W_s|$")

# ── C: contribution profile, observed against the two pure reads ───────────────────────
RP = json.load(open(f"{FITS}/readout_profile.json"))
for k, col, ls, lab in [
    ("map_only", POLE_MAG, "--", r"map only ($\lambda$ = 1)"),
    ("tally_only", POLE_TALLY, "--", r"tally only ($\lambda$ = 0)"),
    ("model_of_record", "#333333", "-", "model of record"),
]:
    axC.fill_between(POINTS, RP[k]["lo"], RP[k]["hi"], color=col, alpha=0.2, lw=0)
    axC.plot(POINTS, RP[k]["mean"], color=col, ls=ls, lw=2.2, label=lab)
o = {k: np.array(v) for k, v in RP["observed"].items()}
axC.errorbar(
    POINTS,
    o["mean"],
    yerr=[o["mean"] - o["lo"], o["hi"] - o["mean"]],
    fmt="o",
    color=LINE,
    ms=7.5,
    lw=2,
    label="observed",
    zorder=5,
)
profile_axes(axC, top=1.16, fs=FS_TICK)
axC.legend(
    fontsize=FS_TICK - 3.5,
    frameon=False,
    loc="lower right",
    bbox_to_anchor=(1.08, -0.03),
    handlelength=1.6,
)

# ── D: the same profile by DeltaValue-slope tertile (learning phase only) ──────────────
SP = json.load(open(f"{FITS}/slope_readout_profile.json"))
tertile_profile(
    axD,
    SP["slope"]["tertiles"],
    ["shallow", "middle", "steep"],
    GREYS,
    r"$\Delta$Value slope",
    fs=FS_TICK - 1,
)
axD.xaxis.label.set_size(FS_TICK)
axD.yaxis.label.set_size(FS_TICK)
# inset in place of a legend: each tertile's fitted learning psychometric (cactus_paper.features.learning_slope --cells),
# averaged over dimensions and participants. Model-implied only, no raw points. Tertiles are cut as
# in cactus_paper.analyses.contribution.slope_readout_profile (load_design subject order).
axD.get_legend().remove()
_slope = np.load(f"{FITS}/map_slope_draws.npy").mean(0)
_tert = np.digitize(_slope, np.quantile(_slope, [1 / 3, 2 / 3]))
_cells = np.load(f"{FITS}/map_slope_cells.npz")
assert list(_cells["run_id"]) == D["cd"].run_id.drop_duplicates().tolist()
axI = axD.inset_axes([0.62, 0.15, 0.36, 0.30])
dgrid = np.linspace(25, 100, 60)
_z = (dgrid / 25 - float(_cells["dist_mean"])) / float(_cells["dist_sd"])
_fit = 1 / (
    1 + np.exp(-(_cells["level"][..., None] + _cells["slope"][..., None] * _z))
)  # nS x 3 x grid
for g in range(3):
    axI.plot(dgrid, _fit[_tert == g].mean((0, 1)), color=GREYS[g], lw=1.8)
    axI.plot(
        dgrid[[0, -1]],
        _fit[_tert == g].mean((0, 1))[[0, -1]],
        ls="none",
        marker="os^"[g],
        color=GREYS[g],
        ms=4,
    )
axI.set(xticks=POINTS, xlim=(18, 107))
axI.set_xticklabels(["25", "", "", "100"])
axI.tick_params(labelsize=FS_TICK - 4.5, length=2.5, pad=1.5)
axI.set_xlabel(r"$\Delta$Value (learning)", fontsize=FS_TICK - 4.5, labelpad=0)
axI.set_ylabel("P(correct)", fontsize=FS_TICK - 4.5, labelpad=1)
sns.despine(ax=axI)

for ax in (axB, axC, axD):
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_TICK)
    ax.yaxis.label.set_size(FS_TICK)
axC.yaxis.set_label_coords(-0.2, 0.5)
axD.yaxis.set_label_coords(-0.2, 0.5)
sns.despine(fig)
LX, LY = -0.30, 1.13  # panel-letter offset, in each data panel's axes fraction
for ax, lab in [(axB, "B"), (axC, "C"), (axD, "D")]:
    panel_label(ax, lab, fs=FS_PANEL, x=LX, y=LY)
# widen A leftward so its letter lines up with C's (x) and B's (y); pin its top to B's
pA, pB, pC = axA.get_position(), axB.get_position(), axC.get_position()
lx, ly = pC.x0 + LX * pC.width, pB.y0 + LY * pB.height
x0 = lx + 0.035
axA.set_position([x0, pB.y1 - pA.height, pA.x1 - x0, pA.height])
pA = axA.get_position()
panel_label(
    axA, "A", fs=FS_PANEL, x=(lx - pA.x0) / pA.width, y=(ly - pA.y0) / pA.height
)
fig.savefig(OUT, dpi=350, bbox_inches="tight")
shutil.copy(OUT, ROOT / "manuscript" / "images" / "fig_ms_model.png")
print("saved ->", OUT, "(+ manuscript/images)")
