"""fig_ms5_crossstream.py — manuscript Fig 5: the choice-model constructs predict distinct
properties of information search, none of which they were fit on.

One row per search property: example card on the left, person-level added-variable scatter in the
middle, the joint model's forest on the right.

  A  worked examples (cactus_paper.visualization.cards): an option-wise trace (Payne > 0) beside row 1 and an
     attribute-wise trace (Payne < 0) beside row 2.
  B  lambda -> Payne index, added-variable (gain and concentration partialled out).
  C  the style forest: payne ~ lam + gain + conc, jointly. Filled = 95% HDI excludes zero. Bottom
     rows are the within-model contrasts |lam| - |conc| and |lam| - |gain|.
  D  concentration -> sampling depth, added-variable (lambda and gain partialled out).
  E  the amount forest: depth ~ lam + gain + conc. Bottom rows are |conc| - |lam| and |conc| - |gain|.

Each forest has its own x-scale (Payne index units on top, z-scored depth below): coefficients are
comparable within a forest, not between them.

Plot only. The forests and added-variable bands read results/fits/fig5_grid_draws.npz, written
(with the fig5_grid.json make_stats.py reads) by `python -m cactus_paper.analyses.crossstream`.

Run: pixi run python scripts/exp2/figures/fig_ms5_crossstream.py  (any cwd)
"""

import json
from typing import Optional, Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from cactus_paper.analyses.crossstream import (
    CONTRASTS,
    GRID_DRAWS,
    GRID_JSON,
    crossstream_frame,
)
from cactus_paper.design import load_design, pole_score
from cactus_paper.paths import DATA
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.cards import (
    XL as _CXL,
)
from cactus_paper.visualization.cards import (
    YL as _CYL,
)
from cactus_paper.visualization.cards import (
    add_trajectory_colorbar,
    load_trajectory_pair,
    plot_trajectory,
)
from cactus_paper.visualization.plotting import panel_label, style_ms

OUT = figpath("fig_ms5_crossstream.png")

D = load_design()
su = D["cd"].run_id.drop_duplicates().tolist()
sub = crossstream_frame(D)  # person-level payne, z(depth), z(lam), z(gain), z(conc)
DR = np.load(GRID_DRAWS)  # "<style|amount>_<coef or contrast>" posterior draws
grid = json.load(open(GRID_JSON))

ROWS = [
    ("lam", "magnitude reliance  $\\lambda$"),
    ("gain", "gain  $|W_s|$"),
    ("conc", "concentration"),
]
CONTRAST_LABELS = {
    "lam_minus_conc": "$|\\lambda| - |$conc$|$",
    "lam_minus_gain": "$|\\lambda| - |$gain$|$",
    "conc_minus_lam": "$|$conc$| - |\\lambda|$",
    "conc_minus_gain": "$|$conc$| - |$gain$|$",
}
# one forest per model: (key, x label, [(contrast name, tick label)])
COLS = [
    (key, xlab, [(c, CONTRAST_LABELS[c]) for c, _a, _b in CONTRASTS[key]])
    for key, xlab in [
        ("style", "coefficient on\nPayne index (+ option-wise)"),
        ("amount", "coefficient on\nsampling depth (z)"),
    ]
]
hdi = lambda d: az.hdi(np.asarray(d), hdi_prob=0.95)

# ── layout ────────────────────────────────────────────────────────────────────────
# canvas width sets printed text size: the text block is ~6.1in and manuscript figures save at
# ~9.6in wide (measure the PNG; bbox_inches="tight" trims). Changing figsize rescales every glyph.
LINE, DOT = style_ms()
FS_PANEL = 24
# tick size chosen to print at ~8.6pt, matching the rc axis labels
FS_TICK = 14.5
FIGW = 9.60
FIGH = 7.30
GRID_TOP, GRID_BOTTOM = 0.95, 0.11  # fractions; frees inch math for the card column
fig = plt.figure(figsize=(FIGW, FIGH))
CARD_W = 1.85  # the example cards, same physical size as fig2's
_CARD_RATIO = (_CYL[1] - _CYL[0]) / (_CXL[1] - _CXL[0])
CARD_H = CARD_W * _CARD_RATIO
CARD_X = 0.35  # left margin, room for the "A" letter
# 2.0in gap between the card column and the grid: clears B's two-line y-tick labels and y-label;
# B/C x-labels are wrapped onto two lines so they do not collide
GRID_LEFT = (CARD_X + CARD_W + 2.0) / FIGW
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.4, 1.0],
    wspace=0.45,
    hspace=0.55,
    left=GRID_LEFT,
    right=0.99,
    top=GRID_TOP,
    bottom=GRID_BOTTOM,
)
axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axD, axE = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# ── A: worked examples, one card per row, aligned with that row's vertical span ────────────
_row_h = (
    (GRID_TOP - GRID_BOTTOM) / (2 + 1 * 0.55) * FIGH
)  # matches gs's 2 equal rows, hspace=0.55
_row1_top, _row1_bot = GRID_TOP * FIGH, GRID_TOP * FIGH - _row_h
_row2_top = _row1_bot - 0.55 * _row_h
_row2_bot = _row2_top - _row_h
axA1 = fig.add_axes(
    [
        CARD_X / FIGW,
        (_row1_bot + (_row_h - CARD_H) / 2) / FIGH,
        CARD_W / FIGW,
        CARD_H / FIGH,
    ]
)
axA2 = fig.add_axes(
    [
        CARD_X / FIGW,
        (_row2_bot + (_row_h - CARD_H) / 2) / FIGH,
        CARD_W / FIGW,
        CARD_H / FIGH,
    ]
)
for ax in (axA1, axA2):
    ax.set_xlim(*_CXL)
    ax.set_ylim(*_CYL)
    ax.set_aspect("auto")
    ax.set_autoscale_on(False)
    ax.axis("off")
row_opt, row_attr = load_trajectory_pair(str(DATA), su)
plot_trajectory(row_opt, axA1, f"Payne = {row_opt.payne_index:+.2f}\noption-wise")
plot_trajectory(row_attr, axA2, f"Payne = {row_attr.payne_index:+.2f}\nattribute-wise")
add_trajectory_colorbar(axA2, fontsize=FS_TICK - 3, orientation="horizontal")
fig.text(
    (CARD_X - 0.30) / FIGW,
    (_row1_top + 0.35) / FIGH,
    "A",
    fontsize=FS_PANEL,
    fontweight="bold",
    va="top",
    ha="left",
)
print(
    f"  A example traces: option-wise payne={row_opt.payne_index:+.2f}  "
    f"attribute-wise payne={row_attr.payne_index:+.2f}"
)

# ── C, E: the two forests ─────────────────────────────────────────────────────────
YPOS = {"lam": 3.0, "gain": 2.0, "conc": 1.0, "contrast": -0.25}
for ax, (key, xlab, cons) in zip((axC, axE), COLS):
    ax.axvline(0, color="0.75", lw=1.0, zorder=0)
    for n, _lab in ROWS:
        d = DR[f"{key}_{n}"]
        lo, hi = hdi(d)
        cred = not (lo < 0 < hi)
        ax.plot(
            [lo, hi], [YPOS[n]] * 2, color=LINE, lw=2.0, solid_capstyle="butt", zorder=2
        )
        ax.plot(
            d.mean(),
            YPOS[n],
            "o",
            ms=8,
            zorder=3,
            color=LINE if cred else "white",
            markeredgecolor=LINE,
            markeredgewidth=1.6,
        )
    for ci, (cname, _clab) in enumerate(cons):
        d = DR[f"{key}_{cname}"]
        lo, hi = hdi(d)
        cred = not (lo < 0 < hi)
        y = YPOS["contrast"] - ci * 0.95
        ax.plot(
            [lo, hi], [y] * 2, color="0.45", lw=2.0, solid_capstyle="butt", zorder=2
        )
        ax.plot(
            d.mean(),
            y,
            "o",
            ms=8,
            zorder=3,
            color="0.45" if cred else "white",
            markeredgecolor="0.45",
            markeredgewidth=1.6,
        )
    ax.axhline(0.35, color="0.85", lw=0.8, zorder=0)  # separates contrasts from cells
    ax.set_ylim(-1.85, 3.6)
    ax.set_yticks(
        [YPOS[n] for n, _ in ROWS]
        + [YPOS["contrast"] - ci * 0.95 for ci in range(len(cons))]
    )
    ax.set_yticklabels([lab for _, lab in ROWS] + [c[1] for c in cons])
    ax.yaxis.tick_right()  # outer edge: keeps the scatter/forest gutter clean
    ax.set_xlabel(xlab.replace("\n", " "))
    ax.tick_params(labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=6)


# ── B, D: the two cells as people ─────────────────────────────────────────────────
def resid_multi(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y on a design matrix X (columns already z-scored), intercept included."""
    M = np.column_stack([np.ones(len(y)), X])
    return y - M @ np.linalg.lstsq(M, y, rcond=None)[0]


def av_panel(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    draws: np.ndarray,
    xlab: str,
    ylab: str,
    yticks: Optional[Sequence[float]] = None,
    yticklabels: Optional[Sequence[str]] = None,
) -> None:
    """Added-variable panel: residualized data, with the posterior of the focal coefficient as the
    line and band (so the band is the MODEL's uncertainty, not a re-fit of the residuals).
    """
    g = np.linspace(np.nanmin(x), np.nanmax(x), 60)
    cur = draws[:, None] * g[None, :]
    lo, hi = np.percentile(cur, [2.5, 97.5], 0)
    ax.fill_between(g, lo, hi, color=LINE, alpha=0.18, lw=0)
    ax.plot(g, cur.mean(0), color=LINE, lw=2)
    ax.scatter(x, y, s=26, c="#333333", alpha=0.45, edgecolor="none", zorder=3)
    ax.axhline(0, ls=":", c="0.6", lw=0.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.yaxis.set_label_coords(-0.115, 0.5)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    ax.tick_params(labelsize=FS_TICK)
    ax.tick_params(axis="y", pad=3)


lz, gz, cz = sub.lam.values, sub.gain.values, sub.conc.values
pay, dep = sub.payne.values, sub.depth.values

Xa = np.column_stack([gz, cz])  # everything except lam
xrA, yrA = resid_multi(lz, Xa), resid_multi(pay, Xa)
av_panel(
    axB,
    xrA,
    yrA,
    DR["style_lam"],
    "magnitude reliance\n(gain, conc. partialled out)",
    "Payne index  |  gain, conc.",
    [-0.5, -0.25, 0, 0.25, 0.5],
    ["attribute-\nwise", "\u22120.25", "0", "0.25", "option-\nwise"],
)
B_YLABEL_X = -0.37  # B's tick labels are wide (two-line words)
axB.yaxis.set_label_coords(B_YLABEL_X, 0.5)

Xc = np.column_stack([lz, gz])  # everything except conc
av_panel(
    axD,
    resid_multi(cz, Xc),
    resid_multi(dep, Xc),
    DR["amount_conc"],
    "concentration\n($\\lambda$, gain partialled out)",
    "sampling depth (z)  |  $\\lambda$, gain",
)
axD.yaxis.set_label_coords(B_YLABEL_X, 0.5)  # matched to B so the two stacked ylabels
# in this column align vertically

for ax, lab in [(axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")]:
    scat = ax in (axB, axD)
    panel_label(
        ax, lab, fs=FS_PANEL, x=-0.22 if scat else -0.10, y=1.14 if scat else 1.12
    )
sns.despine(fig)
for ax in (axC, axE):
    sns.despine(ax=ax, left=True, right=False)

fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved ->", OUT)

# ── console readout ───────────────────────────────────────────────────────────────
print("\nthe grid (joint models, z-scored predictors; * = 95% HDI excludes zero)")
for key, xlab, cons in COLS:
    print(f"  {key.upper():7s} {xlab.splitlines()[0]}")
    for n, lab in list(ROWS) + [(c[0], c[0]) for c in cons]:
        v = grid[key][n]
        star = "" if v["lo"] < 0 < v["hi"] else "  *"
        print(
            f"    {n:16s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  "
            f"P(dir)={v['pdir']:.3f}{star}"
        )

_pole = pole_score(D)
print("\npole correlations (not plotted; for the text only)")
print(f"  pole ~ Payne residual   r = {np.corrcoef(_pole, yrA)[0, 1]:+.3f}")
print(f"  pole ~ lam residual     r = {np.corrcoef(_pole, xrA)[0, 1]:+.3f}")
print("\nraw correlations for reference")
for nm, a, b in [
    ("lam ~ Payne", lz, pay),
    ("gain ~ depth", gz, sub.depth.values),
    ("conc ~ depth", cz, sub.depth.values),
    ("gain ~ conc", gz, cz),
]:
    print(f"  {nm:14s} r={np.corrcoef(a, b)[0, 1]:+.3f}")
