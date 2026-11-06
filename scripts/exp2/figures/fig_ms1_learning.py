"""fig_ms1_learning.py — Fig 1C: learning-task performance.

  LEFT   P(correct) by trial within the attribute's block (1-40), one line per attribute, shaded
         95% CI across participants. Dashed verticals mark the prompt-direction reversals ("which
         is bigger" / "which is smaller" alternates every 10 trials).
  RIGHT  one psychometric per dimension: P(correct) against |DeltaValue| of the pair. Thin lines =
         each participant's fitted curve (results/fits/map_slope_cells.npz from
         cactus_paper.features.learning_slope); bold line + markers = pooled logistic and participant means
         +/- 95% CI. Shares the left panel's y-axis.

--variant psycho draws the signed learning psychometric instead: P(choose right) against the value
difference favoring right, signed toward the prompt (the prompt fold is required; unfolded, the two
prompt directions cancel). --viridis swaps in a colorblind-safe palette.

Run: pixi run python scripts/exp2/figures/fig_ms1_learning.py [--variant curve|psycho] [--viridis]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cactus_paper.constants import DIMS
from cactus_paper.design import load_design
from cactus_paper.paths import DATA, FITS
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import (
    DIM_COLORS,
    fit_curve,
    panel_label,
    style_ms,
)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["curve", "psycho"], default="curve")
ap.add_argument("--viridis", action="store_true", help="colorblind-safe palette")
A = ap.parse_args()
OUT = figpath(f"fig_ms1_learning_{A.variant}.png")
FLIPS = [10, 20, 30]
# display order (matches panels A/B); `DIMS` is the data order that W, dacc and posteriors use
ORDER = ["color", "line", "shape"]
XS = [-100.0, -75.0, -50.0, -25.0, 25.0, 50.0, 75.0, 100.0]

D = load_design()
su = D["cd"].run_id.drop_duplicates().tolist()
ld = pd.read_csv(DATA / "learning_data.csv")
ld = ld[ld.run_id.isin(su)].dropna(subset=["correct", "left_rew", "right_rew"]).copy()
ld["correct"] = ld["correct"].astype(int)
ld["chose_right"] = (~ld.chose_left.astype(bool)).astype(int)
ld["dv"] = (ld.right_rew - ld.left_rew) * np.where(
    ld.prompt_dir == "smaller", -1.0, 1.0
)

acc = ld.groupby(["run_id", "dim"]).correct.mean().unstack()[DIMS]  # the paper's `dacc`

style_ms()
# default palette from cactus_paper.visualization.plotting (shared with fig_ms5_crossstream)
COL = (
    dict(zip(DIMS, sns.color_palette("viridis", 5)[:3]))
    if A.viridis
    else {d: DIM_COLORS[d] for d in DIMS}
)

if A.variant == "psycho":
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    grid = np.linspace(-100, 100, 160)
    for d in DIMS:
        s = ld[ld.dv.abs().isin([25.0, 50.0, 75.0, 100.0]) & (ld.dim == d)]
        ax.plot(
            grid,
            fit_curve(
                s.dv.values.astype(float), s.chose_right.values.astype(float), grid
            ),
            color=COL[d],
            lw=2.5,
            label=d.capitalize(),
            zorder=3,
        )
        per = (
            s.groupby(["run_id", "dv"]).chose_right.mean().unstack().reindex(columns=XS)
        )
        ax.errorbar(
            XS,
            per.mean(0).values,
            yerr=1.96 * per.apply(lambda c: stats.sem(c.dropna())).values,
            fmt="o",
            color=COL[d],
            ms=6,
            capsize=0,
            elinewidth=2,
            mec="white",
            mew=0.8,
            ls="none",
            zorder=4,
        )
    ax.axhline(0.5, ls=":", c="0.6", lw=0.8)
    ax.axvline(0, ls=":", c="0.6", lw=0.8)
    ax.set(
        xlim=(-112, 112),
        ylim=(0, 1),
        xticks=[-100, -50, 0, 50, 100],
        xlabel="value difference favoring right\n(points, signed toward the prompt)",
        ylabel="P(choose right)",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    panel_label(ax, "C", fs=22)
else:
    # canvas width sets printed text size: the text block is ~6.1in and manuscript figures save at
    # ~9.6in wide (measure the PNG; bbox_inches="tight" trims). Changing figsize rescales every glyph.
    fig = plt.figure(figsize=(11.35, 2.67))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 1.0, 1.0, 1.0], wspace=0.12)
    axL = fig.add_subplot(gs[0])
    axP = [fig.add_subplot(gs[i + 1], sharey=axL) for i in range(3)]

    tr = ld.groupby(["run_id", "dim", "trial_per_dim"]).correct.mean().unstack()
    x = np.arange(40)
    xd = (
        x + 1
    )  # displayed 1-40 (data trial_per_dim is 0-39); the reversal is the 11th / 21st / 31st
    for d in ORDER:
        m = tr.xs(d, level="dim").reindex(columns=x)
        mu = m.mean(0).values
        ci = 1.96 * m.apply(lambda c: stats.sem(c.dropna())).values
        axL.fill_between(xd, mu - ci, mu + ci, color=COL[d], alpha=0.22, lw=0)
        axL.plot(xd, mu, color=COL[d], lw=2, label=d.capitalize())
    for f in FLIPS:  # ON the reversal trial (displayed 11 / 21 / 31), where the dip is
        axL.axvline(f + 1, ls="--", c="0.55", lw=1.1, zorder=0)
    axL.text(
        FLIPS[0] + 1.6,
        0.99,
        "prompt\nreversal",
        fontsize=9,
        color="0.4",
        ha="left",
        va="top",
        linespacing=1.05,
    )
    axL.axhline(0.5, ls="--", c="0.7", lw=0.9, zorder=0)
    axL.set(
        xlim=(0.2, 40.8),
        ylim=(0.30, 1.02),
        xticks=[1, 11, 21, 31, 40],
        xlabel="trial (within dimension)",
        ylabel="P(correct)",
    )
    axL.legend(
        frameon=False,
        loc="lower right",
        ncol=3,
        fontsize=10,
        columnspacing=0.8,
        handlelength=1.1,
        handletextpad=0.4,
    )

    # psychometric per dimension. Thin lines: each participant's fitted curve (cactus_paper.features.learning_slope
    # --cells, partially pooled). Bold line + markers: pooled logistic and participant means +/- 95% CI.
    cells = np.load(FITS / "map_slope_cells.npz")
    assert list(cells["run_id"]) == su and list(cells["dims"]) == DIMS
    ld["absdv"] = (ld.right_rew - ld.left_rew).abs()
    lp = ld[ld.absdv.isin([25.0, 50.0, 75.0, 100.0])]
    grid = np.linspace(25, 100, 60)
    zg = (grid / 25 - float(cells["dist_mean"])) / float(cells["dist_sd"])
    for ax, d in zip(axP, ORDER):
        j_ = DIMS.index(d)
        fitted = 1 / (
            1 + np.exp(-(cells["level"][:, [j_]] + cells["slope"][:, [j_]] * zg))
        )
        for row in fitted:
            ax.plot(grid, row, color=COL[d], lw=0.6, alpha=0.18, zorder=2)
        s_d = lp[lp.dim == d]
        ax.plot(
            grid,
            fit_curve(s_d.absdv.values, s_d.correct.values.astype(float), grid),
            color="k",
            lw=2.2,
            zorder=4,
        )
        ps = (
            s_d.groupby(["run_id", "absdv"])
            .correct.mean()
            .groupby("absdv")
            .agg(["mean", "sem"])
        )
        ax.errorbar(
            ps.index,
            ps["mean"],
            yerr=1.96 * ps["sem"],
            fmt="o",
            color=COL[d],
            ms=6,
            mec="k",
            mew=0.6,
            lw=1.6,
            capsize=0,
            ls="none",
            zorder=5,
        )
        ax.axhline(0.5, ls="--", c="0.7", lw=0.9, zorder=0)
        ax.set(xlim=(15, 110), xticks=[25, 50, 75, 100])
        ax.tick_params(labelleft=False)
        ax.set_title(
            d.capitalize(), fontsize=11, color=COL[d], fontweight="bold", pad=3
        )
    axP[1].set_xlabel(r"$|\Delta$Value$|$ of the pair (points)", fontsize=11)
    # letter left of the y-label, to line up with A in manuscript/images/fig1_learning.png
    panel_label(axL, "C", fs=22, x=-0.24)

sns.despine(fig)
fig.savefig(OUT, dpi=350, bbox_inches="tight")
print("saved ->", OUT)

print("\nper-participant attribute accuracy (= the paper's dacc)")
for d in DIMS:
    v = acc[d].dropna()
    print(
        f"  {d:6s} mean {v.mean():.3f}  sd {v.std():.3f}  range {v.min():.2f}-{v.max():.2f}"
    )
sp = acc.max(1) - acc.min(1)
print(
    f"  within-participant spread (best - worst): mean {sp.mean():.3f} sd {sp.std():.3f}"
    "   <- beat 1"
)
mode = acc.values.argmax(1)
print(
    f"  modal best attribute = {DIMS[np.bincount(mode).argmax()]}; "
    f"{100 * (mode != np.bincount(mode).argmax()).mean():.0f}% of participants peak elsewhere"
)
pre = ld[ld.trial_per_dim.isin([f - 1 for f in FLIPS])].correct.mean()
post = ld[ld.trial_per_dim.isin(FLIPS)].correct.mean()
print(
    f"\n  prompt-flip cost: {pre:.3f} on the trial before a switch -> {post:.3f} on the "
    f"switch trial ({post - pre:+.3f})"
)
