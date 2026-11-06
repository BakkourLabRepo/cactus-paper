"""fig_ms_slope_readout.py — the contribution profile split by learning-phase tertiles.

  A  profile (fig_ms_readout panel A estimator) by tertile of the learning distance-accuracy slope
     (per participant, mean over the three dimensions; results/fits/map_slope_draws.npy from
     cactus_paper.features.learning_slope).
  B  the same by tertile of overall learning accuracy K.

Whole-curve summary = tally lift: mean of beta_j / beta_4 over j = 1..3 (tally 1, linear map read
by size 0.5), reported as high minus low tertile, participant bootstrap.

Plot only. Reads results/fits/slope_readout_profile_draws.npz, written (with the
slope_readout_profile.json that fig_ms_model.py panel D and make_stats.py read) by
`python -m cactus_paper.analyses.contribution slope-readout`.

Run: pixi run python scripts/exp2/figures/fig_ms_slope_readout.py  (any cwd)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cactus_paper.analyses.contribution import SLOPE_DRAWS, J
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import panel_label, style_ms

OUT = figpath("fig_ms_slope_readout.png")
LINE, DOT = style_ms()
FS_PANEL, FS_TICK, FS_LAB = 24, 12, 12
COLS = ["#6baed6", "#2171b5", "#08306b"]  # ordered Blues
MARK = ["o", "s", "^"]

A = np.load(SLOPE_DRAWS)
splits = {
    k: [
        {"profile": A[f"{k}_{g}_profile"], "boot": A[f"{k}_{g}_boot"]} for g in range(3)
    ]
    for k in ("slope", "K")
}

fig = plt.figure(figsize=(9.6, 4.0))
gs = fig.add_gridspec(1, 2, wspace=0.32)
panels = [
    ("slope", ["shallow", "middle", "steep"], "learning slope"),
    ("K", ["low", "middle", "high"], "learning accuracy"),
]
for i, (key, labs, ltitle) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(J, np.ones(4), color="0.6", ls=(0, (1, 2)), lw=1.3, zorder=0)
    for g, r in enumerate(splits[key]):
        m = r["profile"]
        lo, hi = np.percentile(r["boot"], [2.5, 97.5], 0)
        ax.errorbar(
            J + (g - 1) * 0.09,
            m,
            yerr=[m - lo, hi - m],
            color=COLS[g],
            marker=MARK[g],
            ms=7,
            lw=2,
            label=labs[g],
        )
    ax.set(xticks=J, xlim=(0.7, 4.3), ylim=(0, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("levels won by", fontsize=FS_LAB)
    if i == 0:
        ax.set_ylabel("contribution to choice\n(4-level win = 1)", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(
        title=ltitle,
        fontsize=FS_TICK - 2,
        title_fontsize=FS_TICK - 2,
        frameon=False,
        loc="lower right",
        handlelength=1.8,
    )
    panel_label(ax, "AB"[i], fs=FS_PANEL, x=-0.26 if i == 0 else -0.16, y=1.12)
sns.despine(fig)
fig.savefig(OUT, dpi=350, bbox_inches="tight")
print("saved ->", OUT)
