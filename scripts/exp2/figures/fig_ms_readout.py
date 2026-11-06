"""fig_ms_readout.py — readout figure, and the contribution-profile JSON fig_ms_model.py panel C reads.

  A  contribution by level difference (cactus_paper.analyses.contribution): how much winning one dimension
     by 1..4 taught levels pushes choice, normalized to a 4-level win. Observed (black, participant
     bootstrap 95% CI) against simulated accounts run through the real trial set and scored with the
     same estimator: value map only (lambda forced to 1), tally only (lambda forced to 0) and the
     model of record.
  B  the strategy plane, lambda x concentration (Gini |w|), one point per participant, classical
     strategies at the limits (same drawing as fig_ms4_model.py).

Plot only. Panel A reads results/fits/readout_profile_draws.npz, written (with the
readout_profile.json that fig_ms_model.py and make_stats.py read) by
`python -m cactus_paper.analyses.contribution readout`.

Run: pixi run python scripts/exp2/figures/fig_ms_readout.py  (any cwd)
"""

import arviz as az
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cactus_paper.analyses.contribution import READOUT_DRAWS, J
from cactus_paper.features.traits import concentration as wl_conc
from cactus_paper.paths import MOR_IDATA
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import panel_label, style_ms

OUT = figpath("fig_ms_readout.png")
LINE, DOT = style_ms()
FS_PANEL, FS_TICK, FS_LAB, FS_CORNER = 24, 12, 12, 12

A = np.load(READOUT_DRAWS)
obs, boot = A["observed"], A["boot"]
prof = {k: A[k] for k in ("map", "mor", "tally")}

po = az.from_netcdf(MOR_IDATA).posterior
nS = po["lam_s"].shape[-1]
absW = np.abs(po["W"].values.reshape(-1, nS, 3).mean(0))
lam_s = po["lam_s"].values.reshape(-1, nS).mean(0)
conc = wl_conc(absW)

fig = plt.figure(figsize=(9.6, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.38)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# ── A: contribution by level difference ────────────────────────────────────────────────
# colors as in fig3_poles: magnitude pole = orange, tally pole = purple
STYLE = [
    ("map", "#b35806", "--", r"value map only ($\lambda$ = 1)"),
    ("tally", "#542788", "--", r"tally only ($\lambda$ = 0)"),
    ("mor", "#333333", "-", "model of record"),
]
for k, col, ls, lab in STYLE:
    lo, hi = np.percentile(prof[k], [2.5, 97.5], 0)
    axA.fill_between(J, lo, hi, color=col, alpha=0.2, lw=0)
    axA.plot(J, prof[k].mean(0), color=col, ls=ls, lw=2.2, label=lab)
lo, hi = np.percentile(boot, [2.5, 97.5], 0)
axA.errorbar(
    J,
    obs,
    yerr=[obs - lo, hi - obs],
    fmt="o",
    color=LINE,
    ms=7.5,
    lw=2,
    label="observed",
    zorder=5,
)
axA.set(xticks=J, xlim=(0.8, 4.2), ylim=(0, 1.16), yticks=[0, 0.25, 0.5, 0.75, 1])
axA.spines["left"].set_bounds(0, 1)  # headroom for the tally band above 1
axA.set_xlabel("levels won by", fontsize=FS_LAB)
axA.set_ylabel("contribution to choice\n(4-level win = 1)", fontsize=FS_LAB)
axA.legend(fontsize=FS_TICK - 2, frameon=False, loc="lower right", handlelength=1.8)
axA.tick_params(labelsize=FS_TICK)

# ── B: the strategy plane (fig_ms4_model.py drawing) ───────────────────────────────────
axB.scatter(lam_s, conc, s=22, c="#333333", alpha=0.45, edgecolor="none")
axB.set_xlabel(r"magnitude reliance  $\lambda_s$", fontsize=FS_LAB)
axB.set_ylabel(r"concentration  (Gini $|w|$)", fontsize=FS_LAB)
axB.set_xlim(-0.015, 1.135)
axB.set_ylim(-0.055, 0.70)
axB.spines["bottom"].set_bounds(0, 1)
axB.spines["left"].set_bounds(0, 0.6)
axB.plot([0, 1], [2 / 3] * 2, color="0.55", ls=(0, (4, 3)), lw=1.1, zorder=1)
_corner = dict(
    fontsize=FS_CORNER,
    style="italic",
    color="0.35",
    zorder=4,
    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")],
)
axB.text(0.0, 0.0, "TALLY", ha="left", va="bottom", **_corner)
axB.text(0.0, 2 / 3, "TTB", ha="left", va="top", **_corner)
axB.text(1.13, -0.052, "WADD", ha="right", va="bottom", **_corner)
axB.tick_params(labelsize=FS_TICK)

sns.despine(fig)
panel_label(axA, "A", fs=FS_PANEL, x=-0.30, y=1.12)
panel_label(axB, "B", fs=FS_PANEL, x=-0.22, y=1.12)
fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved ->", OUT)
print(f"observed         {obs.round(3)}")
for k, *_ in STYLE:
    print(f"{k:16s} {prof[k].mean(0).round(3)}")
