"""fig_supp_knowledge.py — supplement figure: knowledge and the weights / the read.

  A  per-dimension weight |w_sd|, double-centered (subject and dimension means removed), by the
     participant's own learning-accuracy rank (worst / middle / best); one line per participant,
     black group mean +/- 95% CI.
  B  contribution profile by tertile of overall learning accuracy K, on the shared points-won /
     broken-y axes.

Input: results/fits/slope_readout_profile.json (`python -m cactus_paper.analyses.contribution slope-readout`).

Run: pixi run python scripts/exp2/figures/fig_supp_knowledge.py  (any cwd)
"""

import json
import shutil

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from cactus_paper.design import load_design
from cactus_paper.paths import FITS, MOR_IDATA, ROOT
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import panel_label, style_ms, tertile_profile

OUT = figpath("fig_supp_knowledge.png")
LINE, DOT = style_ms()
FS_PANEL, FS_TICK = 24, 13.5

D = load_design()
nS, K, dacc = D["nS"], np.asarray(D["K"], float), D["dacc"]
po = az.from_netcdf(MOR_IDATA).posterior
absW = np.abs(po["W"].values.reshape(-1, nS, 3).mean(0))

# double-centering, identical to make_stats.py's _dc2_arr (subject means, dimension means, grand)
dc = lambda M: M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()

fig = plt.figure(figsize=(8.4, 3.9))
gs = fig.add_gridspec(1, 2, wspace=0.46)
axB, axD = (
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
)  # axB = panel A, axD = panel B

# ── B: the read by K tertile (slope_readout_profile.json) ──────────────────────────────
SP = json.load(open(f"{FITS}/slope_readout_profile.json"))
BLUES = ["#6baed6", "#2171b5", "#08306b"]  # K: same ramp as fig3_poles
tertile_profile(
    axD,
    SP["K"]["tertiles"],
    ["low", "middle", "high"],
    BLUES,
    "value knowledge",
    fs=FS_TICK - 1,
)

# ── A: within participants, as a ladder over the participant's own accuracy rank ──
xC, yC = dc(dacc), dc(absW)  # double-centered
rB = stats.pearsonr(xC.ravel(), yC.ravel())  # double-centered correlation
# order each participant's three dimensions by their OWN learning accuracy
rank = np.argsort(
    np.argsort(np.where(np.isnan(dacc), -np.inf, dacc), 1), 1
)  # 0 = worst known
lad = np.full((nS, 3), np.nan)
for i in range(nS):
    for j in range(3):
        lad[i, rank[i, j]] = yC[i, j]
xs = np.arange(3)
for i in range(nS):  # one line per participant
    axB.plot(xs, lad[i], color="0.65", lw=0.45, alpha=0.22, zorder=1)
m = np.nanmean(lad, 0)
ci = 1.96 * np.nanstd(lad, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(lad), 0))
axB.vlines(xs, m - ci, m + ci, color="k", lw=3.4, zorder=3)
axB.plot(xs, m, "-o", color="k", lw=2.4, ms=8.0, zorder=4)
axB.axhline(0, ls=":", c="0.6", lw=0.8, zorder=0)
axB.set(
    xticks=xs,
    xticklabels=["worst", "middle", "best"],
    xlabel="participant's own\ndimension knowledge",
    ylabel=r"weight  $|w_{sd}|$" + "\n(double-centered)",
)
axB.set_xlim(-0.35, 2.35)
print(
    f"B  ladder means (worst/middle/best) = {m.round(3)}   paired best-worst = "
    f"{np.nanmean(lad[:, 2] - lad[:, 0]):+.3f}"
)

for ax, lab in [(axB, "A"), (axD, "B")]:
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_TICK)
    ax.yaxis.label.set_size(FS_TICK)
    ax.yaxis.set_label_coords(-0.22, 0.5)
    panel_label(ax, lab, fs=FS_PANEL, x=-0.32, y=1.15)
sns.despine(fig)
fig.savefig(OUT, dpi=350, bbox_inches="tight")
shutil.copy(OUT, ROOT / "manuscript" / "images" / "fig_supp_knowledge.png")
print("saved ->", OUT, "(+ manuscript/images)")
h = SP["K"]["high_minus_low"]
print(
    f"B  tally lift high-low K tertile = {h['mean']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}]"
)
print(
    f"A  corr(dacc, |w|) double-centered    = {rB.statistic:+.3f} (p={rB.pvalue:.1e})"
    f"   n = {xC.size} participant x dimension cells"
)
