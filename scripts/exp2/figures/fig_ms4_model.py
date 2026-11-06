"""fig_ms4_model.py — supplement figure: the strategy plane (fig_supp_strategy_plane.png).

One panel: magnitude reliance lam_s vs weight concentration (Gini |w|), one point per participant
(posterior means from the model of record), with classical strategies (TALLY, TTB, WADD) at the
limits. Prints the plane's summary numbers (correlations, min/max weight ratio, one-cue count).

Run: pixi run python scripts/exp2/figures/fig_ms4_model.py  (any cwd)
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from cactus_paper.design import load_design, pole_score
from cactus_paper.features.traits import concentration as wl_conc
from cactus_paper.features.traits import minmax_ratio
from cactus_paper.paths import MOR_IDATA
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import boot_line, panel_label, style_ms

OUT = figpath("fig_supp_strategy_plane.png")

D = load_design()
nS, K = D["nS"], np.asarray(D["K"])
su = D["cd"].run_id.drop_duplicates().tolist()

po = az.from_netcdf(MOR_IDATA).posterior
W = po["W"].values.reshape(-1, nS, 3).mean(0)
lam = po["lam_s"].values.reshape(-1, nS).mean(0)

absW = np.abs(W)
# gain = mean|W_s| on the raw scale (the model's `wgain` deterministic)
gain = absW.mean(1)
prof = absW / absW.sum(1, keepdims=True)  # importance profile on the simplex
conc = wl_conc(absW)  # GINI of |w| (see cactus_paper.features.traits)

style_ms()
FS_NOTE, FS_PANEL, FS_CORNER = 12.5, 24, 12.5

# canvas width sets printed text size: the text block is ~6.1in and bbox_inches="tight" trims the
# save, so measure the PNG. Changing figsize rescales every glyph.
fig, axP = plt.subplots(figsize=(6.40, 4.10))

# the plane; no fit line. Dashed line = one-cue ceiling of the Gini, (D-1)/D = 2/3 for D = 3
axP.scatter(lam, conc, s=22, c="#333333", alpha=0.45, edgecolor="none")
axP.set(
    xlabel=r"magnitude reliance  $\lambda_s$", ylabel=r"concentration  (Gini $|w|$)"
)
# margin past the bounds holds the corner labels; spines are trimmed to the real ranges
axP.set_xlim(-0.015, 1.135)
axP.set_ylim(-0.055, 0.70)
axP.spines["bottom"].set_bounds(0, 1)
axP.spines["left"].set_bounds(0, 0.6)
axP.plot([0, 1], [2 / 3] * 2, color="0.55", ls=(0, (4, 3)), lw=1.1, zorder=1)
# classical strategies as the plane's limit points: TALLY and TTB at lam = 0, WADD at lam = 1
import matplotlib.patheffects as pe

_halo = [pe.withStroke(linewidth=3.2, foreground="white")]
_corner = dict(
    fontsize=FS_CORNER,
    style="italic",
    color="0.35",
    zorder=4,
    path_effects=_halo,
    transform=axP.transData,
)
axP.text(0.0, 0.0, "TALLY", ha="left", va="bottom", **_corner)
axP.text(0.0, 2 / 3, "TTB", ha="left", va="top", **_corner)
axP.text(1.13, -0.052, "WADD", ha="right", va="bottom", **_corner)
axP.tick_params(labelsize=12.5)
sns.despine(fig)
fig.savefig(OUT, dpi=350, bbox_inches="tight")
print("saved ->", OUT)

# ---------------------------------------------------------------- console readout
print(
    f"\nstrategy plane  corr(lam, concentration) = "
    f"{stats.pearsonr(lam, conc).statistic:+.3f} "
    f"(p={stats.pearsonr(lam, conc).pvalue:.2f})   -- independent axes"
)
print(
    f"  lam   mean {lam.mean():.3f}  sd {lam.std():.3f}  range {lam.min():.2f}-{lam.max():.2f}"
)
print(
    f"  conc  mean {conc.mean():.3f}  sd {conc.std():.3f}  range {conc.min():.2f}-{conc.max():.2f}"
)
ratio = minmax_ratio(absW)
print(
    f"  min/max weight ratio: median {np.median(ratio):.3f}  "
    f"(1 = equal weights, 0 = a single cue)"
)
print(
    f"  effectively one-cue (2nd weight < 1/3 of top): {int((np.sort(absW, 1)[:, ::-1][:, 1] < np.sort(absW, 1)[:, ::-1][:, 0] / 3).sum())} of {nS}"
)
hi = (lam > np.median(lam)) & (conc > np.median(conc))
srt = np.sort(absW[hi], 1)[:, ::-1].mean(0)
print(
    f"  high-lam/high-conc region (n={int(hi.sum())}): mean sorted |w| = {srt.round(2)}"
    f"  -> top cue carries {100 * srt[0] / srt.sum():.0f}% of the weight, NOT all of it"
)
print(
    "  => classical strategies are LIMITS of this space; the sample sits in the interior."
    ""
)
print(
    f"\ncorr(lam, gain |W|) = {stats.pearsonr(lam, gain).statistic:+.3f}  "
    f"(reported in the model section)"
)
print(
    f"corr(gain |W|, K)   = {stats.pearsonr(K, gain).statistic:+.3f}  "
    f"(fig_ms_knowledge panel A)"
)
