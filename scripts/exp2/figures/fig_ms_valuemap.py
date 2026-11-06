"""fig_ms_valuemap.py — the value map (main-text figure under the M3lam model of record).

Draws the map as one function per attribute, in u-space.

  ROW 1 (A, B)  schematic hypotheses, synthetic curves with no data: a five-label staircase (flat
     plateaus, a jump at each band edge) and a fully graded map (the diagonal L(a) = (a-16)/200).
  ROW 2 (C, D, E)  the fitted map for color, line, shape (group mean + 95% HDI band, participant
     lines behind). Nothing is drawn across the between-band gaps (no stimulus fell there); a
     full-height dashed line marks each gap's midpoint, in both rows.


Inputs: MOR_IDATA, results/fits/mor_fit.json, results/fits/counterbalance.json.

Run: pixi run python scripts/exp2/figures/fig_ms_valuemap.py
"""

import json
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cactus_paper.constants import BAND_CENTERS, BAND_WIDTH, BANDS, DIMS
from cactus_paper.paths import MOR_IDATA, ROOT
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import DIM_COLORS, panel_label, style_ms

LINE, DOT = style_ms()
from cactus_paper.paths import FITS  # noqa: E402

# display order (as fig_ms1); posterior arrays are indexed in design.DIMS order, so use DIMS.index(d)
ORDER = ["color", "line", "shape"]
GRP = "black"  # group summary, against the attribute-colored participant lines

# model of record: group increments = softmax(psi_mu); per-participant increments from
# psi_mu + sigma_psi * z_psi
po = az.from_netcdf(MOR_IDATA).posterior
_psi = po["psi_mu"].values.reshape(-1, 3, 3)
_psi_full = np.concatenate([np.zeros((_psi.shape[0], 3, 1)), _psi], axis=-1)
FR = np.exp(_psi_full) / np.exp(_psi_full).sum(
    -1, keepdims=True
)  # (draws, dim, 4) group level
# the model of record exposes psi_mu/sigma_psi/z_psi, not a frac_s deterministic
_pm = po["psi_mu"].values.reshape(-1, 3, 3)
_sp = po["sigma_psi"].values.reshape(-1, 3, 3)
_zp = po["z_psi"].values.reshape(-1, po["z_psi"].shape[-3], 3, 3)
_psi_s = _pm[:, None, :, :] + _sp[:, None, :, :] * _zp  # (draws, nS, 3, 3)
_psi_s = np.concatenate([np.zeros(_psi_s.shape[:-1] + (1,)), _psi_s], -1)
FR_S = (np.exp(_psi_s) / np.exp(_psi_s).sum(-1, keepdims=True)).mean(0)  # (nS, dim, 4)
SB = json.load(open(FITS / "mor_fit.json"))["mor"]
SB["_full_ramp_benchmark"] = json.load(open(FITS / "mor_fit.json"))[
    "_full_ramp_benchmark"
]


def hdi2(a: Any) -> np.ndarray:
    return az.hdi(np.asarray(a), hdi_prob=0.95).T


# ---- schematic reference curves (row 1) -- synthetic, no data ----------------------------
CENTERS = np.array(BAND_CENTERS)
SPAN = CENTERS[-1] - CENTERS[0]  # 200 deg between the extreme band centers
NPTS = 25


def _ref(a):
    """Fully graded, unwarped map: linear in position, u = k/4 at each band center."""
    return (a - CENTERS[0]) / SPAN


def _schematic(hypothesis, k, g):
    return np.full_like(g, k / 4.0) if hypothesis == "staircase" else _ref(g)


# ---- the fitted map, in u-space (knot heights U + within-band slope S) ---------------------
# U = cumsum(FR) with a leading zero (knot heights from knot-to-knot increments)
U = np.concatenate(
    [np.zeros((FR.shape[0], 3, 1)), np.cumsum(FR, -1)], -1
)  # (draws,3,5)
US = np.concatenate(
    [np.zeros(FR_S.shape[:-1] + (1,)), np.cumsum(FR_S, -1)], -1
)  # (nS,3,5)
S = po["s_band"].values.reshape(-1, 3)  # (draws,3)
S_MEAN = S.mean(0)  # the model of record has no per-participant slope


def hdi1(a: Any) -> np.ndarray:
    return az.hdi(np.asarray(a), hdi_prob=0.95)


# ---- MAIN: 2 schematic panels (A, B) over 3 fitted-map panels (C, D, E) --------------------
# canvas width fixed at 9.60in so glyph sizes match the other manuscript figures
fig = plt.figure(figsize=(9.60, 6.35))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15], hspace=0.60, wspace=0.35)
axSchem = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
axFit = [
    fig.add_subplot(gs[1, 0:2]),
    fig.add_subplot(gs[1, 2:4]),
    fig.add_subplot(gs[1, 4:6]),
]

# A, B: the two hypotheses, no data
for si, (ax, hyp, title) in enumerate(
    zip(axSchem, ("staircase", "graded"), ("Five-label staircase", "Fully graded map"))
):
    edges = []
    for k, (lo, hi, _) in enumerate(BANDS):
        g = np.linspace(lo, hi, NPTS)
        y = _schematic(hyp, k, g)
        ax.plot(g, y, color="k", lw=2.4, zorder=3)
        edges.append((lo, hi))
    for bi in range(4):  # vertical dashed line AT the discontinuity, full span
        xm = (edges[bi][1] + edges[bi + 1][0]) / 2
        ax.axvline(xm, color="k", ls=(0, (3, 2)), lw=1.4, zorder=3)
    ax.set(
        xticks=CENTERS,
        xticklabels=["0", "25", "50", "75", "100"],
        xlim=(-8, 240),
        ylim=(-0.06, 1.06),
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
    )
    ax.set_title(title, fontsize=13.5, pad=8)
    ax.tick_params(axis="x", labelsize=11.8)
    if si:
        ax.set_yticklabels([])
    panel_label(ax, "AB"[si], x=-0.13 if si else -0.30, y=1.20)
axSchem[0].set_ylabel("value $u$")
axSchem[0].set_xlabel("position along the value axis", x=1.05, fontsize=12)

# C, D, E: the fitted map per attribute, same u-space axis; no join across the between-band gaps
for k, (d, ax) in enumerate(zip(ORDER, axFit)):
    j = DIMS.index(d)
    edges = []
    for bi, (lo, hi, _) in enumerate(BANDS):
        g = np.linspace(lo, hi, NPTS)
        p = (g - lo) / BAND_WIDTH - 0.5
        u = U[:, j, bi][:, None] + S[:, j][:, None] * p[None, :]
        band = hdi1(u[None])  # (NPTS, 2); [None] adds the chain dim
        for i in range(US.shape[0]):  # per-participant spread, band-local
            ax.plot(
                g,
                US[i, j, bi] + S_MEAN[j] * p,
                color=DIM_COLORS[d],
                alpha=0.13,
                lw=0.6,
                zorder=1.4,
            )
        ax.fill_between(
            g, band[:, 0], band[:, 1], color=GRP, alpha=0.28, lw=0, zorder=2
        )
        ax.plot(g, u.mean(0), color=GRP, lw=2.4, zorder=3)
        edges.append((lo, hi))
    for bi in range(4):
        xm = (edges[bi][1] + edges[bi + 1][0]) / 2
        ax.axvline(xm, color=GRP, ls=(0, (3, 2)), lw=1.4, zorder=3)
    ax.set(
        xticks=CENTERS,
        xticklabels=["0", "25", "50", "75", "100"],
        xlim=(-8, 240),
        ylim=(-0.06, 1.06),
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
    )
    ax.tick_params(axis="x", labelsize=11.8)
    if k:
        ax.set_yticklabels([])
    ax.text(
        0.5,
        1.03,
        d,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14.5,
        color=DIM_COLORS[d],
    )
    panel_label(ax, "CDE"[k], x=-0.13 if k else -0.30, y=1.18)
axFit[0].set_ylabel("value $u$")
axFit[1].set_xlabel("position along the value axis (taught points at band centers)")
sns.despine(fig=fig)
fig.savefig(figpath("fig_ms_valuemap.png"), dpi=350, bbox_inches="tight")
print(f"saved -> {figpath('fig_ms_valuemap.png')}")

# ---- console ------------------------------------------------------------------------------
_ss = np.array([SB[d]["s"] for d in DIMS])
print(
    f"\nwithin-band slope, per-dim mean: {_ss.mean():+.4f} = "
    f"{_ss.mean() / SB['_full_ramp_benchmark']:.2f} x a fully graded map "
    f"(0 = five-label staircase, {SB['_full_ramp_benchmark']} = locally angle-linear)"
)
print("\ngroup increments (posterior mean [95% HDI]), taught = 0.250 each")
for d in ORDER:
    j = DIMS.index(d)
    lo, hi = hdi2(FR[:, j, :])
    print(
        f"  {d:6s} "
        + "  ".join(
            f"{FR[:, j, i].mean():.3f}[{lo[i]:.3f},{hi[i]:.3f}]" for i in range(4)
        )
    )
edge = FR[:, 0, 0] + FR[:, 0, 3] - FR[:, 0, 1] - FR[:, 0, 2]
elo, ehi = az.hdi(edge, hdi_prob=0.95)
print(
    f"\ncolor U-shape (edge - interior) {edge.mean():+.3f} [{elo:+.3f}, {ehi:+.3f}]  "
    f"P(>0)={float((edge > 0).mean()):.3f}"
)
