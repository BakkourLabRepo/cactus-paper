"""fig3_poles.py — Fig 3: model-free behavioral signatures of the choice model's ingredients,
on a shared P(correct) axis (chance 0.5).

  A  value sensitivity: P(chose the higher-magnitude option) ~ |DeltaValue|, median-split by
     value knowledge K. Observed logistic fit + binned points (subject-clustered SE); shaded band =
     posterior predictive from the model of record.
  B  magnitude reliance: paired raincloud of per-participant accuracy on congruent vs incongruent
     trials, rain colored by K (Blues). Beside the incongruent cloud, the incongruent example card
     drawn twice: option A (magnitude winner) outlined orange, option B (tally winner) purple.
  C  (off by default; PANEL_C = True) per-attribute weights: on divergent trials, P(followed the
     magnitude) by the knowledge rank of the attribute carrying the magnitude win, with a PPC band
     per column. On a divergent trial exactly one attribute sides with the magnitude winner, so
     the columns are disjoint.

Also writes fig3_poles_A_cong.png: panel A split by K x trial type (congruent solid, incongruent
dashed), and prints the panel numbers and the panel-C PPC comparison.

Run: pixi run python scripts/exp2/figures/fig3_poles.py  (any cwd)
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyBboxPatch
from scipy import stats
from scipy.stats import gaussian_kde

from cactus_paper.constants import DIMS, POS
from cactus_paper.design import (
    build_objective_channels,
    build_objective_per_dim,
    load_design,
    taught_levels,
    within_band_position,
)
from cactus_paper.paths import MOR_IDATA
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.plotting import binned, fit_curve, panel_label, style_ms

OUT = figpath("fig3_poles.png")

# ── data ──────────────────────────────────────────────────────────────────────────
D = load_design()
nS, sc = D["nS"], D["sc"]
Y = np.asarray(D["Y"]).astype(float)
K = np.asarray(D["K"])
cd = D["cd"]
# objective taught point values throughout
M, Sg = build_objective_per_dim(D)  # (N,3) per-attribute magnitude / sign, scaled
Vc, _, _ = build_objective_channels(D)  # summed objective magnitude, z within subject
Vr = np.asarray(D["Vr"])  # objective tally, unchanged

# panel A: fold the psychometric onto accuracy. correct = chose the magnitude winner;
# evidence strength = |value difference|. Drop exact ties (no magnitude evidence -> "correct" undefined).
corr = (((Y == 1) & (Vc > 0)) | ((Y == 0) & (Vc < 0))).astype(float)
absVc = np.abs(Vc)
nz = Vc != 0
hiK_t = (K > np.median(K))[sc]
gA = {"hi": hiK_t & nz, "lo": ~hiK_t & nz}
gridA = np.linspace(0, np.quantile(absVc, 0.99), 100)
# panel A, congruency split (standalone variant -> fig3_poles_A_cong.png): K x trial type. Each
# curve gets its own grid (incongruent trials sit at small |value difference| by construction).
div = np.asarray(D["div"])
gA4 = {
    f"{k}_{t}": gA[k] & (div if t == "inc" else ~div)
    for k in ("hi", "lo")
    for t in ("cong", "inc")
}
gridA4 = {g: np.linspace(0, np.quantile(absVc[m], 0.99), 100) for g, m in gA4.items()}

# panel B: per-subject accuracy, congruent vs incongruent (paired)
XORD = ["Congruent", "Incongruent"]
cdc = cd.assign(
    cong=cd.trial_type.map({"non-divergent": "Congruent", "divergent": "Incongruent"})
)
acc = cdc.groupby(["run_id", "cong"])["correct"].mean().unstack()[XORD].dropna()
cong, inc = acc["Congruent"].values, acc["Incongruent"].values
n_cong, n_inc = (int((cdc.cong == c).sum()) for c in XORD)

# panel C: on divergent trials, P(follow the summed magnitude) split by the learning-task
# knowledge rank of the attribute carrying the magnitude win (exactly one per divergent trial)
dacc = D["dacc"]


def _side(sd, d):
    out = np.full(len(cd), np.nan)
    for p in POS:
        dm = cd[f"choice_trial_{sd}_{p}_dim"].values
        vl = cd[f"choice_trial_{sd}_{p}_value"].values
        m = dm == d
        out[m] = vl[m]
    return out


VDo = np.stack(
    [_side("right", d) - _side("left", d) for d in DIMS], 1
)  # objective, unscaled
agg = np.sign(VDo.sum(1))  # summed-magnitude winner (right > 0)
follow_mag = (((Y == 1) & (agg > 0)) | ((Y == 0) & (agg < 0))).astype(float)
crit = np.argmax(np.abs(VDo), 1)  # the magnitude-carrying attribute
krank = np.argsort(np.argsort(-np.where(np.isnan(dacc), -np.inf, dacc), 1), 1)
critrank = krank[sc, crit]
assert (
    ((np.sign(VDo) == agg[:, None]) & (np.sign(VDo) != 0))[div].sum(1) == 1
).all(), (
    "divergent trials must have exactly one attribute siding with the magnitude winner"
)


def follow_by_rank(r: int, min_tr: int = 5) -> Tuple[np.ndarray, int]:
    """Per-subject P(follow magnitude) on divergent trials whose critical attribute has rank r."""
    a = np.full(nS, np.nan)
    pool = div & (critrank == r)
    for s in range(nS):
        m = pool & (sc == s)
        if m.sum() >= min_tr:
            a[s] = follow_mag[m].mean()
    return a, int(pool.sum())


accR, nR = zip(
    *[follow_by_rank(r) for r in (2, 1, 0)]
)  # worst, middle, best (left -> right)
# each participant's overall P(follow magnitude) on divergent trials (panel C rain color). In
# load_design subject order; panel B's `inc` is in sorted run_id order, so never cross them
# without reindexing.
overallC = np.array([follow_mag[div & (sc == s_)].mean() for s_ in range(nS)])
okC = ~np.isnan(np.column_stack(accR)).any(1)
accR = [a[okC] for a in accR]
overallC = overallC[okC]

# ── posterior-predictive bands from the fit of record (panels A and C, one pass) ─────
POST = az.from_netcdf(MOR_IDATA).posterior
_flat = lambda n: POST[n].values.reshape(-1, *POST[n].values.shape[2:])
draws = {
    n: _flat(n)
    for n in ["lam_s", "W", "a", "a0", "sda", "psi_mu", "sigma_psi", "z_psi", "s_band"]
}

# the predictive rebuilds u_sd on every draw (per-participant value function from
# psi_mu/sigma_psi/z_psi via softmax, plus the within-band map slope). The tally channel is the
# RAW sign (+/-1), not the standardized `Sg`.
_KL, _KR = taught_levels(cd)  # (N,3) taught level index, left / right
_PL, _PR = within_band_position(cd, sc)  # (N,3) value-referenced within-band position
_SGN = np.sign(_KR - _KL).astype(float)  # (N,3) raw tally channel


def _u_parts(d):
    """One posterior draw index -> (u_diff including the within-band map slope, taught-only)."""
    psi = draws["psi_mu"][d] + draws["sigma_psi"][d] * draws["z_psi"][d]  # (nS,3,3)
    psi = np.concatenate([np.zeros((psi.shape[0], 3, 1)), psi], -1)
    frac = np.exp(psi) / np.exp(psi).sum(-1, keepdims=True)
    u = np.concatenate(
        [np.zeros((frac.shape[0], 3, 1)), np.cumsum(frac, -1)], -1
    )  # (nS,3,5)
    du_t = np.stack(
        [u[sc, dd, _KR[:, dd]] - u[sc, dd, _KL[:, dd]] for dd in range(3)], 1
    )
    return du_t + draws["s_band"][d] * (_PR - _PL), du_t


# Panel C's pools are precomputed as boolean masks so the simulation loop stays cheap.
_poolC = [[div & (critrank == r) & (sc == s) for s in range(nS)] for r in (2, 1, 0)]
_useC = [[m for m in row if m.sum() >= 5] for row in _poolC]


def ppc_bands(
    nsim: int = 300, seed: int = 2026
) -> Tuple[Dict[Any, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Simulate choices from the model-of-record posterior; refit panel A's smoother and panel C's column
    means on each simulated dataset. One pass, so A and C share the same draws."""
    lamD, WD, aD = draws["lam_s"], draws["W"], draws["a"]
    a0D, sdaD = draws["a0"], draws["sda"]
    N = len(Y)
    rng = np.random.default_rng(seed)
    sel = rng.choice(a0D.shape[0], min(nsim, a0D.shape[0]), replace=False)
    masks = {**gA, **gA4}
    grids = {**{g: gridA for g in gA}, **gridA4}
    pp = {g: [] for g in masks}
    ppC = np.zeros((len(sel), 3))
    for q, d in enumerate(sel):
        Wt, lam = WD[d][sc], lamD[d][sc]
        du, _ = _u_parts(d)
        eta = (
            a0D[d]
            + aD[d][sc] * sdaD[d]
            + (1 - lam) * (Wt * _SGN).sum(-1)
            + lam * (Wt * du).sum(-1)
        )
        ys = (rng.random(N) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        cm = (((ys == 1) & (Vc > 0)) | ((ys == 0) & (Vc < 0))).astype(float)
        for g, m in masks.items():
            pp[g].append(fit_curve(absVc[m], cm[m], grids[g]))
        fm = (((ys == 1) & (agg > 0)) | ((ys == 0) & (agg < 0))).astype(float)
        ppC[q] = [np.mean([fm[m].mean() for m in row]) for row in _useC]
    band = lambda L: (np.percentile(L, 2.5, 0), np.percentile(L, 97.5, 0))
    return {g: band(pp[g]) for g in pp}, ppC


ppcA, ppcC = ppc_bands()

# ── style + colors ─────────────────────────────────────────────────────────────────
style_ms()
PAL = {
    "hi": "#08306b",
    "lo": "#6baed6",
}  # high / low K: dark / light blue
CAT, REL = "#b35806", "#542788"  # magnitude (orange) <-> tally (purple)
# ordered small-category ramp for panel C's knowledge-rank ladder (worst -> best known)
RANKC = [tuple(sns.color_palette("Greys", 6)[i]) for i in (1, 3, 5)]
CMAP = LinearSegmentedColormap.from_list("pole", [REL, "#f2f2f2", CAT])
NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
FS_NOTE, FS_LEGEND, FS_PANEL, FS_COND = 11, 12, 22, 14
rng = np.random.default_rng(0)

# canvas width sets printed text size (text block ~6.1in); changing figsize rescales every glyph.
# PANEL_C = True draws panel C; its numbers print either way.
PANEL_C = False
if PANEL_C:
    fig, (axA, axB, axC) = plt.subplots(
        1,
        3,
        figsize=(11.4, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]},
    )
else:
    fig, (axA, axB) = plt.subplots(
        1,
        2,
        figsize=(11.4 * 2.2 / 3.4, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.4]},
    )


def raincloud(
    ax: Axes,
    L: np.ndarray,
    R: np.ndarray,
    names: Sequence[str],
    notes: Sequence[str],
    cloudcol: Tuple[str, str] = ("0.85", "0.85"),
    pointcol: Optional[Tuple[Any, Any]] = None,
    cmap: Any = CMAP,
    norm: Any = NORM,
) -> None:
    """Paired raincloud: cloud (outer) + rain (inner) + pairing lines + mean±95%CI, on [0,1] y.
    pointcol=(cL, cR) flat colors, or ('cmap', values) on either side to gradient-color that rain
    with cmap/norm (default: panel B's tally<->magnitude scale).
    """
    yy = np.linspace(0, 1, 200)
    kde = lambda v: (lambda d: d / d.max() * 0.34)(gaussian_kde(v)(yy))
    ax.fill_betweenx(
        yy, -0.10 - kde(L), -0.10, color=cloudcol[0], lw=0, zorder=0
    )  # left cloud
    ax.fill_betweenx(
        yy, 1.10, 1.10 + kde(R), color=cloudcol[1], lw=0, zorder=0
    )  # right cloud
    lx = 0.06 + rng.uniform(-0.03, 0.03, len(L))
    rx = 0.94 + rng.uniform(-0.03, 0.03, len(R))
    for a, b, ya, yb in zip(lx, rx, L, R):
        ax.plot([a, b], [ya, yb], color="0.6", alpha=0.12, lw=0.8, zorder=1)
    cL, cR = pointcol
    if isinstance(cL, tuple):  # gradient-colored left rain (panel B by K)
        ax.scatter(
            lx,
            L,
            s=26,
            c=cL[1],
            cmap=cmap,
            norm=norm,
            edgecolor="0.35",
            lw=0.4,
            zorder=2,
        )
    else:
        ax.scatter(lx, L, s=26, color=cL, edgecolor="none", alpha=0.6, zorder=2)
    if isinstance(cR, tuple):  # gradient-colored right rain (panel B)
        ax.scatter(
            rx,
            R,
            s=36,
            c=cR[1],
            cmap=cmap,
            norm=norm,
            edgecolor="0.35",
            lw=0.4,
            zorder=3,
        )
    else:
        ax.scatter(rx, R, s=36, color=cR, edgecolor="0.35", lw=0.4, alpha=0.8, zorder=3)
    for x0, v in [(0.06, L), (0.94, R)]:
        ax.errorbar(
            x0,
            v.mean(),
            yerr=stats.sem(v) * 1.96,
            fmt="o",
            color="k",
            ms=7.5,
            mec="white",
            mew=1.0,
            capsize=0,
            elinewidth=2.0,
            zorder=6,
        )
    ax.plot([0.06, 0.94], [L.mean(), R.mean()], color="k", lw=2.5, zorder=5)
    ax.axhline(0.5, ls=":", c="0.6", lw=0.8)
    # right margin sized for panel B's color key, drawn just outside the right cloud
    ax.set(xlim=(-0.48, 2.20))
    ax.set_xticks([])
    tr = ax.get_xaxis_transform()
    for x0, name, note in [(-0.05, names[0], notes[0]), (1.05, names[1], notes[1])]:
        ax.text(x0, -0.05, name, transform=tr, ha="center", va="top", fontsize=FS_COND)
        # trial-count note below the condition name
        ax.text(
            x0,
            -0.19,
            note,
            transform=tr,
            ha="center",
            va="top",
            fontsize=FS_NOTE,
            color="0.35",
        )


def raincloud_k(
    ax: Axes,
    cols: Sequence[np.ndarray],
    names: Sequence[str],
    notes: Sequence[str],
    pointcols: Sequence[Any],
    ppc: Optional[np.ndarray] = None,
) -> None:
    """k-column raincloud: half-violin left of each column + rain + pairing lines + mean±95%CI.
    Generalises `raincloud` for ordered factors with more than two levels (panel C)."""
    yy = np.linspace(0, 1, 200)
    kde = lambda v: (lambda d: d / d.max() * 0.13)(gaussian_kde(v)(yy))
    xs = np.linspace(0.10, 0.92, len(cols))
    jx = [x + rng.uniform(-0.025, 0.025, len(v)) for x, v in zip(xs, cols)]
    for a, b in zip(range(len(cols) - 1), range(1, len(cols))):
        for x1, x2, y1, y2 in zip(jx[a], jx[b], cols[a], cols[b]):
            ax.plot([x1, x2], [y1, y2], color="0.6", alpha=0.10, lw=0.8, zorder=1)
    if ppc is not None:  # 95% PPC interval per column
        for x, col in zip(xs, ppc.T):
            lo, hi = np.percentile(col, [2.5, 97.5])
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.075, lo),
                    0.15,
                    hi - lo,
                    facecolor="0.45",
                    alpha=0.30,
                    lw=0,
                    zorder=4,
                )
            )
    for x, v, c, j in zip(xs, cols, pointcols, jx):
        ax.fill_betweenx(
            yy, x - 0.055 - kde(v), x - 0.055, color="0.88", lw=0, zorder=0
        )
        if (
            isinstance(c, tuple) and c and c[0] == "cmap"
        ):  # gradient rain (panel B's scale)
            # c[1] is a per-participant value, not the column's y
            ax.scatter(
                j,
                v,
                s=32,
                c=c[1],
                cmap=CMAP,
                norm=NORM,
                edgecolor="0.35",
                lw=0.4,
                zorder=3,
            )
        else:
            ax.scatter(
                j, v, s=32, color=c, edgecolor="0.35", lw=0.4, alpha=0.8, zorder=3
            )
    ax.plot(xs, [v.mean() for v in cols], color="k", lw=2.5, zorder=5)
    for x, v in zip(xs, cols):
        ax.errorbar(
            x,
            v.mean(),
            yerr=stats.sem(v) * 1.96,
            fmt="o",
            color="k",
            ms=7.5,
            mec="white",
            mew=1.0,
            capsize=0,
            elinewidth=2.0,
            zorder=6,
        )
    ax.axhline(0.5, ls=":", c="0.6", lw=0.8)
    # right margin so the last column's centered "N trials" note is not clipped
    ax.set(xlim=(-0.22, 1.14))
    ax.set_xticks([])
    tr = ax.get_xaxis_transform()
    for x, name, note in zip(xs, names, notes):
        ax.text(x, -0.05, name, transform=tr, ha="center", va="top", fontsize=FS_COND)
        # trial-count note below the column name, as in raincloud()
        ax.text(
            x,
            -0.19,
            note,
            transform=tr,
            ha="center",
            va="top",
            fontsize=FS_NOTE - 1.5,
            color="0.35",
        )


# ── A: knowledge sharpens the accuracy psychometric (value sensitivity) ──────────────
for key, lab in [("hi", "high K"), ("lo", "low K")]:
    m, col = gA[key], PAL[key]
    lo, hi = ppcA[key]
    axA.fill_between(gridA, lo, hi, color=col, alpha=0.20, lw=0)
    axA.plot(gridA, fit_curve(absVc[m], corr[m], gridA), color=col, lw=2, label=lab)
    mx, my, se = binned(absVc[m], corr[m], q=7, groups=sc[m])
    axA.errorbar(
        mx, my, yerr=se, fmt="o", ms=5, color=col, capsize=0, elinewidth=2, ls="none"
    )
axA.axhline(0.5, ls=":", c="0.6", lw=0.8)
axA.set(
    xlabel=r"$|\Delta\,\mathrm{Value}|$",
    ylabel="P(correct)",
    ylim=(0, 1.02),
    xlim=(0, gridA.max()),
)
# lower right is the empty corner
axA.legend(frameon=False, loc="lower right", fontsize=FS_LEGEND)

# ── B: congruent vs incongruent, colored by value knowledge ──────────────────────────────
# rain colored by K (Blues). Example cards: option A (magnitude winner) outlined orange at top,
# option B (tally winner) purple at bottom.
# K is in load_design order; `acc` is in sorted run_id order -- reindex, never index positionally.
K_B = dict(zip(cd.run_id.unique(), K))
K_B = np.array([K_B[r] for r in acc.index])
KCMAP = LinearSegmentedColormap.from_list(
    "Kblues", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256))
)
KNORM = Normalize(*np.quantile(K_B, [0.05, 0.95]), clip=True)
raincloud(
    axB,
    cong,
    inc,
    ("Congruent", "Incongruent"),
    (f"{n_cong:,}\ntrials", f"{n_inc:,}\ntrials"),
    pointcol=(("cmap", K_B), ("cmap", K_B)),
    cmap=KCMAP,
    norm=KNORM,
)
axB.set_xlim(-0.48, 2.30)

INCONG = {
    "A": [100, 25, 25],
    "B": [25, 50, 50],
}  # fig2_task's worked example (color, shape, line)


def card_w(ax: Axes, h: float) -> float:
    """Data-x width of a card of data-y height h, at the card's 2.2 x 3.0 aspect. Call after
    tight_layout: it reads the axes' final size."""
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    y_in = h / np.diff(ax.get_ylim())[0] * pos.height * fh
    return y_in * 2.2 / 3.0 / (pos.width * fw) * np.diff(ax.get_xlim())[0]


def example_card(
    ax: Axes, x0: float, y0: float, h: float, pick: str, col: Any
) -> float:
    """The incongruent example card at data (x0, y0), height h, with option `pick` ("A" or "B")
    outlined in `col`. Drawn in its own equal-aspect inset so the circles stay round inside a
    stretched P(correct) axis."""
    w = card_w(ax, h)
    ia = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    ia.set(xlim=(0, 2.2), ylim=(0, 3.0), aspect="equal")
    ia.axis("off")
    ia.add_patch(
        FancyBboxPatch(
            (0.08, 0.08),
            2.04,
            2.62,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc="#EBEBEB",
            ec="#AAAAAA",
            lw=1.0,
            zorder=0,
        )
    )
    cx, ry = (0.6, 1.6), (2.25, 1.39, 0.53)
    for oi, opt in enumerate("AB"):
        ia.text(
            cx[oi],
            2.86,
            opt,
            ha="center",
            va="center",
            fontsize=CARD_FS[0],
            fontweight="bold",
            color=col if opt == pick else "0.45",
        )
        for di, y in enumerate(ry):
            ia.add_patch(
                Circle((cx[oi], y), 0.36, fc="#E0E0E0", ec="#999999", lw=1.0, zorder=2)
            )
            ia.text(
                cx[oi],
                y,
                str(INCONG[opt][di]),
                ha="center",
                va="center",
                fontsize=CARD_FS[1],
                fontweight="bold",
                color="k",
                zorder=4,
            )
    xp = cx["AB".index(pick)]
    ia.add_patch(
        FancyBboxPatch(
            (xp - 0.45, 0.13),
            0.90,
            2.52,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            fc="none",
            ec=col,
            lw=2.2,
            zorder=5,
        )
    )
    return w


# cards + vertical K key, drawn after tight_layout (card_w reads the axes' final size). The card
# pair overhangs the [0, 1] axis at both ends; the bottom spine stops short of them.
CARD_X, CARD_H, CARD_GAP = 1.72, 0.60, 0.04  # data coords
CARD_FS = (FS_NOTE + 0.5, FS_NOTE - 2)  # option letters, values
CB_X, CB_W, CB_Y = 1.535, 0.055, (0.14, 0.74)


def draw_cards_and_key() -> None:
    y_top = 0.5 + CARD_GAP / 2
    for y0, pick, col in [(y_top, "A", CAT), (1.0 - y_top - CARD_H, "B", REL)]:
        example_card(axB, CARD_X, y0, CARD_H, pick, col)
    cax = axB.inset_axes(
        [CB_X, CB_Y[0], CB_W, CB_Y[1] - CB_Y[0]], transform=axB.transData
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=KNORM, cmap=KCMAP),
        cax=cax,
        orientation="vertical",
        ticks=[],
    )
    cb.outline.set_visible(False)
    xc = CB_X + CB_W / 2
    for y, va, lab in [
        (CB_Y[1] + 0.02, "bottom", "high"),
        (CB_Y[0] - 0.02, "top", "low"),
    ]:
        axB.text(xc, y, lab, ha="center", va=va, fontsize=FS_NOTE)
    axB.text(
        xc,
        CB_Y[1] + 0.12,
        "K",
        ha="center",
        va="bottom",
        fontsize=FS_NOTE + 1,
        fontweight="bold",
    )
    axB.spines["bottom"].set_bounds(axB.get_xlim()[0], CARD_X - 0.06)


# ── C: dissent by the attribute's knowledge rank (per-attribute weights) ────────────
if PANEL_C:
    raincloud_k(
        axC,
        accR,
        ("worst", "middle", "best"),
        tuple(f"{n:,}\ntrials" for n in nR),
        [("cmap", overallC)] * 3,
        ppc=ppcC,
    )
# on divergent trials P(correct) = P(followed the magnitude), so C shares the axis

TITLES = [(axA, "Value sensitivity", "A"), (axB, "Magnitude reliance", "B")]
if PANEL_C:
    TITLES.append((axC, "Per-attribute weights", "C"))
for ax, title, L in TITLES:
    # y pinned: auto-placement would lift B's title over the overhanging top card
    ax.set_title(title, fontsize=FS_LEGEND, fontweight="bold", pad=8, y=1.0)
    panel_label(ax, L, fs=FS_PANEL)
sns.despine(fig)
fig.tight_layout(w_pad=0.4)
draw_cards_and_key()
fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved ->", OUT)

# ── A, congruency split (standalone): K by color as in A, trial type by line style ───
# Solid line + filled points = congruent, dashed + open = incongruent. Same axis and size as
# panel A so it can be swapped in.
figA4, ax4 = plt.subplots(figsize=(11.4 * 2.2 / 3.4 / 2.2, 3.6))
for k, klab in [("hi", "high K"), ("lo", "low K")]:
    for t, tlab, ls, mfc in [
        ("cong", "congruent", "-", PAL[k]),
        ("inc", "incongruent", "--", "white"),
    ]:
        g = f"{k}_{t}"
        m, col, gr = gA4[g], PAL[k], gridA4[g]
        lo, hi = ppcA[g]
        ax4.fill_between(gr, lo, hi, color=col, alpha=0.20, lw=0)
        ax4.plot(
            gr,
            fit_curve(absVc[m], corr[m], gr),
            color=col,
            lw=2,
            ls=ls,
            label=f"{klab}, {tlab}",
        )
        mx, my, se = binned(absVc[m], corr[m], q=7, groups=sc[m])
        ax4.errorbar(
            mx,
            my,
            yerr=se,
            fmt="o",
            ms=5,
            color=col,
            mfc=mfc,
            mew=1.5,
            capsize=0,
            elinewidth=2,
            ls="none",
            zorder=3,
        )
ax4.axhline(0.5, ls=":", c="0.6", lw=0.8)
ax4.set(
    xlabel=r"$|\Delta\,\mathrm{Value}|$",
    ylabel="P(correct)",
    ylim=(0, 1.02),
    xlim=(0, gridA.max()),
)
ax4.legend(frameon=False, loc="lower right", fontsize=FS_NOTE)
ax4.set_title("Value sensitivity", fontsize=FS_LEGEND, fontweight="bold", pad=8)
panel_label(ax4, "A", fs=FS_PANEL)
sns.despine(figA4)
OUT_A4 = figpath("fig3_poles_A_cong.png")
figA4.savefig(OUT_A4, dpi=600, bbox_inches="tight")
print("saved ->", OUT_A4)
for g, m in gA4.items():
    print(f"  A {g:8s} n={m.sum():5d}  P(correct) {corr[m].mean():.3f}")

for nm, v in [("congruent", cong), ("incongruent", inc)]:
    r = stats.pearsonr(K_B, v)
    print(f"  B K x {nm:11s} r = {r.statistic:+.3f}  p = {r.pvalue:.4f}  (n={len(v)})")
print(
    f"B congruent  {cong.mean():.3f} (SD {cong.std():.3f}) | incongruent {inc.mean():.3f} (SD {inc.std():.3f})"
)
w_, m_, b_ = accR
print(f"C divergent trials, by knowledge rank of the magnitude-carrying attribute:")
print(
    f"    worst {w_.mean():.3f} | middle {m_.mean():.3f} | best {b_.mean():.3f}   "
    f"(n={okC.sum()} participants; columns disjoint, {sum(nR):,} = all divergent trials)"
)
A = np.column_stack([b_, m_, w_])
tl = stats.ttest_1samp(A @ np.array([1.0, 0.0, -1.0]), 0)  # linear: best - worst
print(
    f"  linear trend (best - worst)   t({len(A) - 1})={tl.statistic:+.2f}  p={tl.pvalue:.4f}"
)
print(
    "  posterior predictive from the fit of record (NO trial-level parameter anywhere):"
)
for nm, col, obs in zip(
    ("worst", "mid  ", "best "), ppcC.T, (w_.mean(), m_.mean(), b_.mean())
):
    lo, hi = np.percentile(col, [2.5, 97.5])
    print(f"    {nm}: PPC {col.mean():.3f} [{lo:.3f}, {hi:.3f}]   observed {obs:.3f}")
sl = ppcC[:, 2] - ppcC[:, 0]
lo, hi = np.percentile(sl, [2.5, 97.5])
print(
    f"    best-worst slope: observed {b_.mean() - w_.mean():+.3f}   "
    f"PPC {sl.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
)
print(
    "    -> observed inside the band: the TRAIT weight vector already generates this "
    "trial-level-looking effect."
)
