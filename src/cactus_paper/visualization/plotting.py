"""cactus_paper.plotting — shared figure-script helpers (jax-free, so figure scripts import fast).

`binned`: groups=None gives trial-level SE; groups given gives subject-clustered SE.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes

sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))


def logit_irls(X: np.ndarray, y: np.ndarray, iters: int = 30) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = np.clip(p * (1 - p), 1e-9, None)
        step = np.linalg.solve(
            X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p)
        )
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def fit_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Logistic fit evaluated on grid (the observed-smoother every PPC band re-fits)."""
    return sig(
        np.column_stack([np.ones_like(grid), grid])
        @ logit_irls(np.column_stack([np.ones_like(x), x]), y)
    )


def binned(
    x: np.ndarray, y: np.ndarray, q: int = 7, groups: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantile-binned means + SE. groups=None -> trial-level SE; groups -> subject-clustered SE."""
    e = np.quantile(x, np.linspace(0, 1, q + 1))
    e[0] -= 1e-9
    idx = np.clip(np.digitize(x, e) - 1, 0, q - 1)
    if groups is None:
        mx = np.array([x[idx == b].mean() for b in range(q)])
        my = np.array([y[idx == b].mean() for b in range(q)])
        se = np.array(
            [y[idx == b].std() / max(np.sqrt((idx == b).sum()), 1) for b in range(q)]
        )
        return mx, my, se
    mx, my, se = [], [], []
    for b in range(q):
        m = idx == b
        subs = np.unique(groups[m])
        sm = np.array([y[m & (groups == s)].mean() for s in subs])
        mx.append(x[m].mean())
        my.append(y[m].mean())
        se.append(sm.std(ddof=1) / np.sqrt(len(subs)))
    return np.array(mx), np.array(my), np.array(se)


def boot_line(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    B: int = 400,
    seed: int = 0,
    return_slope: bool = False,
) -> Tuple[np.ndarray, ...]:
    """OLS line + case-bootstrap band. return_slope adds the point-estimate slope (fig_ms5)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    cs = np.empty((B, len(grid)))
    b, a = np.polyfit(x, y, 1)
    mean = a + b * grid
    for i in range(B):
        j = rng.randint(0, n, n)
        bb, aa = np.polyfit(x[j], y[j], 1)
        cs[i] = aa + bb * grid
    lo, hi = np.percentile(cs, [2.5, 97.5], 0)
    return (mean, lo, hi, b) if return_slope else (mean, lo, hi)


def panel_label(
    ax: Axes, L: str, fs: float = 20, x: float = -0.1, y: float = 1.2
) -> None:
    """Bold capital tag at the top-left of a subpanel (fig_ms* style)."""
    ax.text(
        x,
        y,
        L,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ---------------------------------------------------------------------------
# One palette for the three dimensions, shared across figures.
# In double-centered panels each dimension's cloud is centered at (0, 0) by construction.
DIM_COLORS = {"color": "#E0A82E", "shape": "#2E9E8F", "line": "#CE7BA8"}


# The tally <-> magnitude "pole" scale, shared with fig3_poles: each participant colored by their
# model-free P(follow magnitude | divergent), purple = tally-consistent, orange = magnitude.
# Use only on axes independent of lambda (the pole score is computed from the same choices).
POLE_TALLY, POLE_MAG = "#542788", "#b35806"


def style_ms() -> Tuple[str, str]:
    """Manuscript (fig_ms*) seaborn theme; returns (LINE, DOT) colors."""
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", font_scale=1.55)
    return "black", "#8c8c8c"


# ---------------------------------------------------------------------------
# Contribution-profile axes, shared by fig_ms_model C/D and fig_supp_knowledge B: x in taught
# points won by (25..100), y starting at 0.25 behind a break mark.
POINTS = np.array([25, 50, 75, 100])
PROFILE_YLAB = "contribution to choice\n(100-point win = 1)"


def axis_break(ax: Axes, y: float, size: float = 0.022, gap: float = 0.016) -> None:
    """Two slanted ticks across the left spine at data-y `y`: this axis does not start at zero."""
    from matplotlib.patches import Rectangle

    y0, y1 = ax.get_ylim()
    yf = (y - y0) / (y1 - y0)
    lw = ax.spines["left"].get_linewidth()
    ax.add_patch(
        Rectangle(
            (-0.03, yf - gap / 2),
            0.06,
            gap,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            clip_on=False,
            zorder=9,
        )
    )
    for dy in (-gap / 2, gap / 2):
        ax.plot(
            [-size / 2, size / 2],
            [yf + dy - size / 2, yf + dy + size / 2],
            transform=ax.transAxes,
            color=ax.spines["left"].get_edgecolor(),
            lw=lw,
            clip_on=False,
            zorder=10,
            solid_capstyle="butt",
        )


def profile_axes(ax: Axes, top: float = 1.1, fs: float = 12, pad: float = 0.09) -> None:
    """Points-won x-axis and broken y-axis for a contribution profile. `pad` is the same fraction of
    each axis' data span below its first value, so (25, 0.25) sits at the same corner on both."""
    xlo = POINTS[0] - pad * (POINTS[-1] - POINTS[0])
    ylo = 0.25 - pad * 0.75
    ax.set(
        xticks=POINTS,
        yticks=[0.25, 0.5, 0.75, 1.0],
        xlim=(xlo, 100 + pad * 75),
        ylim=(ylo, top),
    )
    ax.spines["left"].set_bounds(ylo, 1.0)
    ax.set_xlabel("points won by", fontsize=fs)
    ax.set_ylabel(PROFILE_YLAB, fontsize=fs)
    axis_break(ax, ylo + 0.45 * (0.25 - ylo))


def tertile_profile(
    ax: Axes,
    tertiles: List[Dict[str, Any]],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    fs: float = 12,
    dodge: float = 2.25,
) -> None:
    """Contribution profile per tertile (slope_readout_profile.json), with the pure-tally line."""
    ax.plot(POINTS, np.ones(4), color="0.6", ls=(0, (1, 2)), lw=1.3, zorder=0)
    for g, t in enumerate(tertiles):
        m, lo, hi = (np.array(t[k]) for k in ("profile", "profile_lo", "profile_hi"))
        ax.errorbar(
            POINTS + (g - 1) * dodge,
            m,
            yerr=[m - lo, hi - m],
            color=colors[g],
            marker="os^"[g],
            ms=7,
            lw=2,
            label=labels[g],
        )
    profile_axes(ax, fs=fs)
    ax.legend(
        title=title,
        fontsize=fs - 2,
        title_fontsize=fs - 2,
        frameon=False,
        loc="lower right",
        handlelength=1.8,
    )
