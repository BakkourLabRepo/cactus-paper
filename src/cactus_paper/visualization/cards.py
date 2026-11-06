"""cactus_paper.cards — shared six-slot trial-card geometry, card renderers, and mouse-trajectory
examples, used by fig2_task.py and fig_ms5_crossstream.py.
"""

import json
from typing import Any, Collection, Dict, Optional, Sequence, Tuple

import matplotlib.cm as mplcm
import matplotlib.collections as mcollections
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch

# ── card geometry (identical everywhere a card is drawn) ────────────────────────────
CW, CH = 7.0, 9.0
CX = [1.5, 5.5]  # option column x-centers [left, right]
RY = [7.2, 4.5, 1.8]  # attribute row y-centers [top, mid, bot]
R = 1.0
ALL = {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)}
CARD_BG, SLOT_HID, SLOT_EDG = "#EBEBEB", "#D0D0D0", "#999999"
SHAPE_FRAC = 0.80
XL, YL = (-0.15, 7.15), (-0.15, 9.15)  # identical card data-limits everywhere
PATH_CMAP = mplcm.plasma
SIDE_COL = {"left": 0, "right": 1}
POS_ROW = {"top": 0, "mid": 1, "bot": 2}

PAYNE_OPTIONWISE_TARGET = +0.65  # Payne 1976: within-option positive
PAYNE_ATTRWISE_TARGET = -0.65
MIN_TRANSITIONS, LEGIBLE_N = 6, 9


def draw_slot(
    ax: Axes,
    cx: float,
    cy: float,
    stim: Optional[Tuple[str, Any]],
    highlight: bool = False,
    z: int = 0,
) -> None:
    if stim is None:
        ax.add_patch(
            Circle(
                (cx, cy),
                R,
                facecolor=SLOT_HID,
                edgecolor=SLOT_EDG,
                lw=1.2,
                zorder=2 + z,
            )
        )
        return
    kind, val = stim
    if kind == "color":
        ax.add_patch(
            Circle(
                (cx, cy), R, facecolor=val, edgecolor="#111111", lw=1.5, zorder=2 + z
            )
        )
    elif kind == "line":
        ax.add_patch(
            Circle(
                (cx, cy),
                R,
                facecolor="white",
                edgecolor="#111111",
                lw=1.5,
                zorder=2 + z,
            )
        )
        th = np.radians(val)
        L = R * 0.90
        ax.plot(
            [cx, cx + L * np.cos(th)],
            [cy, cy + L * np.sin(th)],
            color="#111111",
            lw=2.2,
            zorder=3 + z,
        )
    elif kind == "shape":
        ax.add_patch(
            Circle(
                (cx, cy),
                R,
                facecolor="white",
                edgecolor="#111111",
                lw=1.5,
                zorder=2 + z,
            )
        )
        sz = R * SHAPE_FRAC
        ax.imshow(
            val,
            extent=[cx - sz, cx + sz, cy - sz, cy + sz],
            zorder=3 + z,
            interpolation="bilinear",
            aspect="auto",
        )
    if highlight:
        ax.add_patch(
            Circle(
                (cx, cy),
                R + 0.14,
                facecolor="none",
                edgecolor="#444444",
                lw=2.8,
                zorder=4 + z,
                alpha=0.85,
            )
        )


def draw_card(
    ax: Axes,
    stims: Dict[Tuple[int, int], Optional[Tuple[str, Any]]],
    ox: float = 0.0,
    oy: float = 0.0,
    revealed: Collection[Tuple[int, int]] = ALL,
    highlighted: Collection[Tuple[int, int]] = frozenset(),
    arrows: Sequence[Tuple[float, float, float, float]] = (),
    dark: bool = False,
    z: int = 0,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.35 + ox, 0.35 + oy),
            CW - 0.7,
            CH - 0.7,
            boxstyle="round,pad=0.45",
            facecolor=CARD_BG,
            edgecolor="#AAAAAA",
            lw=1.5,
            zorder=1 + z,
        )
    )
    for (c, r), stim in stims.items():
        show = (not dark) or ((c, r) in revealed)
        draw_slot(
            ax,
            CX[c] + ox,
            RY[r] + oy,
            stim if show else None,
            highlight=(c, r) in highlighted,
            z=z,
        )
    for x0, y0, x1, y1 in arrows:
        ax.annotate(
            "",
            xy=(x1 + ox, y1 + oy),
            xytext=(x0 + ox, y0 + oy),
            zorder=6 + z,
            arrowprops=dict(
                arrowstyle="->", color="#333333", lw=2.5, mutation_scale=20
            ),
        )


def card_ax(
    fig: Figure,
    x_in: float,
    y_in: float,
    figw_in: float,
    figh_in: float,
    w_in: float,
    h_in: float,
) -> Axes:
    """A card axes of fixed physical size + fixed data limits (identical cards everywhere)."""
    ax = fig.add_axes([x_in / figw_in, y_in / figh_in, w_in / figw_in, h_in / figh_in])
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_aspect("auto")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return ax


# ═════════════════════════════════ mouse-trajectory examples ═════════════════════════
def _aoi_to_slot(name):
    p = name.lstrip("#").split("-")
    if len(p) < 2:
        return None
    c, r = SIDE_COL.get(p[0]), POS_ROW.get(p[-1])
    return None if c is None or r is None else (c, r)


def _aoi_at(x, y, targets):
    for name, b in targets.items():
        if b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]:
            return name
    return None


def _screen_to_card(xs, ys, targets):
    cx = lambda b: b["x"] + b["width"] / 2
    cy = lambda b: b["y"] + b["height"] / 2
    lt, rt = targets.get("#left-top"), targets.get("#right-top")
    lb, rb = targets.get("#left-bot"), targets.get("#right-bot")
    if None in (lt, rt, lb, rb):
        bx = list(targets.values())
        x0 = min(b["x"] for b in bx)
        x1 = max(b["x"] + b["width"] for b in bx)
        y0 = min(b["y"] for b in bx)
        y1 = max(b["y"] + b["height"] for b in bx)
        return (
            [(x - x0) / (x1 - x0) * CW for x in xs],
            [CH - (y - y0) / (y1 - y0) * CH for y in ys],
        )
    s_lcx = (cx(lt) + cx(lb)) / 2
    s_rcx = (cx(rt) + cx(rb)) / 2
    s_tcy = (cy(lt) + cy(rt)) / 2
    s_bcy = (cy(lb) + cy(rb)) / 2
    xsc = (CX[1] - CX[0]) / (s_rcx - s_lcx)
    ysc = (RY[0] - RY[2]) / (s_bcy - s_tcy)
    return (
        [CX[0] + (x - s_lcx) * xsc for x in xs],
        [RY[0] - (y - s_tcy) * ysc for y in ys],
    )


def plot_trajectory(row: pd.Series, ax: Axes, title: str = "") -> None:
    """Draw one mouse-tracking trace on a dark (hidden-slot) card."""
    targets = json.loads(row["mouse_tracking_targets"])
    data = json.loads(row["mouse_tracking_data"])
    cx, cy = _screen_to_card([e["x"] for e in data], [e["y"] for e in data], targets)
    draw_card(ax, {cr: None for cr in ALL}, dark=True)
    n = len(cx)
    if n > 1:
        segs = [[(cx[i], cy[i]), (cx[i + 1], cy[i + 1])] for i in range(n - 1)]
        cols = [PATH_CMAP(t) for t in np.linspace(0, 1, n - 1)]
        ax.add_collection(
            mcollections.LineCollection(segs, colors=cols, linewidths=1.8, zorder=5),
            autolim=False,
        )
    prev = None
    for x, y, e in zip(cx, cy, data):
        aoi = _aoi_at(e["x"], e["y"], targets)
        if aoi and aoi != prev and _aoi_to_slot(aoi):
            ax.plot(x, y, "o", color="black", markersize=5, zorder=7, alpha=0.9)
            prev = aoi
    ax.set_title(title, fontsize=10.5, color="#333333", pad=6)


def _pick(cd, target):
    elig = cd[cd.n_trans >= MIN_TRANSITIONS]
    near = elig.iloc[np.argsort(np.abs(elig.payne_index.values - target))[:60]].copy()
    return near.iloc[np.argsort(np.abs(near.n_trans.values - LEGIBLE_N))[0]]


def load_trajectory_pair(
    data_dir: str, kept_run_ids: Sequence[int]
) -> Tuple[pd.Series, pd.Series]:
    """One option-wise and one attribute-wise example trace: nearest to the target Payne index
    among legible (>= MIN_TRANSITIONS, close to LEGIBLE_N) trials."""
    cd = pd.read_csv(f"{data_dir}/choice_data.csv")
    dc = pd.read_csv(f"{data_dir}/dim_counts.csv")[
        ["run_id", "trial_idx", "payne_index", "within_option", "between_options"]
    ]
    cd = (
        cd[cd.run_id.isin(kept_run_ids)]
        .merge(dc, on=["run_id", "trial_idx"], how="left")
        .dropna(subset=["mouse_tracking_data", "mouse_tracking_targets", "payne_index"])
        .reset_index(drop=True)
    )
    cd["n_trans"] = cd.within_option + cd.between_options
    row_opt = _pick(cd, PAYNE_OPTIONWISE_TARGET)
    row_attr = _pick(cd, PAYNE_ATTRWISE_TARGET)
    return row_opt, row_attr


def add_trajectory_colorbar(
    ax: Axes, fontsize: float = 10.5, orientation: str = "vertical"
) -> None:
    """orientation="vertical" (default): inset to the RIGHT of ax, Start->End bottom to top.
    orientation="horizontal": inset BELOW ax, Start->End left to right."""
    sm = mplcm.ScalarMappable(cmap=PATH_CMAP, norm=mplcolors.Normalize(0, 1))
    sm.set_array([])
    if orientation == "horizontal":
        cax = ax.inset_axes([0.12, -0.20, 0.76, 0.07])
    else:
        cax = ax.inset_axes([1.12, 0.12, 0.08, 0.76])
    cb = ax.figure.colorbar(sm, cax=cax, orientation=orientation)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Start", "End"])
    cb.ax.tick_params(labelsize=fontsize - 1)
    cb.set_label("Trajectory", fontsize=fontsize, labelpad=6)
