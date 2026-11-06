"""fig2_task.py — Figure 2 (#fig-task): the valuation & choice task, one reproducible composite.

  A  three example trial cards, diagonally stacked (drawn back->front) — each a full two-option
     trial with a different (real, per-trial-randomized) attribute row-order and stimulus values.
  B  the two trial types: incongruent (magnitude vs tally disagree) and congruent (agree).
  C  the reveal sequence of one trial: reference (all revealed 250 ms) -> hidden -> first reveal
     -> second reveal (mouse-contingent aperture). Bottom row, full width.

Every card is drawn in an axes of identical physical size and data limits (card geometry and slot
renderer from cactus_paper.visualization.cards). Stimulus colors follow the experiment's CIELAB locus (L=50,
chroma 60; stimulus-helpers.js). Writes fig2_task.png and .pdf.
Run: pixi run python scripts/exp2/figures/fig2_task.py  (any cwd)
"""

from functools import lru_cache
from typing import Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from cactus_paper.paths import ROOT
from cactus_paper.paths import fig as figpath
from cactus_paper.visualization.cards import (
    ALL as _ALL,
)
from cactus_paper.visualization.cards import (
    CH as _CH,
)
from cactus_paper.visualization.cards import (
    CW as _CW,
)
from cactus_paper.visualization.cards import (
    CX as _CX,
)
from cactus_paper.visualization.cards import (
    RY as _RY,
)
from cactus_paper.visualization.cards import (
    XL as _XL,
)
from cactus_paper.visualization.cards import (
    YL as _YL,
)
from cactus_paper.visualization.cards import (
    R as _R,
)
from cactus_paper.visualization.cards import (
    card_ax as _cards_card_ax,
)
from cactus_paper.visualization.cards import (
    draw_card as _draw_card,
)

OUT = figpath("fig2_task")
plt.rcParams["font.family"] = "Arial"

ASSET = (
    ROOT / "experiment_code" / "exp2" / "public" / "assets" / "transparent_VCSshapes"
)

# physical layout (inches) — one card = CW_IN x CH_IN, identical across panels
FIGW, FIGH = 13.6, 9.5
_RATIO = (_YL[1] - _YL[0]) / (_XL[1] - _XL[0])  # card h/w to keep circles round
CW_IN = 2.15  # B cards (~2.74 tall)
CH_IN = CW_IN * _RATIO
B_CW_IN = 2.15  # C's (reveal-sequence) cards, same scale as B's
B_CH_IN = B_CW_IN * _RATIO
FS_TITLE, FS_LETTER, FS_SMALL, FS_VAL, FS_VERDICT = 12.5, 21, 10.5, 12.5, 10.5


def _card_ax(fig, x_in, y_in, w_in=CW_IN, h_in=CH_IN):
    return _cards_card_ax(fig, x_in, y_in, FIGW, FIGH, w_in, h_in)


@lru_cache(maxsize=None)
def _load_vcs(idx):
    return mpimg.imread(f"{ASSET}/VCS_{int(idx):03d}_transparent_cv2.png")


# ── CIELAB stimulus color (matches experiment_code stimulus-helpers.js) ──────────────
def _lab_to_rgb(L, a, b):
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    inv = lambda t: t**3 if t**3 > 0.008856 else (t - 16 / 116) / 7.787
    X, Y, Z = 95.047 * inv(fx) / 100, 100.0 * inv(fy) / 100, 108.883 * inv(fz) / 100
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bb = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    gam = lambda c: (
        (1.055 * max(0, min(1, c)) ** (1 / 2.4) - 0.055)
        if c > 0.0031308
        else 12.92 * max(0, c)
    )
    return tuple(gam(v) for v in (r, g, bb))


def color_at(
    deg: float, L: float = 50, aC: float = 20, bC: float = 20, rad: float = 60
) -> Tuple[float, float, float]:
    th = np.radians(deg)
    return _lab_to_rgb(L, aC + rad * np.cos(th), bC + rad * np.sin(th))


# ═════════════════════════════════ PANEL A: stimulus cards ═══════════════════════════
def _slot(attr, param):
    if attr == "color":
        return ("color", color_at(param))
    if attr == "line":
        return ("line", param)
    return ("shape", _load_vcs(param))


# each card: attribute rows top->bottom, (attr, left_param, right_param); real per-trial row-orders.
_CARDS_A = [
    [("color", 25, 300), ("line", 205, 160), ("shape", 20, 300)],
    [("line", 35, 125), ("shape", 95, 250), ("color", 95, 235)],
    [("shape", 150, 55), ("color", 165, 30), ("line", 100, 20)],
]
_A_OFF = [
    (0.0, 0.0),
    (3.3, -2.9),
    (6.6, -5.8),
]  # diagonal stack, back -> front (less overlap)
# A's data limits hug the content: padded card extent (boxstyle pad=0.45 in cards.draw_card) is
# x in [-0.10, 13.70], y in [-5.90, 9.10]; the arrow+label reach to about x=-2.6, y=-6.95.
_A_XL, _A_YL = (-3.0, 13.9), (-7.2, 9.3)


def draw_A(ax: Axes) -> None:
    ax.set_autoscale_on(False)
    ax.axis("off")
    for k, ((dx, dy), rows) in enumerate(zip(_A_OFF, _CARDS_A)):
        stims = {
            (oi, ri): _slot(attr, (lp, rp)[oi])
            for ri, (attr, lp, rp) in enumerate(rows)
            for oi in (0, 1)
        }
        _draw_card(
            ax, stims, ox=dx, oy=dy, z=20 * k
        )  # later card fully occludes earlier
    # arrow lying tangent to the cards' bottom-left corners, parallel to the stacking diagonal
    d = np.array(_A_OFF[1]) - np.array(_A_OFF[0])
    u = d / np.hypot(*d)
    nrm = np.array([d[1], -d[0]])
    nrm = nrm / np.hypot(*nrm)  # outward toward lower-left
    c0 = np.array([-0.1, -0.1])  # back card outer bottom-left corner
    c2 = np.array([-0.1 + _A_OFF[-1][0], -0.1 + _A_OFF[-1][1]])
    a0 = c0 + nrm * 0.30 - u * 0.2
    a1 = c2 + nrm * 0.30 + u * 1.2
    ax.add_patch(
        FancyArrowPatch(
            tuple(a0),
            tuple(a1),
            arrowstyle="-|>",
            mutation_scale=22,
            lw=2.4,
            color="#333333",
            shrinkA=0,
            shrinkB=0,
            zorder=100,
        )
    )
    lbl = a0 + nrm * 1.7
    ax.text(
        lbl[0],
        lbl[1],
        "120 choice\ntrials",
        ha="center",
        va="center",
        fontsize=FS_SMALL,
        color="#333333",
        linespacing=1.15,
    )
    ax.set_xlim(*_A_XL)
    ax.set_ylim(*_A_YL)
    ax.set_aspect("auto")


# ═════════════════════════════════ PANEL C: reveal sequence ══════════════════════════
_CAT_STIMS = {
    (0, 0): ("color", color_at(95)),
    (1, 0): ("color", color_at(300)),
    (0, 1): ("line", 215),
    (1, 1): ("line", 198),
    (0, 2): ("shape", _load_vcs(1)),
    (1, 2): ("shape", _load_vcs(210)),
}
_cx0, _cx1 = _CX
_ty, _my, _by = _RY
_FRAMES = [
    dict(revealed=_ALL, highlighted=frozenset(), arrows=(), dark=False),
    dict(revealed=frozenset(), highlighted=frozenset(), arrows=(), dark=True),
    dict(
        revealed={(0, 0)},
        highlighted={(0, 0)},
        arrows=[(_CW / 2, _CH / 2, 2.47, 6.35)],
        dark=True,
    ),
    dict(
        revealed={(0, 1)},
        highlighted={(0, 1)},
        arrows=[(_cx0, _ty, _cx0, _my + _R + 0.12)],
        dark=True,
    ),
]
_TITLES = [
    "Reference\n(all revealed 250 ms)",
    "Hidden display",
    "First reveal",
    "Second reveal",
]


def draw_B(axes: Sequence[Axes]) -> None:
    for ax, fr, title in zip(axes, _FRAMES, _TITLES):
        _draw_card(
            ax,
            _CAT_STIMS,
            revealed=fr["revealed"],
            highlighted=fr["highlighted"],
            arrows=fr["arrows"],
            dark=fr["dark"],
        )
        ax.set_title(title, fontsize=FS_SMALL, pad=6, color="#333333")


# ═════════════════════════════════ PANEL B: trial types ══════════════════════════════
_DIM_LBL = ["Color", "Shape", "Line"]
_INCONG = {"A": [100, 25, 25], "B": [25, 50, 50]}
_CONG = {"A": [75, 75, 25], "B": [25, 50, 50]}


def draw_C_card(
    ax: Axes,
    vals: Sequence[int],
    title: str,
    tcolor: str,
    cat_v: str,
    rel_v: str,
    dim_labels: bool = False,
) -> None:
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", color=tcolor, pad=6)
    ax.add_patch(
        FancyBboxPatch(
            (0.35, 0.35),
            _CW - 0.7,
            _CH - 0.7,
            boxstyle="round,pad=0.45",
            fc="#EBEBEB",
            ec="#AAAAAA",
            lw=1.5,
            zorder=0,
        )
    )
    if dim_labels:
        for di, y in enumerate(_RY):
            ax.text(
                -0.9,
                y,
                _DIM_LBL[di],
                ha="right",
                va="center",
                fontsize=FS_SMALL,
                fontweight="bold",
                color="#000000",
                clip_on=False,
            )
    for di, y in enumerate(_RY):
        for oi, x in enumerate(_CX):
            ax.add_patch(
                Circle((x, y), _R + 0.08, fc="#B0B0B0", ec="#999999", lw=0.8, zorder=1)
            )
            ax.add_patch(
                Circle((x, y), _R, fc="#E0E0E0", ec="#999999", lw=2.0, zorder=2)
            )
            ax.text(
                x,
                y,
                str(vals["AB"[oi]][di]),
                ha="center",
                va="center",
                fontsize=FS_VAL,
                fontweight="bold",
                color="#000000",
                zorder=4,
            )
    for k, txt in enumerate([cat_v, rel_v]):
        ax.text(
            _CW / 2,
            -0.9 - 1.35 * k,
            txt,
            ha="center",
            va="center",
            fontsize=FS_VERDICT,
            color="#000000",
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F0F0F0", ec="none"),
        )


# ═════════════════════════════════ assemble (inch-based, uniform cards) ══════════════
# top row: A and B; bottom row: C at full width
fig = plt.figure(figsize=(FIGW, FIGH), facecolor="white")

GAP = 0.20
ROW2_B = 0.45  # bottom of the B row
row2_top = ROW2_B + B_CH_IN

GROW = 0.5  # gap between the C row and the A/B row above it
ROW1_B = row2_top + GROW  # bottom of the A/B row

# A: stacked cards, sized so one card renders at the same inches per data-unit as a B card
_in_per_unit = CW_IN / (_XL[1] - _XL[0])  # B/C's scale: inches per data-unit
A_W = _in_per_unit * (_A_XL[1] - _A_XL[0])
A_H = A_W * (_A_YL[1] - _A_YL[0]) / (_A_XL[1] - _A_XL[0])
B_W = 2 * CW_IN + GAP
CGAP = 0.6  # gap between A and B
BLOCK_W = A_W + CGAP + B_W
A_X = (FIGW - BLOCK_W) / 2  # center the A+B block on the canvas
A_Y = ROW1_B  # bottom-aligned to the row baseline
axA = fig.add_axes([A_X / FIGW, A_Y / FIGH, A_W / FIGW, A_H / FIGH])
draw_A(axA)

# B: 2 value cards (dim labels on the first, verdicts below), top-aligned with A
bx0 = A_X + A_W + CGAP
by0 = ROW1_B + (A_H - CH_IN)
axB1 = _card_ax(fig, bx0, by0)
axB2 = _card_ax(fig, bx0 + CW_IN + GAP, by0)
draw_C_card(
    axB1,
    _INCONG,
    "Incongruent",
    "#4878CF",
    "Magnitude:  Σ A > Σ B  →  A",
    "Tally:  B wins 2/3  →  B",
    dim_labels=True,
)
draw_C_card(
    axB2,
    _CONG,
    "Congruent",
    "#353535",
    "Magnitude:  Σ A > Σ B  →  A",
    "Tally:  A wins 2/3  →  A",
)

# C: 4 reveal-frame cards, centered across the full canvas width, bottom row
cx0 = (FIGW - (4 * B_CW_IN + 3 * GAP)) / 2
axC = [
    _card_ax(fig, cx0 + i * (B_CW_IN + GAP), ROW2_B, B_CW_IN, B_CH_IN) for i in range(4)
]
draw_B(axC)

# panel letters — top-left of each panel, above its title; A and B share one y (pinned to A)
LY_ROW1 = ROW1_B + A_H + 0.5
LY_ROW2 = row2_top + 0.5
for x_in, y_in, L in [
    (A_X - 0.35, LY_ROW1, "A"),
    (bx0 - 0.55, LY_ROW1, "B"),
    (A_X - 0.35, LY_ROW2, "C"),  # same left margin as A, not centered on C's own cards
]:
    fig.text(
        x_in / FIGW,
        y_in / FIGH,
        L,
        fontsize=FS_LETTER,
        fontweight="bold",
        va="top",
        ha="left",
    )

for ext in ("png", "pdf"):
    fig.savefig(
        f"{OUT}.{ext}",
        dpi=600 if ext == "png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
print("saved ->", f"{OUT}.png")
