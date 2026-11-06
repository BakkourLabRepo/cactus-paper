"""Stimulus geometry: the five taught bands, within-band positions, and mapping direction.

choice_frame() returns per-dimension, value-referenced within-band positions for every choice trial.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from cactus_paper.constants import BAND_WIDTH, BANDS, DIMS, POS
from cactus_paper.paths import DATA


def band_arrays(
    ang: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """angle array -> (lo, hi, cat, within-band position in [-0.5,+0.5], boundary distance 0-15)."""
    lo = np.full(ang.shape, np.nan)
    hi = np.full(ang.shape, np.nan)
    cat = np.full(ang.shape, -1, dtype=int)
    for l, h, c in BANDS:
        m = (ang >= l) & (ang <= h)
        lo[m], hi[m], cat[m] = l, h, c
    pos = (ang - lo) / BAND_WIDTH - 0.5
    bdist = np.minimum(ang - lo, hi - ang)
    return lo, hi, cat, pos, bdist


def choice_frame(
    D: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[int]]:
    """Per trial x dimension: taught levels, value-referenced within-band position, boundary dist."""
    cd = D["cd"]
    su = cd.run_id.drop_duplicates().tolist()
    cat0 = get_directions(su).reindex(su)[DIMS]
    desc = cat0.values > 50  # (nS,3) True = descending
    sc = D["sc"]

    def side(s, d, field):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s}_{p}_dim"].values
            vl = cd[f"choice_trial_{s}_{p}_{field}"].values
            m = dm == d
            out[m] = vl[m]
        return out

    rec = {}
    for j, d in enumerate(DIMS):
        aL, aR = side("left", d, "angle"), side("right", d, "angle")
        vL, vR = side("left", d, "value"), side("right", d, "value")
        _, _, _, pL, bL = band_arrays(aL)
        _, _, _, pR, bR = band_arrays(aR)
        flip = np.where(desc[sc, j], -1.0, 1.0)  # value-reference the position
        rec[d] = dict(
            kL=np.round(vL / 25).astype(int),
            kR=np.round(vR / 25).astype(int),
            pL=pL * flip,
            pR=pR * flip,
            bL=bL,
            bR=bR,
        )
    return rec, su


def get_directions(kept: Sequence[int]) -> pd.DataFrame:
    """Per participant, per dimension: 0.0 if category-0's taught reward is 0 (ascending),
    100.0 if it's 100 (descending)."""
    ld = pd.read_csv(f"{DATA}/learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]
    sub = ld[["run_id", "dim", "left_category", "left_rew"]].rename(
        columns={"left_category": "cat", "left_rew": "rew"}
    )
    sub2 = ld[["run_id", "dim", "right_category", "right_rew"]].rename(
        columns={"right_category": "cat", "right_rew": "rew"}
    )
    long = pd.concat([sub, sub2])
    cat0 = long[long.cat == 0].groupby(["run_id", "dim"])["rew"].mean().unstack("dim")
    return cat0  # (run_id, dim) -> 0.0 (ascending) or 100.0 (descending)
