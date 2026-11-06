"""Shared choice-trial helpers: per-dimension values and differences, taught levels, read noise.

Derives objective (taught) per-dimension value structure from choice_data.csv.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from cactus_paper.constants import DIMS, POS

# canonical unordered level-pair enumeration, index 0-9
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
PAIR_ID = np.full((5, 5), -1, dtype=int)
for idx, (i, j) in enumerate(PAIRS):
    PAIR_ID[i, j] = idx
    PAIR_ID[j, i] = idx


def _side_field(cd, s, d, field):
    out = np.full(len(cd), np.nan)
    for p in POS:
        dm = cd[f"choice_trial_{s}_{p}_dim"].values
        vl = cd[f"choice_trial_{s}_{p}_{field}"].values
        m = dm == d
        out[m] = vl[m]
    return out


def raw_values_and_diffs(cd: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Vl, Vr, VD), each (N,3) DIMS-order: taught left/right point values and
    their difference (right - left). Values are objective (taught {0,25,50,75,100})."""
    Vl = np.stack([_side_field(cd, "left", d, "value") for d in DIMS], 1)
    Vr = np.stack([_side_field(cd, "right", d, "value") for d in DIMS], 1)
    assert not np.isnan(Vl).any() and not np.isnan(Vr).any(), (
        "objective per-dim values incomplete"
    )
    return Vl, Vr, Vr - Vl


def taught_levels(Vl: np.ndarray, Vr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(Vl, Vr) taught point values -> (kl, kr) integer level indices 0-4.

    The taught values are {0, 25, 50, 75, 100}, so the level is value/25.
    """
    return np.round(Vl / 25.0).astype(int), np.round(Vr / 25.0).astype(int)


def follow_magnitude(Y: np.ndarray, VD: np.ndarray) -> np.ndarray:
    """Per-trial binary: did the participant choose the summed-magnitude (categorical) winner?

    Same `follow` definition as cactus_paper.design.pole_score, returned per trial.
    """
    agg = np.sign(VD.sum(1))
    Y = np.asarray(Y).astype(int)
    return (((Y == 1) & (agg > 0)) | ((Y == 0) & (agg < 0))).astype(int)


def read_noise(ld: pd.DataFrame, sig_hi: float = 400.0) -> Dict[Tuple[int, str], float]:
    """Read noise sigma per (run_id, dim), in points, from learning accuracy.

    Solves mean_t Phi(|Delta_t| / (sigma * sqrt2)) = accuracy over that cell's own |Delta|
    distribution. Floors at 1e-3 (better than a perfect read predicts) and caps at `sig_hi`
    (at or below chance)."""
    from scipy import stats
    from scipy.optimize import brentq

    out = {}
    for (rid, d), g in ld.groupby(["run_id", "dim"]):
        delta = np.abs(g["left_vs_right_rew_diff"].values.astype(float))
        delta = delta[np.isfinite(delta) & (delta > 0)]
        acc = float(g["correct"].mean())
        if len(delta) < 5:
            out[(rid, d)] = np.nan
            continue
        f = lambda s: np.mean(stats.norm.cdf(delta / (s * np.sqrt(2)))) - acc
        if f(1e-3) < 0:
            out[(rid, d)] = 1e-3
        elif f(sig_hi) > 0:
            out[(rid, d)] = sig_hi
        else:
            out[(rid, d)] = float(brentq(f, 1e-3, sig_hi))
    return out


def mean_log_read_noise(D: Dict[str, Any]) -> np.ndarray:
    """(nS,) mean over the three dimensions of log read noise, in load_design() participant order."""
    import pandas as pd

    from cactus_paper.paths import DATA

    su = pd.unique(D["cd"].run_id)
    ld = pd.read_csv(DATA / "learning_data.csv")
    sig = read_noise(ld[ld.run_id.isin(su)])
    return np.array([[np.log(sig[(r, d)]) for d in DIMS] for r in su]).mean(1)
