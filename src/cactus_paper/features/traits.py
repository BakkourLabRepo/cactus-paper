"""traits.py — per-subject constructs derived from the fitted weight vector W_s.

These are functions of W_s, not fitted parameters. Concentration is the Gini coefficient of |w_sd|
(range 0 to 2/3 at D = 3).
"""

import numpy as np

__all__ = ["gain", "concentration", "minmax_ratio"]


def gain(absW: np.ndarray, axis: int = -1) -> np.ndarray:
    """`wgain` — mean |w| across dimensions. Raw scale, not logged."""
    return np.asarray(absW).mean(axis)


def concentration(absW: np.ndarray, axis: int = -1) -> np.ndarray:
    """Weight concentration: Gini coefficient of |w| across dimensions.

    0 = perfectly even weights, (D-1)/D = all the weight on one cue (2/3 for D = 3). Works on any
    trailing-axis array, so it takes the per-draw (draw, subject, dim) cube as well as the
    posterior-mean (subject, dim) matrix.
    """
    x = np.sort(np.asarray(absW, float), axis=axis)
    n = x.shape[axis]
    idx = np.arange(1, n + 1).reshape(
        [n if i == (axis % x.ndim) else 1 for i in range(x.ndim)]
    )
    return (2.0 * (idx * x).sum(axis) / (n * x.sum(axis))) - (n + 1.0) / n


def minmax_ratio(absW: np.ndarray, axis: int = -1) -> np.ndarray:
    """min|w| / max|w| — descriptive only; not the concentration metric."""
    a = np.asarray(absW, float)
    return a.min(axis) / a.max(axis)
