"""stats.json: knowledge -> salience, not metric (Fig 4C/D/E)."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from cactus_paper.constants import DIMS
from cactus_paper.manuscript.context import Context, summ


def double_center(M: np.ndarray) -> np.ndarray:
    """Remove participant and attribute means from a (subject, dim) array."""
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def fig4(ctx: Context) -> Dict[str, Any]:
    print("[fig4] knowledge -> salience, not metric ...", flush=True)
    absW, dacc, lamD, WD, K = ctx.absW, ctx.dacc, ctx.lamD, ctx.WD, ctx.K
    L = pd.DataFrame(
        [
            dict(run_id=r, dim=d, W=absW[i, j], dacc=dacc[i, j])
            for i, r in enumerate(ctx.su)
            for j, d in enumerate(DIMS)
        ]
    )
    _dc2 = lambda c: (
        L[c]
        - L.groupby("run_id")[c].transform("mean")
        - L.groupby("dim")[c].transform("mean")
        + L[c].mean()
    )
    _dcD = double_center(dacc).ravel()
    # atol, not the default: the posterior arrays are float32, so residuals near zero differ by ~1e-6
    assert np.allclose(double_center(absW).ravel(), _dc2("W").values, atol=1e-4), (
        "double-centering forms disagree"
    )
    _sub = lambda c: L[c] - L.groupby("run_id")[c].transform("mean")
    return {
        "knowledge": {
            # propagated (per draw) -> text, with HDI
            "corr_gain_K": summ(
                [np.corrcoef(ctx.gainD[d], K)[0, 1] for d in range(lamD.shape[0])]
            ),
            "corr_lam_K": summ(
                [np.corrcoef(lamD[d], K)[0, 1] for d in range(lamD.shape[0])]
            ),
            # posterior-mean point estimates -> what Fig 4C/D plot and annotate
            "corr_gain_K_pointest": round(float(np.corrcoef(ctx.gain, K)[0, 1]), 3),
            "corr_lam_K_pointest": round(float(np.corrcoef(ctx.lam_s, K)[0, 1]), 3),
            # weight <-> per-attribute accuracy at three levels of control
            "corr_weight_dacc_raw": round(float(np.corrcoef(L.W, L.dacc)[0, 1]), 3),
            "corr_weight_dacc_subject_centered": round(
                float(np.corrcoef(_sub("W"), _sub("dacc"))[0, 1]), 3
            ),
            "corr_weight_dacc_double_centered": round(
                float(np.corrcoef(_dc2("W"), _dc2("dacc"))[0, 1]), 3
            ),
            # propagated over W's draws (dacc is observed)
            "corr_weight_dacc_raw_propagated": summ(
                [
                    np.corrcoef(np.abs(WD[d]).ravel(), L.dacc.values)[0, 1]
                    for d in range(WD.shape[0])
                ]
            ),
            "corr_weight_dacc_subject_centered_propagated": summ(
                [
                    np.corrcoef(
                        (np.abs(WD[d]) - np.abs(WD[d]).mean(1, keepdims=True)).ravel(),
                        (dacc - np.nanmean(dacc, 1, keepdims=True)).ravel(),
                    )[0, 1]
                    for d in range(WD.shape[0])
                ]
            ),
            # double-centered per draw of W. Rows of L run subject-major (i*3 + j), so an (nS,3) array
            # ravels into L's order.
            "corr_weight_dacc_double_centered_propagated": summ(
                [
                    np.corrcoef(double_center(np.abs(WD[d])).ravel(), _dcD).ravel()[1]
                    for d in range(WD.shape[0])
                ]
            ),
            "n_subj_dim": int(len(L)),
        }
    }
