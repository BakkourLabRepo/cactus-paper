"""stats.json: validation (CV ladder, fits table, identifiability, recovery, device) and the lambda
continuum. Mostly precomputed results/fits/ artifacts, which are NOT refit on an alternate sample.
"""

import json
from typing import Any, Dict

import numpy as np

from cactus_paper.manuscript.context import Context, summ
from cactus_paper.paths import FITS

RES = str(FITS)


def validation(ctx: Context) -> Dict[str, Any]:
    print(
        "[validation] CV ladder + recovery + device (model of record) ...", flush=True
    )
    out = {}
    # Shared-folds 5-fold CV over the model ladder (cactus_paper.models.cv, CV_SEED=2026).
    # participant_map_lambda and participant_map differ only in lambda.
    _cv = json.load(open(f"{RES}/cv.json"))
    out["cv"] = {
        "_folds": "5-fold trial-wise, shared fold assignments (SEED=2026), objective taught levels",
        "elpd": {k: round(float(v), 1) for k, v in _cv["elpd"].items()},
        # delta_div / delta_nondiv split the contrast by trial type (divergent = incongruent)
        **{
            k: {
                "delta": round(v["delta"], 1),
                "se": round(v["se"], 1),
                **{
                    f"delta_{t}": round(v[f"delta_{t}"], 1)
                    for t in ("div", "nondiv")
                    if f"delta_{t}" in v
                },
            }
            for k, v in _cv["contrasts"].items()
        },
    }
    # Sampler geometry + convergence for every fit of record and validation refit, one row each
    # (cactus_paper.models.fit convergence-table; R-hat/ESS on the SAMPLED parameters only).
    out["fits_table"] = json.load(open(f"{RES}/fits_table.json"))

    # Cross-model identifiability.
    out["identifiability"] = json.load(open(f"{RES}/identifiability.json"))

    _rec = json.load(open(f"{RES}/recovery.json"))
    out["recovery"] = {
        "n_reps": _rec["_n_reps"],
        "_protocol": "10 reps @ 1500/1500 x 4 chains",
        "coverage_mean": _rec["coverage_mean_all"],
        "r_min_across_weights": _rec["r_min_across_weights"],
        **{
            f"corr_{k}": {"mean": _rec[k]["r_mean"], "min": _rec[k]["r_min"]}
            for k in ("W_color", "W_shape", "W_line", "lam", "gain")
        },
        "_caveat": "weights r >= 0.83 (color .89 / shape .86 / line .93), readout r = 0.92, "
        "coverage 92%. Recovery softens monotonically with model complexity "
        "(WL ~.92 -> hier .851 -> band .831) while lambda stays at .92.",
    }

    # Pointer device under the model of record (cactus_paper.analyses.robustness --device-band): c_dev on
    # the readout + w_dev on the weights; c_dev = w_dev = 0 recovers the model of record.
    if ctx.tag:
        print("  ! alternate sample: stats['device'] skipped")
    else:
        out["device"] = json.load(open(f"{RES}/mor_device.json"))
    return out


def continuum(ctx: Context) -> Dict[str, Any]:
    """(a) continuum vs types: GMM on logit(lambda), the scale on which the model assumes normality.
    (b) per-participant elpd gain from lambda.
    """
    from scipy import special as _special
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew
    from sklearn.mixture import GaussianMixture as _GMM

    lam_s, sc, nS = ctx.lam_s, ctx.sc, ctx.nS
    _lam_logit = _special.logit(np.clip(lam_s, 1e-4, 1 - 1e-4)).reshape(-1, 1)
    _bic = {
        k: float(_GMM(k, n_init=20, random_state=0).fit(_lam_logit).bic(_lam_logit))
        for k in (1, 2, 3)
    }
    _sk, _ku = float(_skew(_lam_logit.ravel())), float(_kurt(_lam_logit.ravel()))
    # participant_map_lambda - participant_map per participant (they differ only in lambda). Canonical sample only.
    _cv_lpd = np.load(f"{RES}/cv_lpd.npz")
    _lam_ok = not ctx.tag and len(_cv_lpd["participant_map_lambda"]) == len(sc)
    if _lam_ok:
        _by_subj = lambda a: np.array([a[sc == s].sum() for s in range(nS)])
        _lam_gain = _by_subj(_cv_lpd["participant_map_lambda"]) - _by_subj(
            _cv_lpd["participant_map"]
        )
        _srt = np.sort(_lam_gain)[::-1]
    else:
        print("  ! alternate sample: stats['lam_who'] skipped (cv not re-run)")

    out = {
        "continuum": {
            "_scale": "GMM fit on logit(lambda); the lambda-scale version is a sigmoid artifact",
            "gmm_bic_logit": {str(k): round(v, 1) for k, v in _bic.items()},
            "gmm_best_k": int(min(_bic, key=_bic.get)),
            "bimodality_coef_logit": round((_sk**2 + 1) / (_ku + 3), 3),
            "skew_logit": round(_sk, 3),
            "n_lam_below_05": int((lam_s < 0.05).sum()),
            "n_lam_above_95": int((lam_s > 0.95).sum()),
        }
    }
    if _lam_ok:
        out["lam_who"] = {
            "_what": "per-participant elpd gain from adding lambda (participant_map_lambda - participant_map, "
            "matched on per-participant value-function flexibility)",
            "total": round(float(_lam_gain.sum()), 1),
            "n_helped": int((_lam_gain > 0).sum()),
            "n_subj": int(nS),
            "n_hurt_gt2": int((_lam_gain < -2).sum()),
            "median_gain": round(float(np.median(_lam_gain)), 2),
            "share_top5": round(float(_srt[:5].sum() / _lam_gain.sum()), 3),
            "share_top10": round(float(_srt[:10].sum() / _lam_gain.sum()), 3),
            "share_top20": round(float(_srt[:20].sum() / _lam_gain.sum()), 3),
            "corr_gain_with_lam": round(float(np.corrcoef(_lam_gain, lam_s)[0, 1]), 3),
            # propagated over lam's draws (_lam_gain is a fixed CV quantity)
            "corr_gain_with_lam_propagated": summ(
                [
                    np.corrcoef(_lam_gain, ctx.lamD[d])[0, 1]
                    for d in range(ctx.lamD.shape[0])
                ]
            ),
        }
    return out
