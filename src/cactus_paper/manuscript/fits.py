"""stats.json `fits` namespace: the results/fits/ summaries plus two MOR posterior summaries.

Passes the precomputed validation / value-map JSON through (dropping entries flagged `_INVALID`)
and summarizes the model of record's value function and within-band slope from MOR_IDATA.
"""

import json
import os
from typing import Any, Dict, Optional

import arviz as az
import numpy as np

from cactus_paper.constants import DIMS, FULL_RAMP
from cactus_paper.paths import FITS, MOR_IDATA
from cactus_paper.stats import summarize

STAMP = "_generated_by"
GENERATED_BY = "cactus_paper.manuscript.fits"

# stats.json fits.<name> <- results/fits/<file>
CHOICE_FILES = [
    ("cv", "cv.json"),
    ("identifiability_slope", "identifiability_slope.json"),
    ("slope_controls", "slope_controls.json"),
    ("nested_2x2", "nested_2x2.json"),
    ("slope_controls_noise", "slope_controls_noise.json"),
    ("slope_controls_noise_K", "slope_controls_noise_K.json"),
    ("splithalf", "splithalf.json"),
    ("counterbalance", "counterbalance.json"),
    ("recovery", "recovery.json"),
    ("forward_ppc", "forward_ppc.json"),
]
VALUEMAP_FILES = [
    ("map", "value_map.json"),
    ("controls", "value_map_controls.json"),
    ("mor_fit", "mor_fit.json"),
    # rows differ in trial count: compare delta_per_1k_trials or z, not raw delta
    ("exclusion_sweep", "exclusion_sweep.json"),
]


def load(path: str, key: Optional[str] = None) -> Any:
    if not os.path.exists(path):
        print(f"  ! missing {os.path.relpath(path)} — skipped")
        return None
    d = json.load(open(path))
    return d if key is None else d.get(key)


def band_slopes() -> Optional[Dict[str, Any]]:
    """s_band from MOR_IDATA, as a fraction of a locally angle-linear map (FULL_RAMP)."""
    p = str(MOR_IDATA)
    if not os.path.exists(p):
        return None
    S = az.from_netcdf(p).posterior["s_band"].values.reshape(-1, 3)
    out = {
        "_full_ramp_benchmark": FULL_RAMP,
        "_note": (
            "s = 0 is a five-label staircase; s = 0.15 is a locally angle-linear map "
            "((30/50) x 0.25 — u is anchored at BAND CENTERS, so the benchmark is NOT 30/230)"
        ),
    }
    for j, d in enumerate(DIMS):
        v = S[:, j]
        lo, hi = az.hdi(v, hdi_prob=0.95)
        out[d] = {
            "mean": round(float(v.mean()), 4),
            "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "frac_full_ramp": round(float(v.mean() / FULL_RAMP), 3),
        }
    p_ = S.mean(1)
    lo, hi = az.hdi(p_, hdi_prob=0.95)
    out["pooled"] = {
        "mean": round(float(p_.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "frac_full_ramp": round(float(p_.mean() / FULL_RAMP), 3),
        "frac_full_ramp_lo": round(float(lo / FULL_RAMP), 3),
        "frac_full_ramp_hi": round(float(hi / FULL_RAMP), 3),
    }
    return out


def value_function() -> Optional[Dict[str, Any]]:
    """Fitted value-map increments from MOR_IDATA, plus shape contrasts and between-subject SD.

    Without a `frac` deterministic, the population profile is softmax([0, psi_mu]).
    """
    p = str(MOR_IDATA)
    if not os.path.exists(p):
        return None
    po = az.from_netcdf(p).posterior
    if "frac" in po:
        FR = po["frac"].values.reshape(-1, 3, 4)
    else:
        _p = po["psi_mu"].values.reshape(-1, 3, 3)
        _p = np.concatenate([np.zeros((_p.shape[0], 3, 1)), _p], axis=-1)
        FR = np.exp(_p) / np.exp(_p).sum(-1, keepdims=True)
    out = {"_taught_uniform": 0.25}
    for j, d in enumerate(DIMS):
        lo, hi = az.hdi(FR[:, j, :], hdi_prob=0.95).T
        out[d] = {
            "increments": [round(float(x), 3) for x in FR[:, j, :].mean(0)],
            "lo": [round(float(x), 3) for x in lo],
            "hi": [round(float(x), 3) for x in hi],
        }

    # per-attribute shape contrasts: edge-minus-interior, top-minus-bottom, max-minus-min
    # (direction-free), and each increment vs uniform
    def _c(e):
        return summarize(e, nd=3, direction="pdir")

    for j, d in enumerate(DIMS):
        F = FR[:, j, :]
        out[f"{d}_u_shape_edge_minus_interior"] = _c(
            F[:, 0] + F[:, 3] - F[:, 1] - F[:, 2]
        )
        out[f"{d}_top_minus_bottom"] = _c(F[:, 3] - F[:, 0])
        out[f"{d}_max_minus_min"] = _c(F.max(1) - F.min(1))
        for k in range(4):
            out[f"{d}_inc{k + 1}_vs_uniform"] = _c(F[:, k] - 0.25)

    # between-participant SD of the increments
    if all(k in po for k in ("psi_mu", "sigma_psi", "z_psi")):
        _pm = po["psi_mu"].values.reshape(-1, 3, 3)
        _sp = po["sigma_psi"].values.reshape(-1, 3, 3)
        _zp = po["z_psi"].values.reshape(-1, po["z_psi"].shape[2], 3, 3)
        _ps = _pm[:, None, :, :] + _sp[:, None, :, :] * _zp
        _ps = np.concatenate([np.zeros(_ps.shape[:3] + (1,)), _ps], axis=-1)
        _F = np.exp(_ps)
        _F /= _F.sum(-1, keepdims=True)  # (draw, subj, dim, 4)
        _sd = _F.std(1)  # (draw, dim, 4)
        _m = _sd.mean(-1)  # (draw, dim), mean over increments
        bs = {
            "_note": (
                "SD across participants of the fitted increments, per posterior draw. "
                "`mean_over_increments` is the per-attribute summary; the contrasts test "
                "whether color really is the heterogeneous attribute."
            )
        }
        for j, d in enumerate(DIMS):
            lo, hi = az.hdi(_sd[:, j, :], hdi_prob=0.95).T
            bs[d] = {
                "per_increment": [round(float(x), 3) for x in _sd[:, j, :].mean(0)],
                "lo": [round(float(x), 3) for x in lo],
                "hi": [round(float(x), 3) for x in hi],
                "mean_over_increments": _c(_m[:, j]),
            }
        bs["color_minus_mean_shape_line"] = _c(_m[:, 0] - _m[:, [1, 2]].mean(1))
        bs["color_minus_shape"] = _c(_m[:, 0] - _m[:, 1])
        bs["color_minus_line"] = _c(_m[:, 0] - _m[:, 2])
        bs["p_color_is_max"] = round(float((_m.argmax(1) == 0).mean()), 3)
        out["increment_between_subject_sd"] = bs
    return out


def fits_namespace() -> Dict[str, Any]:
    """The `fits` block of stats.json."""
    choice = {STAMP: GENERATED_BY}
    for name, fname in CHOICE_FILES:
        d = load(os.path.join(FITS, fname))
        if d is not None:
            choice[name] = d
    vf = value_function()
    if vf:
        choice["value_function"] = vf

    valuemap = {STAMP: GENERATED_BY}
    for name, fname in VALUEMAP_FILES:
        d = load(os.path.join(FITS, fname))
        if d is None:
            continue
        # drop entries flagged as unconverged
        if isinstance(d, dict):
            bad = [k for k, v in d.items() if isinstance(v, dict) and "_INVALID" in v]
            for k in bad:
                print(
                    f"  ! dropping {name}.{k} — flagged _INVALID ({d[k]['_INVALID'][:60]}...)"
                )
                d.pop(k)
        valuemap[name] = d
    bs = band_slopes()
    if bs:
        valuemap["within_band_slope"] = bs

    fits = {**choice, **valuemap}
    print(f"  fits/: {sorted(k for k in fits if k != STAMP)}")
    return fits
