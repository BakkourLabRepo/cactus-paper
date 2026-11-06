"""stats.json: model-of-record summaries, per-subject constructs, strategy plane, design counts."""

import os
from typing import Any, Dict

import numpy as np

from cactus_paper.constants import DIMS
from cactus_paper.features.traits import minmax_ratio
from cactus_paper.manuscript.context import Context, summ
from cactus_paper.paths import ROOT

# Gain is not a parameter: it is mean|W_s| (the model's `wgain` deterministic), on the raw scale.
PARAMS = ["w_sd", "c0", "sigma_lam", "a0", "sda"]


def header(ctx: Context) -> Dict[str, Any]:
    po = ctx.po
    out = {
        "_source": os.path.relpath(ctx.idata, ROOT),
        "_interval": "95% HDI (az.hdi)",
        "_model": "model of record (cactus_paper.models.mor): per-participant monotone "
        "value function u_sd (endpoints pinned) with a within-band slope x "
        "per-attribute weights x lambda. lambda is MAGNITUDE RELIANCE "
        "(metric-vs-ordinal comparison), NOT resolution -- quantization, compression "
        "and imprecision were each fit and each lost",
        "_conc_metric": "gini of |w_sd|",
    }
    for n in PARAMS:
        out[n] = summ(po[n].values)
    _wm = po["w_mu"].values.reshape(-1, 3)
    out["w_mu"] = {d: summ(_wm[:, i]) for i, d in enumerate(DIMS)}
    return out


def constructs(ctx: Context) -> Dict[str, Any]:
    """lambda vs gain / conc, and the strategy-plane descriptives (Fig 4B).

    Unsuffixed = per-draw (propagated, for the text with HDI); `*_pointest` = posterior means (what
    the figures plot and annotate).
    """
    lamD, lam_s, gain, conc = ctx.lamD, ctx.lam_s, ctx.gain, ctx.conc
    out = {
        "corr_lam_gain": summ(
            [np.corrcoef(lamD[d], ctx.gainD[d])[0, 1] for d in range(lamD.shape[0])]
        ),
        "corr_lam_conc": summ(
            [np.corrcoef(lamD[d], ctx.concD[d])[0, 1] for d in range(lamD.shape[0])]
        ),
        "corr_lam_gain_pointest": round(float(np.corrcoef(lam_s, gain)[0, 1]), 3),
        "corr_lam_conc_pointest": round(float(np.corrcoef(lam_s, conc)[0, 1]), 3),
    }
    _ratio = minmax_ratio(ctx.absW)  # descriptive only; not the conc metric
    _srt = np.sort(ctx.absW, 1)[:, ::-1]
    out["strategy_plane"] = {
        "lam": {
            "mean": round(float(lam_s.mean()), 3),
            "sd": round(float(lam_s.std()), 3),
            "min": round(float(lam_s.min()), 3),
            "max": round(float(lam_s.max()), 3),
        },
        "conc": {
            "mean": round(float(conc.mean()), 3),
            "sd": round(float(conc.std()), 3),
            "min": round(float(conc.min()), 3),
            "max": round(float(conc.max()), 3),
        },
        "minmax_ratio_median": round(float(np.median(_ratio)), 3),
        "n_effectively_one_cue": int((_srt[:, 1] < _srt[:, 0] / 3).sum()),
        "n_negative_weight_subjects": int((ctx.W < 0).any(1).sum()),
    }
    return out


def design(ctx: Context) -> Dict[str, Any]:
    return {
        "design": {
            "n_trials": int(len(ctx.Y)),
            "n_subjects": int(ctx.nS),
            "n_conflict": int(ctx.conflict.sum()),
            "n_divergent_design": int(ctx.div.sum()),
            "corr_vmag_vtal": round(float(np.corrcoef(ctx.Vmag, ctx.Vtal)[0, 1]), 3),
        }
    }
