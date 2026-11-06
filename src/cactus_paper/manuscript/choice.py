"""stats.json: choice psychometrics, the poles and the divergent-trial ladder (Fig 3), readout profiles."""

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from cactus_paper.manuscript.context import Context, bayes_corr, bfit, summ
from cactus_paper.paths import FITS


def fig3(ctx: Context) -> Dict[str, Any]:
    sc, nS, Y, K, div = ctx.sc, ctx.nS, ctx.Y, ctx.K, ctx.div
    follow_mag, conflict, lamD = ctx.follow_mag, ctx.conflict, ctx.lamD
    out = {}

    # ------------------------------------------------------------ psychometrics (Fig 3A/B)
    print("[fig3] continuous-K moderation on OBJECTIVE magnitude ...", flush=True)
    dfG = pd.DataFrame(
        {"chose_right": Y, "Vc": ctx.Vmag, "K": K[sc], "subject": sc.astype(str)}
    )
    out["fig3_gain"] = bfit(
        "chose_right ~ Vc * K + (Vc | subject)",
        dfG,
        "bernoulli",
        ctx.FIT,
        ["Vc", "K", "Vc:K"],
    )
    _cav = np.abs(ctx.Vmag)[conflict]
    dfA = pd.DataFrame(
        {
            "correct": follow_mag[conflict],
            "cAbsVc": _cav - _cav.mean(),
            "K": K[sc][conflict],
            "subject": sc[conflict].astype(str),
        }
    )
    out["fig3_acc"] = bfit(
        "correct ~ cAbsVc * K + (cAbsVc | subject)",
        dfA,
        "bernoulli",
        ctx.FIT,
        ["cAbsVc", "K", "cAbsVc:K"],
    )

    _incacc = np.array([follow_mag[div & (sc == s)].mean() for s in range(nS)])
    _conacc = np.array([follow_mag[~div & (sc == s)].mean() for s in range(nS)])
    out["fig3_poles_desc"] = {
        "congruent_mean": round(float(_conacc.mean()), 3),
        "congruent_sd": round(float(_conacc.std()), 3),
        "incongruent_mean": round(float(_incacc.mean()), 3),
        "incongruent_sd": round(float(_incacc.std()), 3),
        "corr_incacc_lam": round(float(np.corrcoef(_incacc, ctx.lam_s)[0, 1]), 3),
        # propagated over lam's draws (_incacc is observed)
        "corr_incacc_lam_propagated": summ(
            [np.corrcoef(_incacc, lamD[d])[0, 1] for d in range(lamD.shape[0])]
        ),
    }

    # ------------------------------------------------------------ the ladder (Fig 3C)
    # On divergent trials EXACTLY ONE attribute sides with the summed-magnitude winner (its |delta|
    # exceeds the other two combined), so the three columns are disjoint and sum to the divergent count.
    print("[fig3C] divergent-trial ladder + trait-only PPC ...", flush=True)
    dacc, VDo, agg = ctx.dacc, ctx.VDo, ctx.agg
    krank = np.argsort(np.argsort(-np.where(np.isnan(dacc), -np.inf, dacc), 1), 1)
    crit = np.argmax(np.abs(VDo), 1)
    critrank = krank[sc, crit]
    assert (
        ((np.sign(VDo) == agg[:, None]) & (np.sign(VDo) != 0))[div].sum(1) == 1
    ).all()

    _cols, _n = [], []
    for r in (2, 1, 0):  # worst, middle, best
        a = np.full(nS, np.nan)
        pool = div & (critrank == r)
        for s in range(nS):
            m = pool & (sc == s)
            if m.sum() >= 5:
                a[s] = follow_mag[m].mean()
        _cols.append(a)
        _n.append(int(pool.sum()))
    _okC = ~np.isnan(np.column_stack(_cols)).any(1)
    _c = [a[_okC] for a in _cols]

    # In-sample PPC of the ladder from the model of record.
    po, WD = ctx.po, ctx.WD
    _pool = [[div & (critrank == r) & (sc == s) for s in range(nS)] for r in (2, 1, 0)]
    _use = [[m for m in row if m.sum() >= 5] for row in _pool]
    _rng = np.random.default_rng(2026)
    _a0D, _sdaD, _aD = (
        po[n].values.reshape(-1, *po[n].values.shape[2:]) for n in ("a0", "sda", "a")
    )
    _sel = _rng.choice(_a0D.shape[0], 30 if ctx.quick else 300, replace=False)
    _ppc = np.zeros((len(_sel), 3))
    for q, d_ in enumerate(_sel):
        Wt, lm = WD[d_][sc], lamD[d_][sc]
        eta = (
            _a0D[d_]
            + _aD[d_][sc] * _sdaD[d_]
            + (1 - lm) * (Wt * ctx.Sg).sum(-1)
            + lm * (Wt * ctx.M).sum(-1)
        )
        ys = (_rng.random(len(Y)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        fm = (((ys == 1) & (agg > 0)) | ((ys == 0) & (agg < 0))).astype(float)
        _ppc[q] = [np.mean([fm[m].mean() for m in row]) for row in _use]

    out["fig3_ladder"] = {
        "worst": {"mean": round(float(_c[0].mean()), 3), "n_trials": _n[0]},
        "middle": {"mean": round(float(_c[1].mean()), 3), "n_trials": _n[1]},
        "best": {"mean": round(float(_c[2].mean()), 3), "n_trials": _n[2]},
        "n_subj": int(_okC.sum()),
        "ppc": {
            nm: {
                "mean": round(float(col.mean()), 3),
                "lo": round(float(np.percentile(col, 2.5)), 3),
                "hi": round(float(np.percentile(col, 97.5)), 3),
            }
            for nm, col in zip(("worst", "middle", "best"), _ppc.T)
        },
        "slope_obs": round(float(_c[2].mean() - _c[0].mean()), 3),
        "slope_ppc": {
            "mean": round(float((_ppc[:, 2] - _ppc[:, 0]).mean()), 3),
            "lo": round(float(np.percentile(_ppc[:, 2] - _ppc[:, 0], 2.5)), 3),
            "hi": round(float(np.percentile(_ppc[:, 2] - _ppc[:, 0], 97.5)), 3),
        },
    }
    # ladder moderators (observed quantities -> bayes_corr)
    _sl = _c[2] - _c[0]
    out["fig3_ladder_moderation"] = {
        "r_spread": bayes_corr(ctx.spread[_okC], _sl),
        "r_K": bayes_corr(K[_okC], _sl),
        "r_absposition": bayes_corr(np.abs(_incacc[_okC] - 0.5), np.abs(_sl)),
    }
    return out


# stats.json key <- results/fits/<file>, written by `python -m cactus_paper.analyses.contribution <cmd>`
READOUT_FILES = [
    ("readout_profile", "readout"),
    ("slope_readout_profile", "slope-readout"),
]


def readout_profiles(ctx: Context) -> Dict[str, Any]:
    out = {}
    for key, cmd in READOUT_FILES:
        path = f"{FITS}/{key}.json"
        if os.path.exists(path):
            out[key] = json.load(open(path))
        else:
            print(
                f"  ! {path} missing -- run `python -m cactus_paper.analyses.contribution {cmd}`; "
                f"'{key}' key skipped"
            )
    return out
