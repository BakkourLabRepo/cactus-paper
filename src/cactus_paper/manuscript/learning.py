"""stats.json: the learning task (Fig 1) and Payne descriptives (Fig 2)."""

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from cactus_paper.constants import DIMS
from cactus_paper.manuscript.context import Context, bayes_corr, summ
from cactus_paper.paths import DATA, FITS


def learning(ctx: Context) -> Dict[str, Any]:
    import bambi as bmb

    print("[learning] per-attribute accuracy + hierarchical contrasts ...", flush=True)
    ld = pd.read_csv(f"{DATA}/learning_data.csv")
    ld = ld[ld.run_id.isin(ctx.su)].dropna(subset=["correct"]).copy()
    ld["correct"] = ld["correct"].astype(int)
    acc_by = ld.groupby(["run_id", "dim"]).correct.mean().unstack()
    out = {
        "learning_acc": {
            d: {
                "mean": round(float(acc_by[d].mean()), 3),
                "sd": round(float(acc_by[d].std()), 3),
            }
            for d in DIMS
        }
    }
    m_acc = bmb.Model("correct ~ 0 + dim + (1 | run_id)", ld, family="bernoulli").fit(
        **ctx.FIT
    )
    dimlv = list(m_acc.posterior["dim"].coords["dim_dim"].values)
    dv = {d: m_acc.posterior["dim"].values[..., dimlv.index(d)] for d in DIMS}
    out["learning_contrast"] = {
        "shape_vs_line": summ(dv["shape"] - dv["line"]),
        "shape_vs_color": summ(dv["shape"] - dv["color"]),
        "color_vs_line": summ(dv["color"] - dv["line"]),
    }

    # Within-person spread of learning across attributes (moderator of Fig 3C).
    _srtk = np.sort(ctx.dacc, 1)[:, ::-1]
    out["learning_spread"] = {
        "mean": round(float(np.nanmean(ctx.spread)), 3),
        "sd": round(float(np.nanstd(ctx.spread)), 3),
        "by_rank": [round(float(np.nanmean(_srtk[:, i])), 3) for i in range(3)],
        "corr_with_K": bayes_corr(ctx.K, ctx.spread),
    }

    # DeltaValue slope (Fig 1), written by cactus_paper.features.learning_slope --cells.
    _lslope_path = f"{FITS}/learning_slope.json"
    if os.path.exists(_lslope_path):
        out["learning_slope"] = json.load(open(_lslope_path))
    else:
        print(
            f"  ! {_lslope_path} missing -- run `python -m cactus_paper.features.learning_slope --cells`; "
            "'learning_slope' key skipped"
        )
    # robustness: the same fit without the prompt-reversal trials (learning_slope --cells --exclude-rev 1)
    _norev_path = f"{FITS}/learning_slope_norev1.json"
    if os.path.exists(_norev_path):
        out["learning_slope_norev1"] = json.load(open(_norev_path))
    return out


def payne(ctx: Context) -> Dict[str, Any]:
    p = ctx.payne
    return {
        "payne": {
            "mean": round(float(p.mean()), 3),
            "sd": round(float(p.std()), 3),
            "min": round(float(p.min()), 3),
            "max": round(float(p.max()), 3),
            "frac_positive": round(float((p > 0).mean()), 3),
        }
    }
