"""stats.json: cross-stream (Fig 5) — the choice-model constructs against information search.

The 2-predictor fig5_style / fig5_amount fits here are different models from the 3-predictor grid
fig_ms5_crossstream plots (results/fits/fig5_grid.json, cactus_paper.analyses.crossstream).
"""

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from cactus_paper.constants import DIMS
from cactus_paper.features.sampling import trial_visit_feats
from cactus_paper.manuscript.context import Context, bfit, summ, zc1
from cactus_paper.paths import DATA, FITS


def prop_r(latent_draws: np.ndarray, yobs: np.ndarray) -> Dict[str, Any]:
    """Per-draw correlations, so the couplings carry the latents' estimation uncertainty."""
    ok = np.isfinite(yobs)
    return summ(
        [
            np.corrcoef(latent_draws[d][ok], yobs[ok])[0, 1]
            for d in range(latent_draws.shape[0])
        ]
    )


def fig5(ctx: Context) -> Dict[str, Any]:
    import bambi as bmb

    print("[fig5] cross-stream 2x2 + allocation ...", flush=True)
    su, nS, FIT = ctx.su, ctx.nS, ctx.FIT
    vs = pd.read_csv(f"{DATA}/visit_sequences.csv")
    vs = vs[vs.run_id.isin(su)].dropna(subset=["visit_sequence"])
    vs["depth"] = vs.visit_sequence.astype(str).str.split("-").map(len)
    mean_depth = vs.groupby("run_id").depth.mean().reindex(su).values
    tf = pd.DataFrame([trial_visit_feats(s) for s in vs.visit_sequence])
    tf["run_id"] = vs.run_id.values
    _agg = (
        tf.groupby("run_id")[[f"v_{d}" for d in DIMS] + [f"aw_{d}" for d in DIMS]]
        .sum()
        .reindex(su)
    )
    _vm = _agg[[f"v_{d}" for d in DIMS]].values.astype(float)
    _awm = _agg[[f"aw_{d}" for d in DIMS]].values.astype(float)
    vshare = _vm / _vm.sum(1, keepdims=True)
    awshare = _awm / np.maximum(_awm.sum(1, keepdims=True), 1)
    _first = np.zeros((nS, 3))
    for i, r in enumerate(su):
        g = tf[tf.run_id == r]
        for j, d in enumerate(DIMS):
            _first[i, j] = (g.first_dim.values == d).mean()

    out = {}
    sub_df = pd.DataFrame(
        dict(
            payne=ctx.payne,
            depth=zc1(mean_depth),
            lam=zc1(ctx.lam_s),
            gain=zc1(ctx.gain),
            conc=zc1(ctx.conc),
            K=zc1(ctx.K),
        )
    ).dropna()
    m_style = bmb.Model("payne ~ lam + gain", sub_df, family="gaussian").fit(**FIT)
    m_amount = bmb.Model("depth ~ lam + gain", sub_df, family="gaussian").fit(**FIT)
    _dr = lambda m, n: m.posterior[n].values.ravel()
    out["fig5_style"] = {n: summ(_dr(m_style, n)) for n in ("lam", "gain")}
    out["fig5_amount"] = {n: summ(_dr(m_amount, n)) for n in ("lam", "gain")}
    # within-column |coefficient| contrasts (outcome scales differ across columns)
    out["fig5_dissociation"] = {
        "style_lam_minus_gain": summ(
            np.abs(_dr(m_style, "lam")) - np.abs(_dr(m_style, "gain"))
        ),
        "amount_gain_minus_lam": summ(
            np.abs(_dr(m_amount, "gain")) - np.abs(_dr(m_amount, "lam"))
        ),
    }
    out["fig5_style_adjK"] = bfit(
        "payne ~ lam + gain + K", sub_df, "gaussian", FIT, ["lam", "K"]
    )
    out["fig5_amount_adjK"] = bfit(
        "depth ~ lam + gain + K", sub_df, "gaussian", FIT, ["gain", "K"]
    )
    # depth on gain and concentration jointly
    out["fig5_amount_conc"] = bfit(
        "depth ~ gain + conc", sub_df, "gaussian", FIT, ["gain", "conc"]
    )
    out["fig5_gain_conc_corr"] = round(float(np.corrcoef(ctx.gain, ctx.conc)[0, 1]), 3)
    # 3-predictor grid plotted in fig_ms5_crossstream
    _grid_path = f"{FITS}/fig5_grid.json"
    if os.path.exists(_grid_path):
        out["fig5_grid"] = json.load(open(_grid_path))
    else:
        print(
            f"  ! {_grid_path} missing -- run `python -m cactus_paper.analyses.crossstream`; "
            "'fig5_grid' key skipped"
        )
    # propagated over per-draw gain and conc
    out["fig5_gain_conc_corr_propagated"] = summ(
        [
            np.corrcoef(ctx.gainD[d], ctx.concD[d])[0, 1]
            for d in range(ctx.gainD.shape[0])
        ]
    )

    # allocation, at subject x attribute. Same double-centered estimator as the weight<->knowledge
    # correlation: attribute identity and participant both held fixed.
    absW, dacc = ctx.absW, ctx.dacc
    LA = pd.DataFrame(
        [
            dict(
                run_id=r,
                dim=d,
                W=absW[i, j],
                dacc=dacc[i, j],
                vshare=vshare[i, j],
                awshare=awshare[i, j],
                first=_first[i, j],
            )
            for i, r in enumerate(su)
            for j, d in enumerate(DIMS)
        ]
    )
    for nm, tgt in [("alloc", "vshare"), ("attrwise", "awshare"), ("first", "first")]:
        out[f"fig5_{nm}"] = bfit(
            f"{tgt} ~ W + C(dim) + (1|run_id)", LA, "gaussian", FIT, ["W"]
        )
        out[f"fig5_{nm}_adjK"] = bfit(
            f"{tgt} ~ W + dacc + C(dim) + (1|run_id)",
            LA,
            "gaussian",
            FIT,
            ["W", "dacc"],
        )

    out["fig5_propr_style"] = prop_r(ctx.lamD, ctx.payne)
    out["fig5_propr_amount"] = prop_r(ctx.gainD, mean_depth)
    return out
