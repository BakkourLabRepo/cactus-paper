"""Cross-stream (Fig 5): do the choice-model constructs predict information search?

Person-level: Payne index (search style) and mean sampling depth (search amount) regressed jointly on
z-scored lambda, gain and concentration from the model of record, which never saw the mouse stream.

  python -m cactus_paper.analyses.crossstream
    -> results/fits/fig5_grid.json        summaries (stats.json["fig5_grid"])
       results/fits/fig5_grid_draws.npz   coefficient / contrast draws (fig_ms5_crossstream C, E)
"""

import json
import os
from typing import Any, Dict, Tuple

import arviz as az
import numpy as np
import pandas as pd

from cactus_paper.features.traits import concentration as wl_conc
from cactus_paper.features.traits import gain as wl_gain
from cactus_paper.paths import DATA, FITS, MOR_IDATA

GRID_JSON = FITS / "fig5_grid.json"
GRID_DRAWS = FITS / "fig5_grid_draws.npz"
FIT = dict(
    draws=2000, tune=2000, chains=4, inference_method="nutpie", random_seed=20260821
)
PREDICTORS = ["lam", "gain", "conc"]
FORMULAS = {"style": "payne ~ lam + gain + conc", "amount": "depth ~ lam + gain + conc"}
# per model: the focal construct against each of the other two, |coef a| - |coef b| per draw
CONTRASTS = {
    "style": [("lam_minus_conc", "lam", "conc"), ("lam_minus_gain", "lam", "gain")],
    "amount": [("conc_minus_lam", "conc", "lam"), ("conc_minus_gain", "conc", "gain")],
}
# `contrast` key read downstream: the focal construct against gain
LEGACY = {
    "style": ("lam_minus_gain", "|lam| - |gain|"),
    "amount": ("conc_minus_gain", "|conc| - |gain|"),
}


def _z(v):
    return (np.asarray(v, float) - np.nanmean(v)) / np.nanstd(v)


def crossstream_frame(D: Dict[str, Any]) -> pd.DataFrame:
    """Person-level frame: payne, z(depth), z(lam), z(gain), z(conc); rows with any NaN dropped.

    Depth counts every visit token, 'none' included (the fig_ms5 convention).
    """
    nS, cd = D["nS"], D["cd"]
    su = cd.run_id.drop_duplicates().tolist()
    po = az.from_netcdf(MOR_IDATA).posterior
    absW = np.abs(po["W"].values.reshape(-1, nS, 3).mean(0))
    lam = po["lam_s"].values.reshape(-1, nS).mean(0)
    gain = wl_gain(
        absW
    )  # mean|W_s| on the raw scale (the model's `wgain` deterministic)
    conc = wl_conc(absW)  # Gini of |w|

    dc = pd.read_csv(DATA / "dim_counts.csv")
    dc = dc[dc.run_id.isin(su)]
    payne = dc.groupby("run_id").payne_index.mean().loc[su].values

    vs = pd.read_csv(DATA / "visit_sequences.csv")
    vs = vs[vs.run_id.isin(su)].dropna(subset=["visit_sequence"])
    vs["depth"] = vs.visit_sequence.astype(str).str.split("-").map(len)
    mean_depth = vs.groupby("run_id").depth.mean().reindex(su).values

    return pd.DataFrame(
        dict(
            payne=payne, depth=_z(mean_depth), lam=_z(lam), gain=_z(gain), conc=_z(conc)
        )
    ).dropna()


def _sm(d):
    lo, hi = az.hdi(np.asarray(d), hdi_prob=0.95)
    return {
        "mean": round(float(d.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "pdir": round(float(max((d > 0).mean(), (d < 0).mean())), 3),
    }


def fit_grid(sub: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Fit the two joint models. Returns (summary dict for fig5_grid.json, draws dict)."""
    import bambi as bmb

    draws = {}
    for key, formula in FORMULAS.items():
        m = bmb.Model(formula, sub, family="gaussian").fit(**FIT)
        for n in PREDICTORS:
            draws[f"{key}_{n}"] = m.posterior[n].values.ravel()
        for cname, a, b in CONTRASTS[key]:
            draws[f"{key}_{cname}"] = np.abs(draws[f"{key}_{a}"]) - np.abs(
                draws[f"{key}_{b}"]
            )

    grid = {
        "_formulas": list(FORMULAS.values()),
        "_note": "the 3-predictor joint models PLOTTED in fig_ms5_crossstream panel B; the "
        "2-predictor fig5_style / fig5_amount blocks in stats.json are different models",
        "_conc_metric": "gini of |w_sd|",
    }
    for key in FORMULAS:
        grid[key] = {n: _sm(draws[f"{key}_{n}"]) for n in PREDICTORS}
        for cname, _a, _b in CONTRASTS[key]:
            grid[key][cname] = _sm(draws[f"{key}_{cname}"])
        legacy, what = LEGACY[key]
        grid[key]["contrast"] = dict(grid[key][legacy], _what=what)
    return grid, draws


if __name__ == "__main__":
    from cactus_paper.design import load_design

    grid, draws = fit_grid(crossstream_frame(load_design()))
    os.makedirs(FITS, exist_ok=True)
    with open(GRID_JSON, "w") as f:
        json.dump(grid, f, indent=2)
    np.savez(GRID_DRAWS, **draws)
    print(f"saved -> {GRID_JSON}\n         {GRID_DRAWS}")
    print("\nthe grid (joint models, z-scored predictors; * = 95% HDI excludes zero)")
    for key in FORMULAS:
        print(f"  {key.upper():7s} {FORMULAS[key]}")
        for n in PREDICTORS + [c[0] for c in CONTRASTS[key]]:
            v = grid[key][n]
            star = "" if v["lo"] < 0 < v["hi"] else "  *"
            print(
                f"    {n:16s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  "
                f"P(dir)={v['pdir']:.3f}{star}"
            )
