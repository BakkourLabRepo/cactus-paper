"""Full-data fits of two comparison models, and the convergence table (Table S1).

  full               participant_map_lambda / group_map_noise full fits -> results/fits/<model>_idata.nc
  convergence-table  R-hat / ESS / divergences for every fit of record -> results/fits/fits_table.json

Run from repo root: pixi run python -m cactus_paper.models.fit {full,convergence-table} [args]
"""

import argparse
import functools
import json
import os
import sys
from typing import Any, Dict

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from cactus_paper.design import load_design
from cactus_paper.features.values import raw_values_and_diffs, taught_levels
from cactus_paper.models.comparison import per_participant_map, value_noise
from cactus_paper.paths import FITS, MOR_IDATA

OUT = str(FITS)


# model, seed, sampled sites for R-hat, output
FULL_FITS = {
    "hier": (
        functools.partial(per_participant_map, use_lam=True),
        707,
        ["a0", "sda", "w_mu", "w_sd", "psi_mu", "sigma_psi", "c0", "sigma_lam"],
        "participant_map_lambda_idata.nc",
    ),
    "noise": (
        value_noise,
        301,
        ["a0", "sda", "w_mu", "w_sd", "g0", "sigma_g"],
        "group_map_noise_idata.nc",
    ),
}


def fit_full() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(FULL_FITS))
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    mdl, seed, names, fname = FULL_FITS[a.model]
    D = load_design()
    nS = D["nS"]
    Vl, Vr, _ = raw_values_and_diffs(D["cd"])
    kl, kr = taught_levels(Vl, Vr)
    m = MCMC(
        NUTS(mdl, target_accept_prob=0.92),
        num_warmup=a.warmup,
        num_samples=a.samples,
        num_chains=4,
        progress_bar=False,
    )
    m.run(
        jax.random.PRNGKey(seed),
        jnp.array(D["sc"]),
        jnp.array(kl),
        jnp.array(kr),
        nS,
        y=jnp.array(np.asarray(D["Y"]).astype(int)),
    )
    ida = az.from_numpyro(m)
    os.makedirs(a.out, exist_ok=True)
    ida.to_netcdf(f"{a.out}/{fname}")
    print(
        f"max R-hat {float(az.rhat(ida, var_names=names).to_array().max()):.4f} | "
        f"divergences {int(ida.sample_stats['diverging'].values.sum())}"
    )
    print(f"saved -> {a.out}/{fname}")


TABLE_OUT = f"{FITS}/fits_table.json"


# derived, not sampled -- excluded from R-hat/ESS
DERIVED = {"W", "lam_s", "frac_s", "a", "u", "u_eff", "sig", "p"}


CONVERGENCE_FITS = [
    # model of record, from cactus_paper.paths
    (
        "mor",
        str(MOR_IDATA),
        "fit of record: per-participant value map + within-band slope + magnitude reliance",
    ),
    (
        "participant_map_lambda",
        f"{FITS}/participant_map_lambda_idata.nc",
        "per-participant value map + magnitude reliance, no within-band slope",
    ),
    (
        "group_map_lambda",
        f"{FITS}/group_map_lambda_idata.nc",
        "group value map + magnitude reliance",
    ),
    (
        "group_map_noise",
        f"{FITS}/group_map_noise_idata.nc",
        "value noise in place of magnitude reliance",
    ),
    (
        "group_map_lambda_device",
        f"{FITS}/group_map_lambda_device_idata.nc",
        "group_map_lambda with pointer device as a covariate on the weights and magnitude reliance",
    ),
    (
        "taught_spacing_lambda",
        f"{FITS}/taught_spacing_lambda_idata.nc",
        "magnitude reliance over the taught spacing (value map pinned)",
    ),
]


def row(label: str, path: str, desc: str) -> Dict[str, Any]:
    d = az.from_netcdf(path)
    post = d.posterior
    free = [v for v in post.data_vars if v not in DERIVED]
    sub = post[free]
    rhat = az.rhat(sub)
    ess = az.ess(sub, method="bulk")
    # structurally-constant entries have undefined (NaN) R-hat; drop them
    _r = {v: np.asarray(rhat[v].values).ravel() for v in free}
    _r = {v: x[np.isfinite(x)] for v, x in _r.items()}
    const = sorted(v for v, x in _r.items() if x.size == 0)
    _r = {v: x for v, x in _r.items() if x.size}
    _e = {v: np.asarray(ess[v].values).ravel() for v in _r}
    _e = {v: x[np.isfinite(x)] for v, x in _e.items()}
    worst = max(_r, key=lambda v: _r[v].max())
    max_rhat = float(_r[worst].max())
    min_ess = float(min(x.min() for x in _e.values() if x.size))
    n_par = int(sum(x.size for x in _r.values()))
    div = (
        int(d.sample_stats["diverging"].values.sum())
        if "sample_stats" in d.groups()
        else None
    )
    return {
        "label": label,
        "desc": desc,
        "chains": int(post.sizes["chain"]),
        "draws": int(post.sizes["draw"]),
        "n_par": n_par,
        "max_rhat": round(max_rhat, 4),
        "max_rhat_param": worst,
        "min_ess_bulk": int(min_ess),
        "n_divergences": div,
        "_constant_params": const,
    }


def convergence_table() -> None:
    rows = []
    for label, path, desc in CONVERGENCE_FITS:
        r = row(label, path, desc)
        rows.append(r)
        print(
            f"  {label:20s} n_par {r['n_par']:5d}  Rhat {r['max_rhat']:.4f} "
            f"({r['max_rhat_param']})  ESS {r['min_ess_bulk']:6d}  div {r['n_divergences']}"
        )
    json.dump(
        {"_nominal": "4 chains x 1500 warmup / 1500 draws", "rows": rows},
        open(TABLE_OUT, "w"),
        indent=2,
    )
    print(f"-> {TABLE_OUT}")


COMMANDS = {"full": fit_full, "convergence-table": convergence_table}

if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.models.fit {full|convergence-table} [args]"
        )
    _cmd = sys.argv.pop(1)
    COMMANDS[_cmd]()
