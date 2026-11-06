"""Nested antecedent 2x2: {K_s, DeltaValue slope} x {gain, lambda}, every cell inside the MoR.

  gain term   W_sd = exp(g . x_s) * (w_mu_d + w_sd * z_sd)
              one multiplicative scale per participant on all three weights, so g is the change in
              log gain per SD of the driver; g = 0 is exactly the model of record
  lambda term lambda_s = sigmoid(c0 + c . x_s + sigma_lam * z_s)

Runs (each its own process, so they can go in parallel):
  gain_K        gain <- K
  gain_slope    gain <- slope
  lambda_K      lambda <- K
  lambda_slope  lambda <- slope
  joint         gain <- K + slope, lambda <- K + slope; all four cells in one fit, each net of the
                other driver
Controls on the lambda <- slope cell (same model, a different driver vector):
  lambda_slope_resid_K, lambda_perm1, lambda_perm2, lambda_slope_resid_noise,
  lambda_slope_resid_noiseK
Drivers are z-scored. The slope is map_slope_draws.npy's posterior mean (run
`python -m cactus_paper.features.learning_slope` first). --collect also writes slope_controls.json and slope_controls_noise{,K}.json.

Run:  pixi run antecedents <run>      (one process per run; launch them in parallel)
      pixi run antecedents --collect
The pixi task pins BLAS/XLA to one thread per process; a bare `python -m` does NOT.
Outputs -> results/fits/nested_2x2_<run>.json, merged into results/fits/nested_2x2.json
"""

import json
import os
import sys
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import DIMS
from cactus_paper.design import load_design
from cactus_paper.features.geometry import choice_frame
from cactus_paper.features.values import (
    mean_log_read_noise,
    raw_values_and_diffs,
    taught_levels,
)
from cactus_paper.models.mor import mor
from cactus_paper.paths import FITS

OUT = str(FITS)
RUNS = {
    "gain_K": (["K"], []),
    "gain_slope": (["slope"], []),  # (gain drivers, lambda drivers)
    "lambda_K": ([], ["K"]),
    "lambda_slope": ([], ["slope"]),
    "joint": (["K", "slope"], ["K", "slope"]),
}
CONTROLS = [
    "slope_resid_K",
    "perm1",
    "perm2",
    "slope_resid_noise",
    "slope_resid_noiseK",
]
RUNS.update({f"lambda_{c}": ([], [c]) for c in CONTROLS})


def drivers(D: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float]:
    """Every driver, z-scored, plus corr(slope, K)."""
    nS = D["nS"]
    K = np.asarray(D["K"], float)
    slope = np.load(f"{FITS}/map_slope_draws.npy").mean(0)
    assert slope.shape == (nS,)

    def resid(*cols):
        X = np.column_stack([np.ones(nS), *cols])
        return slope - X @ np.linalg.lstsq(X, slope, rcond=None)[0]

    v = {
        "K": K,
        "slope": slope,
        "slope_resid_K": resid(K),
        "perm1": np.random.default_rng(1).permutation(slope),
        "perm2": np.random.default_rng(2).permutation(slope),
    }
    NZ = mean_log_read_noise(D)
    v["slope_resid_noise"], v["slope_resid_noiseK"] = resid(NZ), resid(NZ, K)
    zc = lambda x: (x - x.mean()) / x.std()
    return {k: zc(x) for k, x in v.items()}, float(np.corrcoef(slope, K)[0, 1])


def summ(v: np.ndarray) -> Dict[str, float]:
    """v is (chain, draw). Mean, 95% HDI, P(>0), plus Monte Carlo error of the mean and of the
    2.5% / 97.5% quantiles, and bulk ESS."""
    flat = v.ravel()
    lo, hi = az.hdi(flat, hdi_prob=0.95)
    ds = az.convert_to_dataset({"x": v})
    one = lambda r: float(r["x"].values)
    return {
        "mean": float(flat.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_pos": float((flat > 0).mean()),
        "mcse_mean": one(az.mcse(ds)),
        "mcse_q025": one(az.mcse(ds, method="quantile", prob=0.025)),
        "mcse_q975": one(az.mcse(ds, method="quantile", prob=0.975)),
        "ess_bulk": one(az.ess(ds)),
    }


def run(name: str) -> None:
    gd, ldv = RUNS[name]
    D = load_design()
    nS = D["nS"]
    drv, r_slope_K = drivers(D)
    X = lambda names: jnp.array(
        np.column_stack([drv[k] for k in names]) if names else np.zeros((nS, 0))
    )
    Vl, Vr, _ = raw_values_and_diffs(D["cd"])
    _kl, _kr = taught_levels(Vl, Vr)
    rec, _ = choice_frame(D)
    pl = jnp.array(np.stack([rec[d]["pL"] for d in DIMS], 1))
    pr = jnp.array(np.stack([rec[d]["pR"] for d in DIMS], 1))
    m = MCMC(
        NUTS(mor, target_accept_prob=0.92),
        num_warmup=1500,
        num_samples=1500,
        num_chains=4,
        progress_bar=False,
    )
    m.run(
        jax.random.PRNGKey(401),
        jnp.array(D["sc"]),
        jnp.array(_kl),
        jnp.array(_kr),
        pl,
        pr,
        nS,
        gain_x=X(gd),
        lam_x=X(ldv),
        y=jnp.array(np.asarray(D["Y"]).astype(int)),
    )
    ida = az.from_numpyro(m)
    po = ida.posterior
    out = {"run": name, "gain_drivers": gd, "lambda_drivers": ldv}
    for j, k in enumerate(gd):
        out[f"gain_{k}"] = summ(po["g"].values[..., j])
    for j, k in enumerate(ldv):
        out[f"lambda_{k}"] = summ(po["c"].values[..., j])
    vn = ["a0", "sda", "w_mu", "w_sd"] + (["g"] if gd else []) + (["c"] if ldv else [])
    out["max_rhat"] = float(az.rhat(ida, var_names=vn).to_array().max())
    out["n_div"] = int(ida.sample_stats["diverging"].values.sum())
    out["r_slope_K"] = r_slope_K
    out["sigma_lam"] = float(po["sigma_lam"].values.mean())
    print(json.dumps(out), flush=True)
    json.dump(out, open(f"{OUT}/nested_2x2_{name}.json", "w"), indent=2)


def _slope_controls_cell(name):
    """A lambda-only run in slope_controls.json's per-driver format."""
    r = json.load(open(f"{OUT}/nested_2x2_{name}.json"))
    drv = r["lambda_drivers"][0]
    return {
        "driver": drv,
        **r[f"lambda_{drv}"],
        "max_rhat": r["max_rhat"],
        "n_div": r["n_div"],
        "r_slope_K": r["r_slope_K"],
        "sigma_lam": r["sigma_lam"],
    }


def collect() -> None:
    res = {
        "_what": "antecedent 2x2 nested in the MoR likelihood; gain cells are log-gain per SD "
        "(W scaled by exp(g*x)), lambda cells are logit-lambda per SD."
    }
    for name in ("gain_K", "gain_slope", "joint"):
        res[name] = json.load(open(f"{OUT}/nested_2x2_{name}.json"))
    for cell in ("lambda_K", "lambda_slope"):
        res[cell] = json.load(open(f"{OUT}/nested_2x2_{cell}.json"))[cell]
    json.dump(res, open(f"{OUT}/nested_2x2.json", "w"), indent=2)
    print(f"saved -> {OUT}/nested_2x2.json")

    # c_slope control files (slope_controls.json, slope_controls_noise{,K}.json)
    cs = {
        d: _slope_controls_cell(f"lambda_{d}")
        for d in ("perm1", "perm2", "slope", "slope_resid_K")
    }
    cs["_note"] = (
        "learning distance-accuracy slope -> lambda inside the MoR, written by "
        "`pixi run antecedents --collect`. Negative = steeper gradient -> more "
        "direction-only. Drivers z-scored; slope = posterior mean of map_slope_draws.npy."
    )
    json.dump(cs, open(f"{OUT}/slope_controls.json", "w"), indent=2)
    for d in ("slope_resid_noise", "slope_resid_noiseK"):
        if os.path.exists(f"{OUT}/nested_2x2_lambda_{d}.json"):
            _f = {
                "slope_resid_noise": "slope_controls_noise",
                "slope_resid_noiseK": "slope_controls_noise_K",
            }[d]
            json.dump(
                _slope_controls_cell(f"lambda_{d}"),
                open(f"{OUT}/{_f}.json", "w"),
                indent=2,
            )
    print(f"saved -> {OUT}/slope_controls.json (+ noise controls)")


if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    collect() if sys.argv[1] == "--collect" else run(sys.argv[1])
