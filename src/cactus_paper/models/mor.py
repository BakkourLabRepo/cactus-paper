"""The model of record: per-participant value map, within-band slope, magnitude reliance.

    u_eff_d(side) = u_d(k) + s_d * p          p in [-0.5,+0.5], value-referenced

s_d = 0 is a pure staircase; FULL_RAMP (0.15) is a locally angle-linear map, so s_d / FULL_RAMP is
the fraction of a fully graded map.


Run from repo root: pixi run python -m cactus_paper.models.mor
Outputs -> results/fits/mor_idata.nc (= MOR_IDATA) and
           results/fits/mor_fit.json
"""

import json
from typing import Any, Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import DIMS

TAU = 0.05  # tanh sharpness of the ordinal channel; see mor


def mor(
    sc: jnp.ndarray,
    kl: jnp.ndarray,
    kr: jnp.ndarray,
    pl: jnp.ndarray,
    pr: jnp.ndarray,
    n: int,
    tau: float = TAU,
    gain_x: Optional[jnp.ndarray] = None,
    lam_x: Optional[jnp.ndarray] = None,
    device: Optional[jnp.ndarray] = None,
    y: Optional[jnp.ndarray] = None,
) -> None:
    """Participant-level u + lambda + within-band slope, with a tanh(x/tau) ordinal channel.

    tanh(x/tau) is a differentiable stand-in for sign (jnp.sign gives NUTS no gradient in s).
    With tau = 0.05, a taught step (~0.25) gives tanh = 0.9999, i.e. effectively sign().

    The ordinal channel reads the taught category difference, so within-band position moves the
    magnitude channel alone.

    Optional participant-level terms, each zero-able back to the model of record:
      gain_x  (n, k) drivers of log gain:    W_sd = exp(gain_x @ g) * (w_mu_d + w_sd z_sd)
      lam_x   (n, k) drivers of lambda:      logit lambda_s = c0 + lam_x @ c + sigma_lam z_s
      device  (n,)   sum-coded pointer device, additive on both: w_dev on W, c_dev on lambda
    The sample sites each option adds keep their original positions, so existing fits replay.
    """
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    w_mu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    if device is not None:
        w_dev = numpyro.sample("w_dev", dist.Normal(0.0, 1.0))
    w_sd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    psi_mu = numpyro.sample("psi_mu", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2))
    sigma_psi = numpyro.sample(
        "sigma_psi", dist.HalfNormal(jnp.ones((3, 3))).to_event(2)
    )
    s = numpyro.sample("s_band", dist.Normal(jnp.zeros(3), 0.25).to_event(1))
    c0 = numpyro.sample("c0", dist.Normal(0.0, 1.5))
    if device is not None:
        c_dev = numpyro.sample("c_dev", dist.Normal(0.0, 1.0))
    slam = numpyro.sample("sigma_lam", dist.HalfNormal(1.0))
    if gain_x is not None:
        gain_terms = jnp.zeros(n)
        if gain_x.shape[1]:
            g = numpyro.sample(
                "g", dist.Normal(jnp.zeros(gain_x.shape[1]), 1.0).to_event(1)
            )
            gain_terms = gain_x @ g
    if lam_x is not None:
        lam_terms = 0.0
        if lam_x.shape[1]:
            c = numpyro.sample(
                "c", dist.Normal(jnp.zeros(lam_x.shape[1]), 1.0).to_event(1)
            )
            lam_terms = lam_x @ c
    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        z_psi = numpyro.sample("z_psi", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2))
        zl = numpyro.sample("z_lam", dist.Normal(0.0, 1.0))
    w = w_mu if device is None else w_mu + w_dev * device[:, None]
    w = w + w_sd * zw
    W = numpyro.deterministic(
        "W", w if gain_x is None else jnp.exp(gain_terms)[:, None] * w
    )
    logit = c0 if device is None else c0 + c_dev * device
    if lam_x is not None:
        logit = logit + lam_terms
    lam = numpyro.deterministic("lam_s", jax.nn.sigmoid(logit + slam * zl))
    psi_full = jnp.concatenate(
        [jnp.zeros((n, 3, 1)), psi_mu + sigma_psi * z_psi], axis=-1
    )
    u = jnp.concatenate(
        [jnp.zeros((n, 3, 1)), jnp.cumsum(jax.nn.softmax(psi_full, -1), -1)], -1
    )
    uR = jnp.stack([u[sc, d, kr[:, d]] for d in range(3)], 1) + s * pr
    uL = jnp.stack([u[sc, d, kl[:, d]] for d in range(3)], 1) + s * pl
    du = uR - uL
    du_taught = jnp.stack([u[sc, d, kr[:, d]] for d in range(3)], 1) - jnp.stack(
        [u[sc, d, kl[:, d]] for d in range(3)], 1
    )
    ordinal = jnp.tanh(du_taught / tau)
    veff = (1 - lam[sc])[:, None] * ordinal + lam[sc][:, None] * du
    numpyro.sample(
        "obs", dist.Bernoulli(logits=a0 + a[sc] * sda + (W[sc] * veff).sum(-1)), obs=y
    )


def fit_mor(
    D: Dict[str, Any],
    rec: Dict[str, Any],
    OUT: str,
    FULL_RAMP: float,
    warmup: int = 1500,
    samples: int = 1500,
) -> None:
    import functools as _ft

    nS = D["nS"]
    sc = jnp.array(D["sc"])
    Y = jnp.array(np.asarray(D["Y"]).astype(int))
    kl = jnp.array(np.stack([rec[d]["kL"] for d in DIMS], 1))
    kr = jnp.array(np.stack([rec[d]["kR"] for d in DIMS], 1))
    pl = jnp.array(np.stack([rec[d]["pL"] for d in DIMS], 1))
    pr = jnp.array(np.stack([rec[d]["pR"] for d in DIMS], 1))
    res = {"_tau": TAU, "_full_ramp_benchmark": round(float(FULL_RAMP), 4)}
    for tag in ["mor"]:
        mdl = mor
        m = MCMC(
            NUTS(mdl, target_accept_prob=0.92),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=4,
            progress_bar=False,
        )
        m.run(jax.random.PRNGKey(621), sc, kl, kr, pl, pr, nS, y=Y)
        ida = az.from_numpyro(m)
        po = ida.posterior
        ida.to_netcdf(f"{OUT}/{tag}_idata.nc")
        S = po["s_band"].values.reshape(-1, 3)
        rh = float(
            az.rhat(ida, var_names=["a0", "sda", "w_mu", "w_sd", "s_band"])
            .to_array()
            .max()
        )
        nd = int(ida.sample_stats["diverging"].values.sum())
        blk = {
            "max_rhat": round(rh, 4),
            "n_divergences": nd,
            "converged": bool(rh <= 1.01),
        }
        print(
            f"\n[{tag}]  R-hat {rh:.4f}  div {nd}"
            f"{'' if rh <= 1.01 else '   <-- STILL NOT CONVERGED'}",
            flush=True,
        )
        for j, d in enumerate(DIMS):
            v = S[:, j]
            lo, hi = az.hdi(v, hdi_prob=0.95)
            blk[d] = {
                "s": round(float(v.mean()), 4),
                "lo": round(float(lo), 4),
                "hi": round(float(hi), 4),
                "frac_of_full_ramp": round(float(v.mean() / FULL_RAMP), 3),
            }
            print(
                f"  s[{d:6s}] {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                f"= {blk[d]['frac_of_full_ramp']:.2f} x full ramp",
                flush=True,
            )
        res[tag] = blk
    json.dump(res, open(f"{OUT}/mor_fit.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/mor_fit.json", flush=True)


def main() -> None:
    from cactus_paper.constants import FULL_RAMP
    from cactus_paper.design import load_design
    from cactus_paper.features.geometry import choice_frame
    from cactus_paper.paths import FITS

    D = load_design()
    rec, _ = choice_frame(D)
    fit_mor(D, rec, str(FITS), FULL_RAMP)


if __name__ == "__main__":
    main()
