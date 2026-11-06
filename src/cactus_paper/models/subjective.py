"""subjective.py — stage-1 learning-task subjective utility model (anchored hierarchical Bayes).

Per attribute: a group value SHAPE mu (zero-mean, unit RMS) with per-subject deviations
u = mu + sigma*z, and a log-normal sensitivity beta; learning-phase pairwise choices follow
sign * beta * (u_left - u_right). Fit on learning-phase data only.

Consumed by cactus_paper.features.udim_cache (posterior-mean u cache for the model scripts)
and manuscript/make_stats.py (Bayesian beta*u steepness ~ attribute-knowledge correlation).
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import LEVELS

LVL = {v: i for i, v in enumerate(LEVELS)}


def u_dim_model(
    scode: jnp.ndarray,
    li: jnp.ndarray,
    ri: jnp.ndarray,
    sign: jnp.ndarray,
    n_subj: int,
    y: Optional[jnp.ndarray] = None,
) -> None:
    mu_raw = numpyro.sample("mu_raw", dist.Normal(jnp.zeros(5), 1.0).to_event(1))
    mu_c = mu_raw - mu_raw.mean()
    mu = numpyro.deterministic(
        "mu", mu_c / jnp.sqrt((mu_c**2).mean() + 1e-6)
    )  # unit-RMS group SHAPE
    sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.ones(5)).to_event(1))
    lb_mu = numpyro.sample("lb_mu", dist.Normal(0.0, 1.0))
    lb_sd = numpyro.sample("lb_sd", dist.HalfNormal(1.0))
    with numpyro.plate("subj", n_subj):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(5), 1.0).to_event(1))
        lb = numpyro.sample("lb", dist.Normal(0.0, 1.0))
    u = numpyro.deterministic("u", mu + sigma * z)
    beta = numpyro.deterministic("beta", jnp.exp(lb_mu + lb_sd * lb))
    diff = sign * beta[scode] * (u[scode, li] - u[scode, ri])
    numpyro.sample("obs", dist.Bernoulli(logits=diff), obs=y)


def fit_u_dim_draws(
    learning_data: pd.DataFrame,
    seed: int = 20260618,
    warmup: int = 1000,
    samples: int = 1000,
    chains: int = 4,
    target: float = 0.92,
) -> Dict[str, Dict[str, Any]]:
    """Fit the utility model per attribute; return per-dim posterior draws.

    Returns {dim: {"suniq": run_ids in subject order, "u": (S, n_subj, 5) draws,
                   "beta": (S, n_subj) draws}}.
    """
    out = {}
    for dim, ld in learning_data.groupby("dim"):
        s = ld.dropna(
            subset=["left_rew", "right_rew", "chose_left", "prompt_dir"]
        ).copy()
        scode, suniq = pd.factorize(s.run_id)
        sign = np.where(s.prompt_dir.values == "bigger", 1.0, -1.0)
        li = s.left_rew.map(LVL).values.astype(int)
        ri = s.right_rew.map(LVL).values.astype(int)
        mcmc = MCMC(
            NUTS(u_dim_model, target_accept_prob=target),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(seed),
            jnp.array(scode),
            jnp.array(li),
            jnp.array(ri),
            jnp.array(sign),
            len(suniq),
            y=jnp.array(s.chose_left.astype(int).values),
        )
        smp = mcmc.get_samples()
        out[dim] = {
            "suniq": [int(r) for r in suniq],
            "u": np.array(smp["u"]),
            "beta": np.array(smp["beta"]),
        }
        print(
            f"  {dim}: {len(suniq)} subjects, {out[dim]['u'].shape[0]} draws",
            flush=True,
        )
    return out
