"""Shared inputs for the stats.json sections: the posterior, the design, and what every section reuses.

Built once by `Context.load`; each section module takes it and returns its block of keys.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import DIMS, POS
from cactus_paper.design import (
    build_objective_channels,
    build_objective_per_dim,
    load_design,
)
from cactus_paper.features.traits import concentration as wl_conc
from cactus_paper.features.traits import gain as wl_gain
from cactus_paper.paths import DATA
from cactus_paper.stats import summarize


def summ(draws_1d: Any) -> Dict[str, Any]:
    """Posterior mean, 95% HDI (az.hdi), P(direction). `pdir` is TWO-SIDED."""
    return summarize(draws_1d, nd=3, direction="pdir")


def zc1(v: Any) -> np.ndarray:
    return (np.asarray(v, float) - np.nanmean(v)) / np.nanstd(v)


def bayes_corr(
    x: Any,
    y: Any,
    warmup: int = 1000,
    samples: int = 2000,
    chains: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Bayesian Pearson correlation: bivariate-normal likelihood on standardized (x, y),
    rho ~ Uniform(-1, 1). For correlations between observed quantities with no upstream posterior.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    xs = (x[ok] - x[ok].mean()) / x[ok].std()
    ys = (y[ok] - y[ok].mean()) / y[ok].std()
    xy = jnp.stack([jnp.asarray(xs), jnp.asarray(ys)], axis=-1)

    def model(xy):
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        cov = jnp.array([[1.0, rho], [rho, 1.0]])
        numpyro.sample(
            "obs", dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov), obs=xy
        )

    mcmc = MCMC(
        NUTS(model),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), xy)
    return summ(mcmc.get_samples()["rho"])


def bfit(
    formula: str,
    df: pd.DataFrame,
    family: str,
    fit_kw: Dict[str, Any],
    focals: List[str],
) -> Dict[str, Dict[str, Any]]:
    import bambi as bmb

    m = bmb.Model(formula, df, family=family).fit(**fit_kw)
    return {f: summ(m.posterior[f].values) for f in focals}


def sampler(quick: bool) -> Dict[str, Any]:
    """bambi/nutpie settings for every stats.json refit; `quick` = smoke-test scale (~1 min)."""
    return dict(
        draws=200 if quick else 2000,
        tune=200 if quick else 2000,
        chains=2 if quick else 4,
        inference_method="nutpie",
        random_seed=20260821,
    )


class Context(SimpleNamespace):
    @classmethod
    def load(cls, idata: str, quick: bool = False, tag: str = "") -> "Context":
        po = az.from_netcdf(idata).posterior
        D = load_design()
        sc, nS, cd = D["sc"], D["nS"], D["cd"]
        K = np.asarray(D["K"])
        Y = np.asarray(D["Y"]).astype(int)
        dacc = D["dacc"]
        div = np.asarray(D["div"])
        su = cd.run_id.drop_duplicates().tolist()

        # per-subject constructs: per-draw (`*D`) and posterior means
        _nS = po["lam_s"].shape[-1]
        lamD = po["lam_s"].values.reshape(-1, _nS)
        WD = po["W"].values.reshape(-1, _nS, 3)
        lam_s = lamD.mean(0)
        W = WD.mean(0)
        absW = np.abs(W)
        gain = wl_gain(absW)  # `wgain`; raw scale, not logged
        conc = wl_conc(absW)  # GINI of |w|; 0 = even, 2/3 = one cue (D = 3)
        gainD = wl_gain(np.abs(WD))
        concD = wl_conc(np.abs(WD))

        # objective channels
        M, Sg = build_objective_per_dim(D)
        Vmag, _, _ = build_objective_channels(
            D
        )  # summed objective magnitude, z within subject
        Vtal = np.asarray(D["Vr"])  # objective tally
        conflict = (np.sign(Vmag) != np.sign(Vtal)) & (Vmag != 0) & (Vtal != 0)

        def _side(s_, d):
            out = np.full(len(cd), np.nan)
            for p in POS:
                dm = cd[f"choice_trial_{s_}_{p}_dim"].values
                vl = cd[f"choice_trial_{s_}_{p}_value"].values
                m = dm == d
                out[m] = vl[m]
            return out

        VDo = np.stack(
            [_side("right", d) - _side("left", d) for d in DIMS], 1
        )  # objective, unscaled
        agg = np.sign(VDo.sum(1))
        follow_mag = (((Y == 1) & (agg > 0)) | ((Y == 0) & (agg < 0))).astype(int)

        dc = pd.read_csv(f"{DATA}/dim_counts.csv")
        dc = dc[dc.run_id.isin(su)]
        payne = dc.groupby("run_id").payne_index.mean().loc[su].values

        # within-person spread of learning across attributes (learning block; Fig 3C moderator)
        spread = np.nanmax(dacc, 1) - np.nanmin(dacc, 1)

        return cls(
            idata=idata,
            quick=quick,
            tag=tag,
            FIT=sampler(quick),
            po=po,
            D=D,
            sc=sc,
            nS=nS,
            cd=cd,
            K=K,
            Y=Y,
            dacc=dacc,
            div=div,
            su=su,
            lamD=lamD,
            WD=WD,
            lam_s=lam_s,
            W=W,
            absW=absW,
            gain=gain,
            conc=conc,
            gainD=gainD,
            concD=concD,
            M=M,
            Sg=Sg,
            Vmag=Vmag,
            Vtal=Vtal,
            conflict=conflict,
            VDo=VDo,
            agg=agg,
            follow_mag=follow_mag,
            payne=payne,
            spread=spread,
        )
