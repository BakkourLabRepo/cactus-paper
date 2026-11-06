"""Comparison models for the CV ladder, plus full-data fits of the group-map family and WL.

  group_map            group-level monotone value map; use_lam adds magnitude reliance;
                       fixed_uniform pins it to the taught spacing
  per_participant_map  participant-level value maps; use_lam adds magnitude reliance
  value_noise          noisy-value (erf) readout in place of magnitude reliance
  weighted_linear      per-dimension weights x lambda on the objective magnitude / sign channels

Full-data fits (results/fits/):
  group-map                     -> group_map_lambda_idata.nc
  group-map --fixed_uniform     -> taught_spacing_lambda_idata.nc
  weighted-linear               -> weighted_linear_idata.nc

Run from repo root: pixi run python -m cactus_paper.models.comparison {group-map,weighted-linear} [args]
"""

import argparse
import functools
import os
import sys
from typing import Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import DIMS
from cactus_paper.design import build_objective_per_dim, load_design
from cactus_paper.features.values import raw_values_and_diffs, taught_levels
from cactus_paper.paths import FITS, WL_IDATA
from cactus_paper.stats import rep

OUT = str(FITS)
os.makedirs(OUT, exist_ok=True)


def group_map(
    sc: jnp.ndarray,
    kl: jnp.ndarray,
    kr: jnp.ndarray,
    n: int,
    use_lam: bool = False,
    fixed_uniform: bool = False,
    y: Optional[jnp.ndarray] = None,
) -> None:
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    w_mu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    w_sd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    if fixed_uniform:
        psi_free = jnp.zeros((3, 3))  # pins increments to exactly [0.25]*4
    else:
        psi_free = numpyro.sample(
            "psi_free", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2)
        )  # (dim, 3)
    psi_full = jnp.concatenate([jnp.zeros((3, 1)), psi_free], axis=1)  # (dim,4)
    frac = numpyro.deterministic(
        "frac", jax.nn.softmax(psi_full, axis=-1)
    )  # (dim,4), sums to 1
    u_lvls = jnp.cumsum(frac, axis=-1)  # (dim,4): u(1),u(2),u(3),u(4)=1
    u_full = numpyro.deterministic(
        "u_full", jnp.concatenate([jnp.zeros((3, 1)), u_lvls], axis=1)
    )  # (dim,5)

    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        if use_lam:
            zlam = numpyro.sample("z_lam", dist.Normal(0.0, 1.0))
    W = numpyro.deterministic("W", w_mu + w_sd * zw)  # (n,3)

    u_r = jnp.stack([u_full[d, kr[:, d]] for d in range(3)], axis=1)  # (N,3)
    u_l = jnp.stack([u_full[d, kl[:, d]] for d in range(3)], axis=1)
    u_diff = u_r - u_l  # (N,3)

    if use_lam:
        c0 = numpyro.sample("c0", dist.Normal(0.0, 1.5))
        slam = numpyro.sample("sigma_lam", dist.HalfNormal(1.0))
        lam_s = numpyro.deterministic("lam_s", jax.nn.sigmoid(c0 + slam * zlam))
        sign_diff = jnp.sign(u_diff)
        veff = (1 - lam_s[sc])[:, None] * sign_diff + lam_s[sc][:, None] * u_diff
        contrib = (W[sc] * veff).sum(-1)
    else:
        contrib = (W[sc] * u_diff).sum(-1)

    eta = a0 + a[sc] * sda + contrib
    numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)


def fit_group_map() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixed_uniform",
        action="store_true",
        help="pin increments to exactly [0.25]*4 (no shape freedom) -- isolates pure "
        "channel-scale effects on lambda from shape effects",
    )
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.92)
    a = ap.parse_args()

    D = load_design()
    cd, sc, nS = D["cd"], D["sc"], D["nS"]
    Y = np.asarray(D["Y"]).astype(int)
    Vl, Vr, _ = raw_values_and_diffs(cd)
    kl, kr = taught_levels(Vl, Vr)
    tag = "taught_spacing_lambda" if a.fixed_uniform else "group_map_lambda"
    print(f"N={len(Y)} trials, nS={nS}  [{tag}]")

    model = functools.partial(group_map, use_lam=True, fixed_uniform=a.fixed_uniform)
    m = MCMC(
        NUTS(model, target_accept_prob=a.target),
        num_warmup=a.warmup,
        num_samples=a.samples,
        num_chains=a.chains,
        progress_bar=False,
    )
    m.run(
        jax.random.PRNGKey(201),
        jnp.array(sc),
        jnp.array(kl),
        jnp.array(kr),
        nS,
        y=jnp.array(Y),
    )
    ida = az.from_numpyro(m)
    po = ida.posterior
    ida.to_netcdf(os.path.join(OUT, f"{tag}_idata.nc"))

    max_rhat = float(az.rhat(ida).to_array().max())
    n_div = int(ida.sample_stats["diverging"].values.sum())
    print(f"max R-hat {max_rhat:.3f} | divergences {n_div}")

    frac = po["frac"].values.reshape(-1, 3, 4).mean(0)
    print("\n=== fitted increments (posterior mean), taught spacing = 0.25 each ===")
    for di, dim in enumerate(DIMS):
        print(f"  {dim}: {frac[di].round(3).tolist()}  (sum={frac[di].sum():.3f})")

    wm = po["w_mu"].values.reshape(-1, 3)
    print("\n=== w_mu (group weight means) ===")
    for di, dim in enumerate(DIMS):
        lo, hi = az.hdi(wm[:, di], hdi_prob=0.95)
        print(f"  w_mu[{dim:6s}] {wm[:, di].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    c0 = po["c0"].values.ravel()
    slam = po["sigma_lam"].values.ravel()
    lo, hi = az.hdi(c0, hdi_prob=0.95)
    print(f"\nc0 {c0.mean():+.3f} [{lo:+.3f},{hi:+.3f}]   sigma_lam {slam.mean():.3f}")
    lam_s = po["lam_s"].values.reshape(-1, nS).mean(0)
    print(
        f"lam_s across participants: mean={lam_s.mean():.3f} sd={lam_s.std():.3f} "
        f"range=[{lam_s.min():.3f},{lam_s.max():.3f}]"
    )

    print(f"\nsaved -> {OUT}/{tag}_idata.nc")


def value_noise(
    sc: jnp.ndarray,
    kl: jnp.ndarray,
    kr: jnp.ndarray,
    n: int,
    y: Optional[jnp.ndarray] = None,
) -> None:
    """Noisy-value readout: retrieved value v~ ~ Normal(u_d(k), sigma_s) per participant.

    Each dimension contributes E[sign(v~_R - v~_L)] = erf(delta_u / (2*sigma_s)), which runs from
    sign(delta) (sigma -> 0) to a scaled delta (sigma -> inf). Otherwise identical to group_map_lambda.
    At large sigma, w_sd and sigma_s trade off.
    """
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    w_mu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    w_sd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    psi_free = numpyro.sample(
        "psi_free", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2)
    )
    psi_full = jnp.concatenate([jnp.zeros((3, 1)), psi_free], axis=1)
    frac = numpyro.deterministic("frac", jax.nn.softmax(psi_full, axis=-1))
    u_full = numpyro.deterministic(
        "u_full", jnp.concatenate([jnp.zeros((3, 1)), jnp.cumsum(frac, -1)], axis=1)
    )  # (dim,5)
    g0 = numpyro.sample("g0", dist.Normal(0.0, 1.0))
    sg = numpyro.sample("sigma_g", dist.HalfNormal(1.0))
    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        zg = numpyro.sample("z_g", dist.Normal(0.0, 1.0))
    W = numpyro.deterministic("W", w_mu + w_sd * zw)
    sig_s = numpyro.deterministic(
        "sig_s", jnp.exp(g0 + sg * zg)
    )  # per-participant noise SD
    u_r = jnp.stack([u_full[d, kr[:, d]] for d in range(3)], axis=1)
    u_l = jnp.stack([u_full[d, kl[:, d]] for d in range(3)], axis=1)
    veff = jax.scipy.special.erf((u_r - u_l) / (2.0 * sig_s[sc])[:, None])
    contrib = (W[sc] * veff).sum(-1)
    numpyro.sample("obs", dist.Bernoulli(logits=a0 + a[sc] * sda + contrib), obs=y)


def per_participant_map(
    sc: jnp.ndarray,
    kl: jnp.ndarray,
    kr: jnp.ndarray,
    n: int,
    use_lam: bool = False,
    y: Optional[jnp.ndarray] = None,
) -> None:
    """M3 with participant-level increments (hierarchical psi_mu + sigma_psi * z_psi).

    use_lam=False -> participant_map; use_lam=True adds per-participant lambda (participant_map_lambda).
    """
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    w_mu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    w_sd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    psi_mu = numpyro.sample("psi_mu", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2))
    sigma_psi = numpyro.sample(
        "sigma_psi", dist.HalfNormal(jnp.ones((3, 3))).to_event(2)
    )
    if use_lam:
        c0 = numpyro.sample("c0", dist.Normal(0.0, 1.5))
        slam = numpyro.sample("sigma_lam", dist.HalfNormal(1.0))
    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        z_psi = numpyro.sample("z_psi", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2))
        if use_lam:
            zlam = numpyro.sample("z_lam", dist.Normal(0.0, 1.0))
    W = numpyro.deterministic("W", w_mu + w_sd * zw)
    psi_full = jnp.concatenate(
        [jnp.zeros((n, 3, 1)), psi_mu + sigma_psi * z_psi], axis=-1
    )
    frac = numpyro.deterministic("frac_s", jax.nn.softmax(psi_full, axis=-1))  # (n,3,4)
    u_full = jnp.concatenate(
        [jnp.zeros((n, 3, 1)), jnp.cumsum(frac, -1)], -1
    )  # (n,3,5)
    u_r = jnp.stack([u_full[sc, d, kr[:, d]] for d in range(3)], axis=1)
    u_l = jnp.stack([u_full[sc, d, kl[:, d]] for d in range(3)], axis=1)
    du = u_r - u_l
    if use_lam:
        lam_s = numpyro.deterministic("lam_s", jax.nn.sigmoid(c0 + slam * zlam))
        veff = (1 - lam_s[sc])[:, None] * jnp.sign(du) + lam_s[sc][:, None] * du
        contrib = (W[sc] * veff).sum(-1)
    else:
        contrib = (W[sc] * du).sum(-1)
    numpyro.sample("obs", dist.Bernoulli(logits=a0 + a[sc] * sda + contrib), obs=y)


def weighted_linear(
    sc: jnp.ndarray,
    M: jnp.ndarray,
    Sg: jnp.ndarray,
    n: int,
    y: Optional[jnp.ndarray] = None,
) -> None:
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    wmu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    wsd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    c0 = numpyro.sample("c0", dist.Normal(0.0, 1.5))
    slam = numpyro.sample("sigma_lam", dist.HalfNormal(1.0))
    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zl = numpyro.sample("z_lam", dist.Normal(0.0, 1.0))
        zwf = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
    W = numpyro.deterministic("W", wmu + wsd * zwf)  # (n,3) per-attribute weights
    lam = numpyro.deterministic("lam_s", jax.nn.sigmoid(c0 + slam * zl))
    Wt = W[sc]
    Veff = (1.0 - lam[sc]) * (Wt * Sg).sum(-1) + lam[sc] * (Wt * M).sum(-1)
    # gain = scale of W; per-subject gain = mean |W_s|
    numpyro.deterministic("wgain", jnp.abs(W).mean(-1))
    eta = a0 + a[sc] * sda + Veff
    numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)


def fit_weighted_linear() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.92)
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for the output file, e.g. --tag _prereg "
        "(use with CACTUS_SAMPLE_FILE so a refit on an "
        "alternate sample cannot clobber the fit of record)",
    )
    a = ap.parse_args()

    D = load_design()
    M, Sg = build_objective_per_dim(D)
    nS = D["nS"]
    K = np.asarray(D["K"])
    dacc = D["dacc"]
    print(
        f"N={len(D['Y'])} trials, nS={nS}  [objective per-attribute channels]",
        flush=True,
    )

    m = MCMC(
        NUTS(weighted_linear, target_accept_prob=a.target),
        num_warmup=a.warmup,
        num_samples=a.samples,
        num_chains=a.chains,
        progress_bar=False,
    )
    m.run(
        jax.random.PRNGKey(13),
        jnp.array(D["sc"]),
        jnp.array(M),
        jnp.array(Sg),
        nS,
        y=jnp.array(D["Y"]),
    )
    ida = az.from_numpyro(m)
    po = ida.posterior

    print("\n=== attribute weights (group means) ===")
    wm = po["w_mu"].values.reshape(-1, 3)
    for i, d in enumerate(DIMS):
        lo, hi = az.hdi(wm[:, i], hdi_prob=0.95)
        print(f"  w_mu[{d:6s}] {wm[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rep(po, "w_sd", "between-subject spread of the weights")
    print("=== resolution & gain ===")
    rep(po, "c0", "")
    rep(po, "sigma_lam", "")

    g = lambda k, s: po[k].values.reshape(-1, *s).mean(0)
    W = g("W", (nS, 3))
    lam = g("lam_s", (nS,))
    gain = g("wgain", (nS,))  # gain = magnitude of the weight vector
    print("\n=== per-subject structure ===")
    print(
        f"  lam:  mean {lam.mean():.3f}  sd {lam.std():.3f}  range {lam.min():.2f}-{lam.max():.2f}"
    )
    print(f"  gain: mean {gain.mean():.3f}  sd {gain.std():.3f}")
    print(
        f"  corr(lam, |W|) = {np.corrcoef(lam, gain)[0, 1]:+.3f}   (expect ~0: separable dials)"
    )
    print(
        f"  corr(lam, K) = {np.corrcoef(lam, K)[0, 1]:+.3f} | "
        f"corr(log gain, K) = {np.corrcoef(np.log(gain), K)[0, 1]:+.3f}"
    )
    print("\n  DO THE WEIGHTS TRACK PER-ATTRIBUTE KNOWLEDGE?")
    dk = np.where(np.isnan(dacc), np.nanmean(dacc, 0), dacc)
    wc = W - W.mean(1, keepdims=True)
    dc = dk - dk.mean(1, keepdims=True)
    print(
        f"    pooled within-subject corr(weight, attribute accuracy) = "
        f"{np.corrcoef(wc.ravel(), dc.ravel())[0, 1]:+.3f}"
    )
    for i, d in enumerate(DIMS):
        print(
            f"    {d:6s} corr(w, dacc) across subjects = {np.corrcoef(W[:, i], dk[:, i])[0, 1]:+.3f}"
        )

    names = ["a0", "sda", "w_mu", "w_sd", "c0", "sigma_lam"]
    r = float(az.rhat(ida, var_names=names).to_array().max())
    print(
        f"\n  max R-hat {r:.3f} | divergences {int(ida.sample_stats['diverging'].values.sum())}"
    )
    out = str(WL_IDATA).replace("_idata.nc", f"{a.tag}_idata.nc")
    ida.to_netcdf(out)
    print(f"saved -> {out}")


COMMANDS = {"group-map": fit_group_map, "weighted-linear": fit_weighted_linear}

if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.models.comparison {group-map|weighted-linear} [args]"
        )
    _cmd = sys.argv.pop(1)
    COMMANDS[_cmd]()
