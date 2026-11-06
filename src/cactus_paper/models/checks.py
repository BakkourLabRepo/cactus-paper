"""Model checks: parameter recovery, forward posterior-predictive checks, identifiability.

  recovery         simulate from the model of record, refit, score per-participant recovery
                   -> results/fits/recovery.json
  ppc              forward PPC of the descriptive choice signatures -> results/fits/forward_ppc.json
  identifiability  simulate from each rival model, re-run the CV contrast
                   -> results/fits/identifiability{,_slope}.json

Run from repo root: pixi run python -m cactus_paper.models.checks {recovery,ppc,identifiability} [args]
"""

import argparse
import functools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Sequence

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood

from cactus_paper.constants import CV_SEED, DIMS
from cactus_paper.design import load_design, within_band_position
from cactus_paper.features.values import (
    follow_magnitude,
    raw_values_and_diffs,
    taught_levels,
)
from cactus_paper.models.cv import build_models
from cactus_paper.models.mor import mor
from cactus_paper.paths import FITS, MOR_IDATA, WL_IDATA

OUT = str(FITS)


def run_recovery() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()
    D = load_design()
    nS, sc = D["nS"], jnp.array(D["sc"])
    Vl, Vr, _ = raw_values_and_diffs(D["cd"])
    _kl, _kr = taught_levels(Vl, Vr)
    kl, kr = jnp.array(_kl), jnp.array(_kr)
    _pl, _pr = within_band_position(D["cd"], np.asarray(D["sc"]))
    mdl = mor
    sites = [
        "a0",
        "sda",
        "w_mu",
        "w_sd",
        "psi_mu",
        "sigma_psi",
        "s_band",
        "c0",
        "sigma_lam",
        "a",
        "z_w",
        "z_psi",
        "z_lam",
    ]
    _extra = [jnp.array(_pl), jnp.array(_pr)]

    po = az.from_netcdf(str(MOR_IDATA)).posterior
    flat = {k: po[k].values.reshape(-1, *po[k].values.shape[2:]) for k in sites}
    rng = np.random.default_rng(2026)
    idx = rng.choice(flat["a0"].shape[0], a.reps, replace=False)
    sub = {k: jnp.array(v[idx]) for k, v in flat.items()}
    # simulate one rep at a time via substitute+seed (batched Predictive fails on the band model)
    from numpyro.handlers import seed as npy_seed
    from numpyro.handlers import substitute
    from numpyro.handlers import trace as npy_trace

    Ysim = np.stack(
        [
            np.asarray(
                npy_trace(
                    npy_seed(
                        substitute(mdl, {k: v[r] for k, v in sub.items()}),
                        jax.random.PRNGKey(77 + r),
                    )
                ).get_trace(sc, kl, kr, *_extra, nS)["obs"]["value"]
            )
            for r in range(a.reps)
        ]
    ).astype(int)

    # true per-participant quantities, recomputed from the drawn latents exactly as the model does
    trueW = np.asarray(
        flat["w_mu"][idx][:, None, :]
        + flat["w_sd"][idx][:, None, None] * flat["z_w"][idx]
    )  # (reps,nS,3)
    truelam = 1 / (
        1
        + np.exp(
            -(
                flat["c0"][idx][:, None]
                + flat["sigma_lam"][idx][:, None] * flat["z_lam"][idx]
            )
        )
    )
    truegain = np.abs(trueW).mean(-1)

    acc = {k: [] for k in ["W_color", "W_shape", "W_line", "lam", "gain"]}
    cov = {k: [] for k in acc}
    # per-rep true and recovered corr(lam, gain), for the scale-confound check
    true_lam_gain_r, rec_lam_gain_r = [], []
    for r in range(a.reps):
        m = MCMC(
            NUTS(mdl, target_accept_prob=0.92),
            num_warmup=a.warmup,
            num_samples=a.samples,
            num_chains=4,
            progress_bar=False,
        )
        m.run(
            jax.random.PRNGKey(900 + r), sc, kl, kr, *_extra, nS, y=jnp.array(Ysim[r])
        )
        p = az.from_numpyro(m).posterior
        Wd = p["W"].values.reshape(-1, nS, 3)
        ld = p["lam_s"].values.reshape(-1, nS)
        gd = np.abs(Wd).mean(-1)
        est = {
            "lam": (ld, truelam[r]),
            "gain": (gd, truegain[r]),
            **{f"W_{d}": (Wd[:, :, j], trueW[r, :, j]) for j, d in enumerate(DIMS)},
        }
        for k, (draws, tru) in est.items():
            acc[k].append(float(np.corrcoef(draws.mean(0), tru)[0, 1]))
            lo, hi = az.hdi(draws, hdi_prob=0.95).T
            cov[k].append(float(((tru >= lo) & (tru <= hi)).mean()))
        true_lam_gain_r.append((float(np.corrcoef(truelam[r], truegain[r])[0, 1]),))
        rec_lam_gain_r.append(float(np.corrcoef(ld.mean(0), gd.mean(0))[0, 1]))
        print(
            f"  rep {r + 1}/{a.reps}  "
            + "  ".join(
                f"{k} r={acc[k][-1]:.3f}/cov={cov[k][-1]:.2f}" for k in ("lam", "gain")
            )
            + f"  |  corr(true lam,true gain)={true_lam_gain_r[-1][0]:+.3f}"
            f"  corr(recovered lam,recovered gain)={rec_lam_gain_r[-1]:+.3f}",
            flush=True,
        )

    res = {"_n_reps": a.reps, "_n_subj": int(nS), "_nominal_coverage": 0.95}
    for k in acc:
        res[k] = {
            "r_mean": round(float(np.mean(acc[k])), 3),
            "r_min": round(float(np.min(acc[k])), 3),
            "coverage_mean": round(float(np.mean(cov[k])), 3),
            "coverage_min": round(float(np.min(cov[k])), 3),
        }
    res["r_min_across_weights"] = round(min(res[f"W_{d}"]["r_min"] for d in DIMS), 3)
    res["coverage_mean_all"] = round(
        float(np.mean([res[k]["coverage_mean"] for k in acc])), 3
    )

    # scale-confound check: recovered corr(lam, gain) in reps whose true corr is near zero
    tlg = np.array([t[0] for t in true_lam_gain_r])
    rlg = np.array(rec_lam_gain_r)
    near_zero = np.abs(tlg) < 0.10  # "uncorrelated" reps
    res["scale_confound_check"] = {
        "per_rep_true_lam_gain_corr": [round(float(v), 3) for v in tlg],
        "per_rep_recovered_lam_gain_corr": [round(float(v), 3) for v in rlg],
        "n_reps_with_near_zero_true_corr": int(near_zero.sum()),
        "recovered_corr_mean_all_reps": round(float(rlg.mean()), 3),
        "recovered_corr_mean_near_zero_true_reps": (
            round(float(rlg[near_zero].mean()), 3) if near_zero.any() else None
        ),
        "_interpretation": "if recovered_corr_mean_near_zero_true_reps is near the observed "
        "+0.249 (corr_lam_gain) despite true corr ~0, "
        "fitting alone manufactures the coupling -- it is a null, not a "
        "finding. If recovered corr stays near 0 there too, the fitting "
        "procedure is not the source of +0.249.",
    }
    print(f"\nscale-confound check: true lam-gain corr per rep = {np.round(tlg, 3)}")
    print(
        f"                       recovered lam-gain corr per rep = {np.round(rlg, 3)}"
    )
    print(
        f"                       {int(near_zero.sum())}/{a.reps} reps have |true corr| < 0.10"
    )
    if near_zero.any():
        print(
            f"                       recovered corr on those reps: mean = "
            f"{rlg[near_zero].mean():+.3f}  (compare to observed +0.249)"
        )

    print(
        "\n"
        + f"{'param':10s} {'r mean':>8s} {'r min':>8s} {'cov mean':>9s} {'cov min':>8s}"
    )
    for k in acc:
        v = res[k]
        print(
            f"{k:10s} {v['r_mean']:8.3f} {v['r_min']:8.3f} {v['coverage_mean']:9.3f} {v['coverage_min']:8.3f}"
        )
    print(
        f"\nweights: min r across dimensions = {res['r_min_across_weights']:.3f} "
        f"(manuscript standard: >= 0.90)"
    )
    print(f"mean interval coverage = {res['coverage_mean_all']:.3f} (nominal 0.95)")
    json.dump(res, open(f"{OUT}/recovery.json", "w"), indent=2)
    print(f"saved -> {OUT}/recovery.json")


def irls(X: np.ndarray, y: np.ndarray, iters: int = 25) -> np.ndarray:
    """Plain Newton IRLS logistic fit; same estimator for observed and simulated."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-6)
        z = eta + (y - p) / w
        XtW = X.T * w
        try:
            b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        except np.linalg.LinAlgError:
            return b
        if np.max(np.abs(b_new - b)) < 1e-8:
            return b_new
        b = b_new
    return b


def stats_of(y: np.ndarray, ctx: Dict[str, Any]) -> Dict[str, float]:
    """The four reported signatures, from one (real or simulated) choice vector."""
    X, Xa, div, sc, nS, rank, VD = (
        ctx[k] for k in ("X", "Xa", "div", "sc", "nS", "rank", "VD")
    )
    out = {}
    b = irls(X, y)
    out["Vc"], out["Vc_K"] = float(b[1]), float(b[3])
    fm = follow_magnitude(y, VD)
    ba = irls(Xa, fm[div])
    out["cAbsVc"], out["cAbsVc_K"] = float(ba[1]), float(ba[3])
    # per-participant accuracy on the two trial types, then mean and BETWEEN-participant SD
    for nm, msk in (("congruent", ~div), ("incongruent", div)):
        per = np.array(
            [
                fm[msk & (sc == s)].mean() if (msk & (sc == s)).any() else np.nan
                for s in range(nS)
            ]
        )
        out[f"{nm}_mean"] = float(np.nanmean(per))
        out[f"{nm}_sd"] = float(np.nanstd(per))
    for j, nm in enumerate(("worst", "middle", "best")):
        m = div & (rank == j)
        out[f"ladder_{nm}"] = float(fm[m].mean()) if m.any() else np.nan
    return out


def run_ppc() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    D = load_design()
    cd, sc, nS = D["cd"], D["sc"], D["nS"]
    Y = np.asarray(D["Y"]).astype(int)
    Vc, K, div, dacc = D["Vc"], np.asarray(D["K"], float), D["div"], D["dacc"]
    VD = D["VD"]
    Vl, Vr, _ = raw_values_and_diffs(cd)
    kl, kr = taught_levels(Vl, Vr)

    # rank of the critical attribute within that participant's own knowledge ordering (0=worst)
    dk = np.where(np.isnan(dacc), np.nanmean(dacc, 0), dacc)
    order = np.argsort(np.argsort(dk, 1), 1)  # (nS,3) 0=worst .. 2=best
    crit_j = np.array([DIMS.index(d) for d in cd.crit_dim])
    rank = order[sc, crit_j]

    Kt = K[sc]
    X = np.column_stack([np.ones(len(Y)), Vc, Kt, Vc * Kt])
    cav = np.abs(Vc)[div]
    cav = cav - cav.mean()
    Ka = Kt[div]
    Xa = np.column_stack([np.ones(div.sum()), cav, Ka, cav * Ka])
    ctx = dict(X=X, Xa=Xa, div=div, sc=sc, nS=nS, rank=rank, VD=VD)

    obs = stats_of(Y, ctx)
    print(
        f"N={len(Y)} trials, {nS} participants, {int(div.sum())} incongruent",
        flush=True,
    )

    _pl, _pr = within_band_position(D["cd"], sc)
    po = az.from_netcdf(MOR_IDATA).posterior
    sites = [
        "a0",
        "sda",
        "w_mu",
        "w_sd",
        "psi_mu",
        "sigma_psi",
        "s_band",
        "c0",
        "sigma_lam",
        "a",
        "z_w",
        "z_psi",
        "z_lam",
    ]
    mdl = mor
    margs = [
        jnp.array(sc),
        jnp.array(kl),
        jnp.array(kr),
        jnp.array(_pl),
        jnp.array(_pr),
        nS,
    ]
    flat = {k: po[k].values.reshape(-1, *po[k].values.shape[2:]) for k in sites}
    rng = np.random.default_rng(2026)
    idx = rng.choice(flat["a0"].shape[0], a.draws, replace=False)
    sub = {k: jnp.array(v[idx]) for k, v in flat.items()}
    Ysim = np.asarray(
        Predictive(mdl, posterior_samples=sub)(jax.random.PRNGKey(88), *margs)["obs"]
    ).astype(int)
    print(f"simulated {Ysim.shape[0]} datasets", flush=True)

    sims = {k: [] for k in obs}
    for i in range(Ysim.shape[0]):
        s = stats_of(Ysim[i], ctx)
        for k, v in s.items():
            sims[k].append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{Ysim.shape[0]}", flush=True)

    res = {
        "_n_draws": int(a.draws),
        "_estimator": "pooled Newton IRLS, identical for obs and sim",
    }
    print(
        f"\n{'signature':18s} {'observed':>10s} {'sim mean':>10s} {'sim 95%':>20s} {'pct':>6s} {'2-sided p':>10s}"
    )
    for k in obs:
        v = np.asarray(sims[k], float)
        v = v[np.isfinite(v)]
        pct = float((v < obs[k]).mean())
        p2 = float(2 * min(pct, 1 - pct))
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = {
            "observed": round(obs[k], 4),
            "sim_mean": round(float(v.mean()), 4),
            "sim_lo": round(float(lo), 4),
            "sim_hi": round(float(hi), 4),
            "percentile": round(pct, 3),
            "two_sided_p": round(p2, 3),
            "reproduced": bool(0.025 <= pct <= 0.975),
        }
        flag = "" if res[k]["reproduced"] else "   <-- NOT REPRODUCED"
        print(
            f"{k:18s} {obs[k]:10.4f} {v.mean():10.4f} [{lo:8.4f},{hi:8.4f}] {pct:6.3f} {p2:10.3f}{flag}"
        )
    nrep = [k for k in obs if not res[k]["reproduced"]]
    res["_not_reproduced"] = nrep
    print(
        f"\n{len(obs) - len(nrep)}/{len(obs)} signatures reproduced"
        + (f"; FAILED: {nrep}" if nrep else "")
    )
    json.dump(res, open(f"{OUT}/forward_ppc.json", "w"), indent=2)
    print(f"saved -> {OUT}/forward_ppc.json")


# generator -> (idata path, key into build_models for its args/model, sampled (non-deterministic) sites)
GENERATORS = {
    "weighted_linear": (
        str(WL_IDATA),
        "weighted_linear",
        ["a0", "sda", "w_mu", "w_sd", "c0", "sigma_lam", "a", "z_lam", "z_w"],
    ),
    "group_map_lambda": (
        f"{OUT}/group_map_lambda_idata.nc",
        "group_map_lambda",
        [
            "a0",
            "sda",
            "w_mu",
            "w_sd",
            "psi_free",
            "c0",
            "sigma_lam",
            "a",
            "z_lam",
            "z_w",
        ],
    ),
}


FITTED = ["weighted_linear", "taught_spacing_lambda", "group_map_lambda"]


CONTRASTS = [
    ("group_map_lambda", "weighted_linear"),
    ("group_map_lambda", "taught_spacing_lambda"),
    ("taught_spacing_lambda", "weighted_linear"),
]


# --band: within-band slope contrast (mor vs participant_map_lambda)
_H = [
    "a0",
    "sda",
    "w_mu",
    "w_sd",
    "psi_mu",
    "sigma_psi",
    "c0",
    "sigma_lam",
    "a",
    "z_w",
    "z_psi",
    "z_lam",
]


GENERATORS_BAND = {
    "participant_map_lambda": (
        f"{OUT}/participant_map_lambda_idata.nc",
        "participant_map_lambda",
        _H,
    ),
    "mor": (
        str(__import__("cactus_paper.paths", fromlist=["MOR_IDATA"]).MOR_IDATA),
        "mor",
        _H + ["s_band"],
    ),
}


FITTED_BAND = ["participant_map_lambda", "mor"]


CONTRASTS_BAND = [("mor", "participant_map_lambda")]


def simulate(
    path: str,
    model: Callable[..., None],
    args: Sequence[Any],
    sites: Sequence[str],
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `reps` synthetic datasets, each from ONE posterior draw of the generating model."""
    po = az.from_netcdf(path).posterior
    flat = {k: po[k].values.reshape(-1, *po[k].values.shape[2:]) for k in sites}
    idx = rng.choice(flat["a0"].shape[0], reps, replace=False)
    sub = {k: jnp.array(v[idx]) for k, v in flat.items()}
    obs = Predictive(model, posterior_samples=sub)(jax.random.PRNGKey(CV_SEED), *args)[
        "obs"
    ]
    return np.asarray(obs).astype(int)  # (reps, N)


def cv_lpd(
    mdl: Callable[..., None],
    arrs: Sequence[np.ndarray],
    ta: float,
    nS: int,
    Y: np.ndarray,
    perm: np.ndarray,
    folds_ix: Sequence[np.ndarray],
    warmup: int,
    samples: int,
    chains: int,
) -> np.ndarray:
    out = np.zeros(len(Y))
    for fi, te in enumerate(folds_ix):
        tr = np.setdiff1d(perm, te)
        m = MCMC(
            NUTS(mdl, target_accept_prob=ta),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress_bar=False,
        )
        m.run(
            jax.random.PRNGKey(CV_SEED + fi),
            *[jnp.array(x[tr]) for x in arrs],
            nS,
            y=jnp.array(Y[tr]),
        )
        ll = log_likelihood(
            mdl,
            m.get_samples(),
            *[jnp.array(x[te]) for x in arrs],
            nS,
            y=jnp.array(Y[te]),
        )["obs"]
        out[te] = np.asarray(logsumexp(ll, 0) - np.log(ll.shape[0]))
        del m, ll
    return out


def run_identifiability() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--band",
        action="store_true",
        help="check the map-slope contrast (mor vs participant_map_lambda) instead",
    )
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    D = load_design()
    nS, N = D["nS"], len(D["Y"])
    models = build_models(D)
    rng = np.random.default_rng(CV_SEED)
    perm = rng.permutation(N)
    folds_ix = np.array_split(perm, a.folds)

    gens, fitted, contrasts = (
        (GENERATORS_BAND, FITTED_BAND, CONTRASTS_BAND)
        if a.band
        else (GENERATORS, FITTED, CONTRASTS)
    )
    observed = json.load(open(os.path.join(OUT, "cv.json")))["contrasts"]
    res = {
        "n_trials": N,
        "reps": a.reps,
        "folds": a.folds,
        "observed": {
            f"{b}_minus_{s}": observed[f"{b}_minus_{s}"]["delta"] for b, s in contrasts
        },
        "by_generator": {},
    }

    se = lambda x: float(x.std() * np.sqrt(len(x)))
    for gen, (path, mkey, sites) in gens.items():
        gmdl, gargs, _ = models[mkey]
        print(f"\n### simulating {a.reps} datasets from generator '{gen}'", flush=True)
        Ysim = simulate(
            path, gmdl, [jnp.array(x) for x in gargs] + [nS], sites, a.reps, rng
        )
        print(
            f"    mean P(right) across reps: "
            f"{[round(float(y.mean()), 3) for y in Ysim]}",
            flush=True,
        )
        res["by_generator"][gen] = {"reps": []}
        for r in range(a.reps):
            lpd, t0 = {}, time.time()
            for nm in fitted:
                mdl, arrs, ta = models[nm]
                lpd[nm] = cv_lpd(
                    mdl,
                    arrs,
                    ta,
                    nS,
                    Ysim[r],
                    perm,
                    folds_ix,
                    a.warmup,
                    a.samples,
                    a.chains,
                )
                print(
                    f"    [{gen} rep{r}] {nm:14s} elpd={lpd[nm].sum():+9.1f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            row = {}
            for b, s in contrasts:
                d = lpd[b] - lpd[s]
                row[f"{b}_minus_{s}"] = {"delta": float(d.sum()), "se": se(d)}
                print(
                    f"      {b:14s} - {s:14s} = {d.sum():+8.1f} +/- {se(d):5.1f}",
                    flush=True,
                )
            res["by_generator"][gen]["reps"].append(row)
        for b, s in contrasts:
            k = f"{b}_minus_{s}"
            v = [rr[k]["delta"] for rr in res["by_generator"][gen]["reps"]]
            res["by_generator"][gen][k] = {
                "mean": float(np.mean(v)),
                "sd": float(np.std(v)),
                "reps": v,
            }

    print("\n=== IDENTIFIABILITY: recovered ΔELPD by generating model ===")
    g1, g2 = list(gens)
    print(f"{'contrast':36s} {'observed':>10s} {'gen=' + g1:>18s} {'gen=' + g2:>18s}")
    for b, s in contrasts:
        k = f"{b}_minus_{s}"
        o = res["observed"][k]
        A, B = res["by_generator"][g1][k], res["by_generator"][g2][k]
        print(
            f"{k:36s} {o:+10.1f} {A['mean']:+13.1f}±{A['sd']:4.1f} {B['mean']:+13.1f}±{B['sd']:4.1f}"
        )
    _n = "identifiability_slope.json" if a.band else "identifiability.json"
    with open(os.path.join(OUT, _n), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {OUT}/{_n}")


COMMANDS = {
    "recovery": run_recovery,
    "ppc": run_ppc,
    "identifiability": run_identifiability,
}

if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.models.checks {recovery|ppc|identifiability} [args]"
        )
    _cmd = sys.argv.pop(1)
    COMMANDS[_cmd]()
