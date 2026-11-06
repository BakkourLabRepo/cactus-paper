"""Robustness: split-half reliability, the pointer-device refits, and the exclusion-rule sweep.

  refits           --splithalf | --device | --device-band  -> results/fits/{splithalf,
                   group_map_lambda_device,mor_device}.json
  exclusion-sweep  refit the main results under alternative exclusion rules
                   -> results/fits/exclusion_sweep.json

Run from repo root: pixi run python -m cactus_paper.analyses.robustness {refits,exclusion-sweep} [args]
"""

import argparse
import functools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood

from cactus_paper.constants import CV_SEED, DIMS, FULL_RAMP
from cactus_paper.data.device import device_codes
from cactus_paper.data.sample import SAMPLE_OVERRIDE_ENV, TEST_RUN_IDS, canonical_sample
from cactus_paper.design import load_design
from cactus_paper.features.geometry import choice_frame
from cactus_paper.features.sampling import trial_visit_feats
from cactus_paper.features.traits import gain as wl_gain
from cactus_paper.features.values import raw_values_and_diffs, taught_levels
from cactus_paper.models.comparison import group_map
from cactus_paper.models.mor import mor
from cactus_paper.paths import DATA, FITS, RAW_EXP2
from cactus_paper.stats import summarize

OUT = str(FITS)


def group_map_device(
    sc: np.ndarray,
    kl: np.ndarray,
    kr: np.ndarray,
    dev: np.ndarray,
    n: int,
    y: Optional[np.ndarray] = None,
) -> None:
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    sda = numpyro.sample("sda", dist.HalfNormal(2.5))
    w_mu = numpyro.sample("w_mu", dist.Normal(jnp.ones(3), 1.0).to_event(1))
    w_dev = numpyro.sample("w_dev", dist.Normal(0.0, 1.0))
    w_sd = numpyro.sample("w_sd", dist.HalfNormal(1.0))
    psi_free = numpyro.sample(
        "psi_free", dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2)
    )
    psi = jnp.concatenate([jnp.zeros((3, 1)), psi_free], 1)
    u = jnp.concatenate([jnp.zeros((3, 1)), jnp.cumsum(jax.nn.softmax(psi, -1), -1)], 1)
    c0 = numpyro.sample("c0", dist.Normal(0.0, 1.5))
    c_dev = numpyro.sample("c_dev", dist.Normal(0.0, 1.0))
    slam = numpyro.sample("sigma_lam", dist.HalfNormal(1.0))
    with numpyro.plate("subj", n):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("z_w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        zl = numpyro.sample("z_lam", dist.Normal(0.0, 1.0))
    W = numpyro.deterministic("W", w_mu + w_dev * dev[:, None] + w_sd * zw)
    lam = numpyro.deterministic("lam_s", jax.nn.sigmoid(c0 + c_dev * dev + slam * zl))
    du = jnp.stack([u[d, kr[:, d]] for d in range(3)], 1) - jnp.stack(
        [u[d, kl[:, d]] for d in range(3)], 1
    )
    veff = (1 - lam[sc])[:, None] * jnp.sign(du) + lam[sc][:, None] * du
    numpyro.sample(
        "obs", dist.Bernoulli(logits=a0 + a[sc] * sda + (W[sc] * veff).sum(-1)), obs=y
    )


def run(
    mdl: Callable[..., Any],
    args: Sequence[Any],
    y: Optional[np.ndarray],
    seed: int,
    warmup: int = 1500,
    samples: int = 1500,
    **model_kw: Any,
) -> az.InferenceData:
    m = MCMC(
        NUTS(mdl, target_accept_prob=0.92),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=4,
        progress_bar=False,
    )
    m.run(jax.random.PRNGKey(seed), *args, y=y, **model_kw)
    return az.from_numpyro(m)


def run_refits() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splithalf", action="store_true")
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--device-band", dest="device_band", action="store_true")
    a = ap.parse_args()
    D = load_design()
    nS, cd, sc = D["nS"], D["cd"], D["sc"]
    Vl, Vr, _ = raw_values_and_diffs(cd)
    kl, kr = taught_levels(Vl, Vr)
    Y = np.asarray(D["Y"]).astype(int)
    K = np.asarray(D["K"], float)

    if a.splithalf:
        within = (
            cd.groupby("run_id").cumcount().values
        )  # trial order within participant
        halves = {}
        for nm, msk in [("odd", within % 2 == 1), ("even", within % 2 == 0)]:
            ida = run(
                functools.partial(group_map, use_lam=True),
                [jnp.array(sc[msk]), jnp.array(kl[msk]), jnp.array(kr[msk]), nS],
                jnp.array(Y[msk]),
                seed=501 if nm == "odd" else 502,
            )
            halves[nm] = {
                "lam": ida.posterior["lam_s"].values.reshape(-1, nS).mean(0),
                "gain": wl_gain(
                    np.abs(ida.posterior["W"].values.reshape(-1, nS, 3))
                ).mean(0),
                "frac": ida.posterior["frac"].values.reshape(-1, 3, 4).mean(0),
            }
            print(f"  [{nm}] n_trials={int(msk.sum())}", flush=True)
        sb = lambda r: 2 * r / (1 + r)
        res = {}
        for k in ("lam", "gain"):
            r = float(np.corrcoef(halves["odd"][k], halves["even"][k])[0, 1])
            res[k] = {"r_odd_even": round(r, 3), "spearman_brown": round(sb(r), 3)}
            print(f"  {k:5s} r={r:+.3f}  SB={sb(r):.3f}")
        fo, fe = halves["odd"]["frac"], halves["even"]["frac"]
        res["value_function"] = {
            "r_increments_odd_even": round(
                float(np.corrcoef(fo.ravel(), fe.ravel())[0, 1]), 3
            ),
            "max_abs_diff": round(float(np.abs(fo - fe).max()), 3),
            "odd": fo.round(3).tolist(),
            "even": fe.round(3).tolist(),
        }
        print(
            f"  value fn: r={res['value_function']['r_increments_odd_even']:+.3f}  "
            f"max|diff|={res['value_function']['max_abs_diff']:.3f}"
        )
        json.dump(res, open(f"{OUT}/splithalf.json", "w"), indent=2)
        print(f"saved -> {OUT}/splithalf.json")

    if a.device:
        su = cd.run_id.drop_duplicates().tolist()
        dev = np.asarray(device_codes(su), float)
        ida = run(
            group_map_device,
            [jnp.array(sc), jnp.array(kl), jnp.array(kr), jnp.array(dev), nS],
            jnp.array(Y),
            seed=503,
        )
        po = ida.posterior
        g = lambda n: po[n].values.ravel()
        res = {"n_trackpad": int((dev > 0).sum()), "n_mouse": int((dev < 0).sum())}
        for n in ("c_dev", "w_dev", "w_sd", "sigma_lam", "c0"):
            v = g(n)
            lo, hi = az.hdi(v, hdi_prob=0.95)
            res[n] = {
                "mean": round(float(v.mean()), 3),
                "lo": round(float(lo), 3),
                "hi": round(float(hi), 3),
            }
            print(f"  {n:10s} {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        W = np.abs(po["W"].values.reshape(-1, nS, 3)).mean(0)
        lam = po["lam_s"].values.reshape(-1, nS).mean(0)
        res["corr_gain_K_adj"] = round(float(np.corrcoef(wl_gain(W), K)[0, 1]), 3)
        res["corr_lam_K_adj"] = round(float(np.corrcoef(lam, K)[0, 1]), 3)
        res["max_rhat"] = round(
            float(
                az.rhat(ida, var_names=["a0", "sda", "w_mu", "w_sd", "c_dev", "w_dev"])
                .to_array()
                .max()
            ),
            4,
        )
        res["n_divergences"] = int(ida.sample_stats["diverging"].values.sum())
        print(
            f"  corr(gain,K)|dev {res['corr_gain_K_adj']:+.3f}   "
            f"corr(lam,K)|dev {res['corr_lam_K_adj']:+.3f}   "
            f"R-hat {res['max_rhat']}  div {res['n_divergences']}"
        )
        ida.to_netcdf(f"{OUT}/group_map_lambda_device_idata.nc")
        json.dump(res, open(f"{OUT}/group_map_lambda_device.json", "w"), indent=2)
        print(f"saved -> {OUT}/device.json")

    if a.device_band:
        rec, su = choice_frame(D)
        dev = np.asarray(device_codes(su), float)
        pl = np.stack([rec[d]["pL"] for d in DIMS], 1)
        pr = np.stack([rec[d]["pR"] for d in DIMS], 1)
        ida = run(
            mor,
            [
                jnp.array(sc),
                jnp.array(kl),
                jnp.array(kr),
                jnp.array(pl),
                jnp.array(pr),
                nS,
            ],
            jnp.array(Y),
            seed=504,
            device=jnp.array(dev),
        )
        po = ida.posterior
        g = lambda n: po[n].values.ravel()
        res = {
            "_model": "mor + device (model of record)",
            "n_trackpad": int((dev > 0).sum()),
            "n_mouse": int((dev < 0).sum()),
        }
        for n in ("c_dev", "w_dev", "w_sd", "sigma_lam", "c0"):
            v = g(n)
            lo, hi = az.hdi(v, hdi_prob=0.95)
            res[n] = {
                "mean": round(float(v.mean()), 3),
                "lo": round(float(lo), 3),
                "hi": round(float(hi), 3),
            }
            print(f"  {n:10s} {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        # within-band slope under the device control
        Sb = po["s_band"].values.reshape(-1, 3)
        res["s_band"] = {
            d: {
                "mean": round(float(Sb[:, j].mean()), 4),
                "lo": round(float(az.hdi(Sb[:, j], hdi_prob=0.95)[0]), 4),
                "hi": round(float(az.hdi(Sb[:, j], hdi_prob=0.95)[1]), 4),
            }
            for j, d in enumerate(DIMS)
        }
        W = np.abs(po["W"].values.reshape(-1, nS, 3)).mean(0)
        lam = po["lam_s"].values.reshape(-1, nS).mean(0)
        res["corr_gain_K_adj"] = round(float(np.corrcoef(wl_gain(W), K)[0, 1]), 3)
        res["corr_lam_K_adj"] = round(float(np.corrcoef(lam, K)[0, 1]), 3)
        res["max_rhat"] = round(
            float(
                az.rhat(
                    ida,
                    var_names=["a0", "sda", "w_mu", "w_sd", "s_band", "c_dev", "w_dev"],
                )
                .to_array()
                .max()
            ),
            4,
        )
        res["n_divergences"] = int(ida.sample_stats["diverging"].values.sum())
        print(
            f"  corr(gain,K)|dev {res['corr_gain_K_adj']:+.3f}   "
            f"corr(lam,K)|dev {res['corr_lam_K_adj']:+.3f}   "
            f"R-hat {res['max_rhat']}  div {res['n_divergences']}",
            flush=True,
        )
        ida.to_netcdf(f"{OUT}/mor_device_idata.nc")
        json.dump(res, open(f"{OUT}/mor_device.json", "w"), indent=2)
        print(f"saved -> {OUT}/mor_device.json")


JSON_OUT = os.path.join(OUT, "exclusion_sweep.json")


SCRATCH = os.path.join(OUT, "_sweep_samples")


CV_MODELS = ["participant_map", "participant_map_lambda", "mor"]


CONTRASTS = [
    ("participant_map_lambda", "participant_map"),
    ("mor", "participant_map_lambda"),
]


def summ(d: Any, nd: int = 4) -> Dict[str, Any]:
    return summarize(d, nd=nd, direction="pdir")


# --------------------------------------------------------------------------- samples
def prereg_learning_failures(final_block_only: bool) -> Tuple[Set[str], pd.DataFrame]:
    """run_ids below 60% on ANY dimension, per the preregistered learning-failure rule.

    final_block_only=True scores the last 10-trial prompt run of each dimension; False scores
    every learning trial of the dimension (as cactus_paper.analyses.prereg does).
    """
    ld = pd.read_csv(os.path.join(DATA, "learning_data.csv"))
    ld = ld.sort_values(["run_id", "dim", "trial_per_dim"])
    if final_block_only:

        def last_run(g):
            p = g.prompt_dir.values
            blk = np.concatenate([[0], (p[1:] != p[:-1]).cumsum()])
            return g[blk == blk.max()]

        ld = ld.groupby(["run_id", "dim"], group_keys=False).apply(last_run)
    acc = ld.groupby(["run_id", "dim"]).correct.mean().unstack()
    return set(acc.index[(acc < 0.60).any(axis=1)].values.tolist()), acc


def prereg_cascade() -> np.ndarray:
    """The prereg's own cascade, as implemented in cactus_paper.analyses.prereg.audit_exclusions."""
    raw = pd.read_csv(
        RAW_EXP2,
        low_memory=False,
        usecols=["run_id", "trial_type", "task"],
    )
    fv = pd.read_csv(os.path.join(DATA, "first_visits.csv"))
    allids = np.sort(raw.run_id.unique())
    n_choice = (
        raw[raw.trial_type.isin(["divergent", "non-divergent"])]
        .groupby("run_id")
        .size()
    )
    ended = raw[raw.task.eq("experiment_complete_questionnaire")].run_id.unique()
    incomplete = np.array(
        [i for i in allids if n_choice.get(i, 0) < 120 or i not in ended]
    )
    surv = np.setdiff1d(allids, np.union1d(np.array(TEST_RUN_IDS), incomplete))
    fail, _ = prereg_learning_failures(final_block_only=False)
    surv2 = np.setdiff1d(surv, np.array(sorted(fail), dtype=surv.dtype))
    nf = (
        fv[fv.run_id.isin(surv2)]
        .assign(n=lambda x: x.first_pos.eq("none"))
        .groupby("run_id")
        .n.mean()
    )
    return np.setdiff1d(surv2, nf.index[nf > 0.05].values)


def build_samples() -> Dict[str, Tuple[np.ndarray, str]]:
    """tag -> (sorted run_id array, one-line description). Computed with the override CLEARED."""
    os.environ.pop(SAMPLE_OVERRIDE_ENV, None)
    base = canonical_sample(DATA)
    fail_final, _ = prereg_learning_failures(final_block_only=True)
    fail_whole, _ = prereg_learning_failures(final_block_only=False)
    keep = lambda ids: np.array(sorted(set(base.tolist()) - set(ids)), dtype=base.dtype)
    S = {
        "applied": (
            base,
            "as reported: mouse >5% non-compliance, >20% of learning RTs below p10",
        ),
        "mouse_strict": (
            canonical_sample(DATA, mouse_none_thresh=0.02),
            "mouse non-compliance threshold 5% -> 2%",
        ),
        "mouse_loose": (
            canonical_sample(DATA, mouse_none_thresh=0.10),
            "mouse non-compliance threshold 5% -> 10%",
        ),
        "rt_strict": (
            canonical_sample(DATA, rt_frac=0.10),
            "fast-RT rule 20% -> 10% of learning trials below p10",
        ),
        "rt_loose": (
            canonical_sample(DATA, rt_frac=0.30),
            "fast-RT rule 20% -> 30% of learning trials below p10",
        ),
        "no_rt_rule": (
            canonical_sample(DATA, rt_frac=1.01),
            "the unregistered fast-RT rule dropped entirely",
        ),
        "prereg_learn_final": (
            keep(fail_final),
            "applied sample + the preregistered learning rule, scored on the "
            "final 10-trial prompt run (the prereg's wording)",
        ),
        "prereg_learn_whole": (
            keep(fail_whole),
            "applied sample + the preregistered learning rule, scored on all "
            "~39 learning trials (what cactus_paper.analyses.prereg does)",
        ),
        "prereg_cascade": (
            prereg_cascade(),
            "the preregistered cascade in full, replacing the applied one",
        ),
    }
    return S


# --------------------------------------------------------------------------- the three results
def fit_mor(
    D: Dict[str, Any], rec: Dict[str, Dict[str, np.ndarray]], warmup: int, samples: int
) -> az.InferenceData:
    """Full refit of the model of record (mor). Returns (idata, per-dim s_band draws)."""
    from cactus_paper.models.mor import mor

    sc = jnp.array(D["sc"])
    Y = jnp.array(np.asarray(D["Y"]).astype(int))
    arrs = [
        jnp.array(np.stack([rec[d][k] for d in DIMS], 1))
        for k in ("kL", "kR", "pL", "pR")
    ]
    mdl = mor
    m = MCMC(
        NUTS(mdl, target_accept_prob=0.92),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=4,
        progress_bar=False,
    )
    m.run(jax.random.PRNGKey(621), sc, *arrs, D["nS"], y=Y)
    return az.from_numpyro(m)


def slope_block(ida: az.InferenceData) -> Dict[str, Any]:
    po = ida.posterior
    S = po["s_band"].values.reshape(-1, 3)
    out = {
        "max_rhat": round(
            float(
                az.rhat(ida, var_names=["a0", "sda", "w_mu", "w_sd", "s_band"])
                .to_array()
                .max()
            ),
            4,
        ),
        "n_divergences": int(ida.sample_stats["diverging"].values.sum()),
    }
    for j, d in enumerate(DIMS):
        b = summ(S[:, j])
        b["frac_of_full_ramp"] = round(float(b["mean"] / FULL_RAMP), 3)
        out[d] = b
    p_ = S.mean(1)
    b = summ(p_)
    b.update(
        frac_of_full_ramp=round(float(b["mean"] / FULL_RAMP), 3),
        frac_lo=round(float(b["lo"] / FULL_RAMP), 3),
        frac_hi=round(float(b["hi"] / FULL_RAMP), 3),
    )
    out["pooled"] = b
    return out


def run_cv(D: Dict[str, Any], folds: int, warmup: int, samples: int) -> Dict[str, Any]:
    """5-fold trial-wise CV over CV_MODELS on this sample's trials, folds shared across models."""
    from cactus_paper.models.cv import build_models

    models = build_models(D)
    Y, nS, div = np.asarray(D["Y"]).astype(int), D["nS"], D["div"]
    N = len(Y)
    perm = np.random.default_rng(CV_SEED).permutation(N)
    folds_ix = np.array_split(perm, folds)

    lpd = {}
    for nm in CV_MODELS:
        mdl, arrs, ta = models[nm]
        out = np.zeros(N)
        t0 = time.time()
        for fi, te in enumerate(folds_ix):
            tr = np.setdiff1d(perm, te)
            args_tr = [jnp.array(x[tr]) for x in arrs] + [nS]
            args_te = [jnp.array(x[te]) for x in arrs] + [nS]
            m = MCMC(
                NUTS(mdl, target_accept_prob=ta),
                num_warmup=warmup,
                num_samples=samples,
                num_chains=4,
                progress_bar=False,
            )
            m.run(jax.random.PRNGKey(CV_SEED + fi), *args_tr, y=jnp.array(Y[tr]))
            ll = log_likelihood(mdl, m.get_samples(), *args_te, y=jnp.array(Y[te]))[
                "obs"
            ]
            out[te] = np.asarray(logsumexp(ll, 0) - np.log(ll.shape[0]))
            del m, ll
        lpd[nm] = out
        print(
            f"    cv[{nm:12s}] elpd {out.sum():+10.1f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    res = {
        "n_trials": int(N),
        "folds": folds,
        "elpd": {k: round(float(v.sum()), 1) for k, v in lpd.items()},
        "contrasts": {},
    }
    for b, s in CONTRASTS:
        d = lpd[b] - lpd[s]
        se = float(d.std() * np.sqrt(len(d)))
        res["contrasts"][f"{b}_minus_{s}"] = {
            "delta": round(float(d.sum()), 1),
            "se": round(se, 1),
            "z": round(float(d.sum() / se), 2),
            "delta_per_1k_trials": round(float(d.sum() / N * 1000), 2),
            "delta_div": round(float(d[div].sum()), 1),
            "delta_nondiv": round(float(d[~div].sum()), 1),
        }
    return res


def knowledge_block(D: Dict[str, Any], ida: az.InferenceData) -> Dict[str, Any]:
    """The knowledge couplings, propagated over posterior draws.

    Same estimators as make_stats.py's `knowledge` block.
    """
    nS = D["nS"]
    dacc = D["dacc"]
    K = np.asarray(D["K"], float)
    po = ida.posterior
    WD = np.abs(po["W"].values.reshape(-1, nS, 3))
    lamD = po["lam_s"].values.reshape(-1, nS)
    gainD = wl_gain(WD)
    dc2 = lambda M: M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    dcD = dc2(dacc).ravel()
    return {
        "n_subj_dim": int(nS * 3),
        "corr_gain_K": summ(
            [np.corrcoef(gainD[d], K)[0, 1] for d in range(len(gainD))], 3
        ),
        "corr_lam_K": summ(
            [np.corrcoef(lamD[d], K)[0, 1] for d in range(len(lamD))], 3
        ),
        "corr_weight_dacc_double_centered_propagated": summ(
            [np.corrcoef(dc2(WD[d]).ravel(), dcD)[0, 1] for d in range(len(WD))], 3
        ),
    }


def lam_to_payne(D: Dict[str, Any], ida: az.InferenceData) -> Dict[str, Any]:
    """`payne ~ lam + gain` on the refit's latents."""
    import bambi as bmb

    nS = D["nS"]
    su = D["cd"].run_id.drop_duplicates().tolist()
    po = ida.posterior
    lam_s = po["lam_s"].values.reshape(-1, nS).mean(0)
    gain = wl_gain(np.abs(po["W"].values.reshape(-1, nS, 3).mean(0)))

    dc = pd.read_csv(os.path.join(DATA, "dim_counts.csv"))
    payne = (
        dc[dc.run_id.isin(su)].groupby("run_id").payne_index.mean().reindex(su).values
    )
    zc1 = lambda v: (np.asarray(v, float) - np.nanmean(v)) / np.nanstd(v)
    sub = pd.DataFrame(dict(payne=payne, lam=zc1(lam_s), gain=zc1(gain))).dropna()
    fit = bmb.Model("payne ~ lam + gain", sub, family="gaussian").fit(
        draws=2000,
        tune=2000,
        chains=4,
        inference_method="nutpie",
        random_seed=20260821,
        progressbar=False,
    )
    return {
        "n": int(len(sub)),
        **{k: summ(fit.posterior[k].values, 3) for k in ("lam", "gain")},
    }


# --------------------------------------------------------------------------- driver
def run_exclusion_sweep() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of sample tags")
    ap.add_argument(
        "--quick", action="store_true", help="short chains; for wiring checks only"
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--knowledge-only",
        action="store_true",
        help="refit the MoR per sample and (re)compute only the knowledge block, "
        "leaving the CV results in place. Added after the first full run, which "
        "did not save per-sample idata.",
    )
    a = ap.parse_args()
    mor_w, mor_s = (200, 200) if a.quick else (1500, 1500)
    cv_w, cv_s = (150, 150) if a.quick else (600, 600)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    samples = build_samples()
    want = [t.strip() for t in a.only.split(",") if t.strip()] or list(samples)

    res = json.load(open(JSON_OUT)) if os.path.exists(JSON_OUT) else {}
    res.setdefault("_meta", {})
    res["_meta"].update(
        {
            "full_ramp_benchmark": round(FULL_RAMP, 4),
            "fold_seed": CV_SEED,
            "mor_draws": f"{mor_w} warmup / {mor_s} draws x 4 chains",
            "cv_draws": f"{cv_w} warmup / {cv_s} draws x 4 chains, {a.folds} folds",
            "quick": bool(a.quick),
            "_warning": (
                "delta-elpd sums over trials and the samples differ in trial count; compare "
                "rows on delta_per_1k_trials or z, never on raw delta"
            ),
        }
    )
    res.setdefault("samples", {})
    res["_meta"]["sample_sizes"] = {t: int(len(v[0])) for t, v in samples.items()}

    print(f"\n{'tag':22s} {'N':>5s}  description")
    for t, (ids, desc) in samples.items():
        print(f"{t:22s} {len(ids):5d}  {desc}")
    print()

    for tag in want:
        if tag in res["samples"] and not a.quick and not a.knowledge_only:
            print(f"[{tag}] cached — skip", flush=True)
            continue
        if a.knowledge_only and "knowledge" in res["samples"].get(tag, {}):
            print(f"[{tag}] knowledge cached — skip", flush=True)
            continue
        ids, desc = samples[tag]
        path = os.path.join(SCRATCH, f"{tag}.txt")
        np.savetxt(path, np.asarray(ids, dtype=int), fmt="%d")
        os.environ[SAMPLE_OVERRIDE_ENV] = path

        print(f"=== [{tag}] N={len(ids)} — {desc}", flush=True)
        t0 = time.time()
        from cactus_paper.features.geometry import choice_frame

        D = load_design()
        rec, _ = choice_frame(D)
        assert D["nS"] == len(ids), f"{tag}: load_design kept {D['nS']} of {len(ids)}"

        ida = fit_mor(D, rec, mor_w, mor_s)

        if a.knowledge_only:
            blk = res["samples"][tag]
            blk["knowledge"] = knowledge_block(D, ida)
            k = blk["knowledge"]
            print(
                f"    gain~K {k['corr_gain_K']['mean']:+.3f} "
                f"[{k['corr_gain_K']['lo']:+.3f}, {k['corr_gain_K']['hi']:+.3f}]   "
                f"W~dacc(2x-centered) "
                f"{k['corr_weight_dacc_double_centered_propagated']['mean']:+.3f} "
                f"[{k['corr_weight_dacc_double_centered_propagated']['lo']:+.3f}, "
                f"{k['corr_weight_dacc_double_centered_propagated']['hi']:+.3f}]   "
                f"lam~K {k['corr_lam_K']['mean']:+.3f} "
                f"[{k['corr_lam_K']['lo']:+.3f}, {k['corr_lam_K']['hi']:+.3f}]",
                flush=True,
            )
            res["samples"][tag] = blk
            json.dump(res, open(JSON_OUT, "w"), indent=2)
            print(
                f"    [{tag}] knowledge done in {time.time() - t0:.0f}s\n", flush=True
            )
            continue

        blk = {
            "_description": desc,
            "n_subjects": int(D["nS"]),
            "n_trials": int(len(D["Y"])),
            "within_band_slope": slope_block(ida),
        }
        print(
            f"    slope pooled {blk['within_band_slope']['pooled']['mean']:+.4f} "
            f"= {blk['within_band_slope']['pooled']['frac_of_full_ramp']:.2f} x full ramp",
            flush=True,
        )

        blk["knowledge"] = knowledge_block(D, ida)
        blk["lam_to_payne"] = lam_to_payne(D, ida)
        print(
            f"    lam->Payne {blk['lam_to_payne']['lam']['mean']:+.3f} "
            f"[{blk['lam_to_payne']['lam']['lo']:+.3f}, "
            f"{blk['lam_to_payne']['lam']['hi']:+.3f}]",
            flush=True,
        )

        blk["cv"] = run_cv(D, a.folds, cv_w, cv_s)
        blk["_seconds"] = round(time.time() - t0, 1)

        res["samples"][tag] = blk
        json.dump(res, open(JSON_OUT, "w"), indent=2)
        print(
            f"    [{tag}] done in {blk['_seconds']:.0f}s — checkpointed\n", flush=True
        )

    os.environ.pop(SAMPLE_OVERRIDE_ENV, None)
    report(res)
    json.dump(res, open(JSON_OUT, "w"), indent=2)
    print(f"\nsaved -> {os.path.relpath(JSON_OUT)}")


def report(res: Dict[str, Any]) -> None:
    S = res["samples"]
    print("\n=== gradedness: within-band slope as a fraction of a fully graded map ===")
    print(f"{'sample':22s} {'N':>4s} {'trials':>7s}  {'pooled frac':>22s}")
    for t, b in S.items():
        p = b["within_band_slope"]["pooled"]
        print(
            f"{t:22s} {b['n_subjects']:4d} {b['n_trials']:7d}  "
            f"{p['frac_of_full_ramp']:6.2f} [{p['frac_lo']:5.2f}, {p['frac_hi']:5.2f}]"
        )

    for b_, s_ in CONTRASTS:
        key = f"{b_}_minus_{s_}"
        print(f"\n=== {key} ===")
        print(
            f"{'sample':22s} {'delta':>9s} {'SE':>6s} {'z':>7s} {'per 1k trials':>14s}"
        )
        for t, b in S.items():
            c = b["cv"]["contrasts"][key]
            print(
                f"{t:22s} {c['delta']:+9.1f} {c['se']:6.1f} {c['z']:+7.2f} "
                f"{c['delta_per_1k_trials']:+14.2f}"
            )

    if all("knowledge" in b for b in S.values()):
        print("\n=== knowledge couplings (propagated) ===")
        print(
            f"{'sample':22s} {'gain ~ K':>24s} {'W ~ dacc (2x-centered)':>26s} "
            f"{'lam ~ K':>24s}"
        )
        for t, b in S.items():
            k = b["knowledge"]
            f = lambda n: f"{k[n]['mean']:+.3f} [{k[n]['lo']:+.3f}, {k[n]['hi']:+.3f}]"
            print(
                f"{t:22s} {f('corr_gain_K'):>24s} "
                f"{f('corr_weight_dacc_double_centered_propagated'):>26s} "
                f"{f('corr_lam_K'):>24s}"
            )

    print("\n=== lambda -> Payne  (payne ~ lam + gain) ===")
    print(f"{'sample':22s} {'n':>4s}  {'lam':>26s}  {'P(dir)':>7s}")
    for t, b in S.items():
        v = b["lam_to_payne"]["lam"]
        print(
            f"{t:22s} {b['lam_to_payne']['n']:4d}  "
            f"{v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]".rjust(26)
            + f"  {v['pdir']:7.3f}"
        )


COMMANDS = {"refits": run_refits, "exclusion-sweep": run_exclusion_sweep}

if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.analyses.robustness {refits|exclusion-sweep} [args]"
        )
    _cmd = sys.argv.pop(1)
    COMMANDS[_cmd]()
