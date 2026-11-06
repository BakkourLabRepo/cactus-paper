"""Shared-folds K-fold trial-wise elpd across the model ladder.

Folds: CV_SEED=2026, rng.permutation(N) then np.array_split, on load_design's canonical N.

Models (see build_models):
  wl              models.weighted_linear: per-attribute weights x lambda on the objective
                  magnitude/sign channels
  group_map             free monotone group-level u per dimension, no lambda
  group_map_lambda           the same u + per-participant lambda blending sign(u_diff) against u_diff
  taught_spacing_lambda   increments pinned to [.25]x4 + lambda (wl in the M3 parameterization)
  group_map_by_direction         group_map / group_map_lambda with u estimated separately per counterbalance direction
  group_map_lambda_by_direction
  participant_map        participant-level increments, without / with lambda
  participant_map_lambda
  group_map_noise         noisy-value (erf) readout in place of lambda
  mor     the model of record (models.mor)

Checkpointed per model to results/fits/cv_lpd.npz; safe to interrupt and resume.

Run from repo root:
  pixi run python -m cactus_paper.models.cv [--warmup 600 --samples 600 --chains 4 --folds 5]
  pixi run python -m cactus_paper.models.cv --report-only
  pixi run python -m cactus_paper.models.cv --only wl,group_map,group_map_lambda,taught_spacing_lambda
Outputs -> results/fits/cv_lpd.npz (per-trial held-out lpd) + results/fits/cv.json
"""

import argparse
import functools
import itertools
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, log_likelihood

from cactus_paper.constants import CV_SEED, DIMS
from cactus_paper.design import build_objective_per_dim, load_design
from cactus_paper.features.geometry import choice_frame
from cactus_paper.features.values import (
    raw_values_and_diffs,
    taught_levels,
)
from cactus_paper.models.comparison import (
    group_map,
    per_participant_map,
    value_noise,
    weighted_linear,
)
from cactus_paper.models.mor import mor
from cactus_paper.paths import FITS

OUT = str(FITS)
os.makedirs(OUT, exist_ok=True)
LPD_NPZ = os.path.join(OUT, os.environ.get("CV_LPD_NAME", "cv_lpd.npz"))
JSON_OUT = os.path.join(OUT, os.environ.get("CV_JSON_NAME", "cv.json"))


def build_models(
    D: Dict[str, Any],
) -> Dict[str, Tuple[Callable[..., None], List[np.ndarray], float]]:
    """name -> (model_fn, list_of_trialwise_arrays, target_accept). Trial arrays are sliced by fold;
    nS is appended after them at call time."""
    M, Sg = build_objective_per_dim(D)
    cd, sc = D["cd"], D["sc"]
    Vl, Vr, _ = raw_values_and_diffs(cd)
    kl, kr = taught_levels(Vl, Vr)

    # within-band positions for the band_* models (ordinal channel = tanh surrogate of sign)
    _rec, _ = choice_frame(D)
    pl = np.stack([_rec[d]["pL"] for d in DIMS], 1)
    pr = np.stack([_rec[d]["pR"] for d in DIMS], 1)

    return {
        "weighted_linear": (weighted_linear, [sc, M, Sg], 0.92),
        "participant_map": (per_participant_map, [sc, kl, kr], 0.92),
        "participant_map_lambda": (
            functools.partial(per_participant_map, use_lam=True),
            [sc, kl, kr],
            0.92,
        ),
        "group_map_lambda": (
            functools.partial(group_map, use_lam=True),
            [sc, kl, kr],
            0.92,
        ),
        "group_map_noise": (value_noise, [sc, kl, kr], 0.92),
        "taught_spacing_lambda": (
            functools.partial(group_map, use_lam=True, fixed_uniform=True),
            [sc, kl, kr],
            0.92,
        ),
        "mor": (mor, [sc, kl, kr, pl, pr], 0.92),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--only", default="", help="comma-separated subset of model names to fit"
    )
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    D = load_design()
    Y, nS, div = np.asarray(D["Y"]).astype(int), D["nS"], D["div"]
    N = len(Y)
    models = build_models(D)

    rng = np.random.default_rng(CV_SEED)
    perm = rng.permutation(N)
    folds_ix = np.array_split(perm, a.folds)

    lpd = {}
    if os.path.exists(LPD_NPZ):
        Z = np.load(LPD_NPZ)
        lpd = {k: Z[k] for k in Z.files}
        print(f"resuming: found {sorted(lpd)} in {LPD_NPZ}", flush=True)

    if not a.report_only:
        want = [n.strip() for n in a.only.split(",") if n.strip()] or list(models)
        print(f"N={N} trials, nS={nS}, {a.folds} folds | fitting {want}", flush=True)
        for nm in want:
            if nm in lpd:
                print(f"[{nm}] cached — skip", flush=True)
                continue
            mdl, arrs, ta = models[nm]
            out = np.zeros(N)
            t0 = time.time()
            for fi, te in enumerate(folds_ix):
                tr = np.setdiff1d(perm, te)
                args_tr = [jnp.array(x[tr]) for x in arrs] + [nS]
                args_te = [jnp.array(x[te]) for x in arrs] + [nS]
                m = MCMC(
                    NUTS(mdl, target_accept_prob=ta),
                    num_warmup=a.warmup,
                    num_samples=a.samples,
                    num_chains=a.chains,
                    progress_bar=False,
                )
                m.run(jax.random.PRNGKey(CV_SEED + fi), *args_tr, y=jnp.array(Y[tr]))
                ll = log_likelihood(mdl, m.get_samples(), *args_te, y=jnp.array(Y[te]))[
                    "obs"
                ]
                out[te] = np.asarray(logsumexp(ll, 0) - np.log(ll.shape[0]))
                print(
                    f"  [{nm}] fold {fi + 1}/{a.folds}  elpd={out[te].sum():+.1f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                del m, ll
            lpd[nm] = out
            np.savez(LPD_NPZ, **lpd)
            print(f"[{nm}] total elpd = {out.sum():+.1f}  (checkpointed)", flush=True)

    # ---------------------------------------------------------------- report
    se = lambda x: float(x.std() * np.sqrt(len(x)))
    res = {
        "n_trials": N,
        "folds": a.folds,
        "seed": CV_SEED,
        "warmup": a.warmup,
        "samples": a.samples,
        "chains": a.chains,
        "elpd": {nm: float(v.sum()) for nm, v in lpd.items()},
        "contrasts": {},
    }
    print(f"\n=== elpd ({a.folds}-fold, shared folds, {N:,} held-out trials) ===")
    for nm, v in sorted(lpd.items(), key=lambda kv: -kv[1].sum()):
        print(f"  {nm:16s} {v.sum():+10.1f}")
    print("\n=== pairwise ΔELPD (paired per-trial, ± SE) ===")
    for b, s in itertools.permutations(sorted(lpd), 2):
        if (b, s) not in [
            ("group_map_lambda", "taught_spacing_lambda"),
            ("group_map_lambda", "weighted_linear"),
            ("taught_spacing_lambda", "weighted_linear"),
            ("group_map_lambda", "participant_map"),
            ("participant_map", "weighted_linear"),
            ("participant_map_lambda", "group_map_lambda"),
            ("participant_map_lambda", "participant_map"),
            ("participant_map_lambda", "weighted_linear"),
            ("group_map_lambda", "group_map_noise"),
            ("group_map_noise", "weighted_linear"),
            ("group_map_noise", "participant_map"),
            ("mor", "participant_map_lambda"),
        ]:
            continue
        d = lpd[b] - lpd[s]
        res["contrasts"][f"{b}_minus_{s}"] = {
            "delta": float(d.sum()),
            "se": se(d),
            "z": float(d.sum() / se(d)),
            "delta_div": float(d[div].sum()),
            "delta_nondiv": float(d[~div].sum()),
        }
        print(
            f"  {b:16s} - {s:16s} = {d.sum():+8.1f} ± {se(d):5.1f}  ({d.sum() / se(d):+5.1f} SE)"
            f"  | div {d[div].sum():+7.1f} / non-div {d[~div].sum():+7.1f}"
        )
    with open(JSON_OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {JSON_OUT}")


if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    main()
