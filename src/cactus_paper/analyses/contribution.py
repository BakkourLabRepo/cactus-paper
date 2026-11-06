"""Contribution-by-level-difference profile

How much does winning one dimension by j = 1..4 taught levels push choice? Regress choice on
X_j = sum_d sign(dk_d) * 1[|dk_d| = j] (plus a side intercept) and normalize beta_j by beta_4.

  tally                        -> flat (winning by 1 counts as much as winning by 4)
  any value map, read metrically -> climbs (a j-level win is j map increments; ~ j/4 if linear)
  lambda-mixture               -> affine: a bonus for being ahead plus a climbing part
  value noise (erf)            -> concave: lifts the low end but saturates early

Nulls are SIMULATED through the real trial set and scored with the same pooled estimator as the
observed data, so design structure (which margins co-occur on which trials) cancels.

Paper artifacts (`python -m cactus_paper.analyses.contribution {readout|slope-readout}`):
  results/fits/readout_profile.json        observed vs simulated profile   (stats.json, fig_ms_model C)
  results/fits/slope_readout_profile.json  profile by learning tertile     (stats.json, fig_ms_model D)
each with a *_draws.npz of the bootstrap / simulation arrays the figures plot from.
"""

import json
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from cactus_paper.paths import FITS, MOR_IDATA

J = np.arange(1, 5)
MOR_SITES = [
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


def design(cd: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taught levels (N, 3) for each side and the margin design matrix [1, X_1..X_4]."""
    from cactus_paper.features.values import raw_values_and_diffs, taught_levels

    Vl, Vr, _ = raw_values_and_diffs(cd)
    kl, kr = taught_levels(Vl, Vr)
    dk = kr - kl
    X = np.column_stack(
        [np.ones(len(cd))] + [(np.sign(dk) * (np.abs(dk) == j)).sum(1) for j in J]
    )
    return kl, kr, X


def profile(X: np.ndarray, y: np.ndarray, rows: slice = slice(None)) -> np.ndarray:
    """beta_j / beta_4 from a pooled logistic fit (Newton IRLS, as in cactus_paper.models.checks.irls)."""
    from cactus_paper.models.checks import irls

    b = irls(X[rows], y[rows])[1:]
    return b / b[-1]


def boot_profile(
    X: np.ndarray, y: np.ndarray, sc: np.ndarray, nS: int, B: int = 500, seed: int = 1
) -> np.ndarray:
    """Participant-level (cluster) bootstrap of the observed profile, shape (B, 4)."""
    rng = np.random.default_rng(seed)
    rows = [np.flatnonzero(sc == s) for s in range(nS)]
    return np.array(
        [
            profile(X, y, np.concatenate([rows[s] for s in rng.integers(0, nS, nS)]))
            for _ in range(B)
        ]
    )


def _draw_idx(n_total, n_draws, seed):
    return np.random.default_rng(seed).choice(n_total, n_draws, replace=False)


def simulate_mor(
    cd: pd.DataFrame,
    sc: np.ndarray,
    nS: int,
    kl: np.ndarray,
    kr: np.ndarray,
    n_draws: int = 200,
    lam_one: bool = False,
    lam_zero: bool = False,
    seed: int = 2026,
    key: int = 88,
) -> np.ndarray:
    """Choices simulated from the model of record; lam_one=True forces lambda = 1 for everyone
    (each person's own fitted value map and weights, no direction channel); lam_zero=True forces
    lambda = 0 (own weights, direction channel only -- a weighted tally)."""
    assert not (lam_one and lam_zero)
    import arviz as az
    import jax
    import jax.numpy as jnp
    from numpyro.infer import Predictive

    from cactus_paper.design import within_band_position
    from cactus_paper.models.mor import mor

    pl, pr = within_band_position(cd, sc)
    po = az.from_netcdf(MOR_IDATA).posterior
    flat = {k: po[k].values.reshape(-1, *po[k].values.shape[2:]) for k in MOR_SITES}
    idx = _draw_idx(flat["a0"].shape[0], n_draws, seed)
    sub = {k: jnp.array(v[idx]) for k, v in flat.items()}
    if lam_one or lam_zero:
        sub.update(
            c0=jnp.full(n_draws, 30.0 if lam_one else -30.0),
            sigma_lam=jnp.zeros(n_draws),
        )
    return np.asarray(
        Predictive(mor, posterior_samples=sub)(
            jax.random.PRNGKey(key),
            jnp.array(sc),
            jnp.array(kl),
            jnp.array(kr),
            jnp.array(pl),
            jnp.array(pr),
            nS,
        )["obs"]
    ).astype(int)


# --------------------------------------------------------------------------- paper artifacts
READOUT_JSON = FITS / "readout_profile.json"
READOUT_DRAWS = FITS / "readout_profile_draws.npz"
SLOPE_JSON = FITS / "slope_readout_profile.json"
SLOPE_DRAWS = FITS / "slope_readout_profile_draws.npz"
SLOPE_B = 1000  # participant bootstrap replicates per tertile
SPLITS = ["slope", "slope_resid_K", "K"]


def _band(a):
    return {
        "mean": [round(float(v), 4) for v in a.mean(0)],
        "lo": [round(float(v), 4) for v in np.percentile(a, 2.5, 0)],
        "hi": [round(float(v), 4) for v in np.percentile(a, 97.5, 0)],
    }


def readout_profile(D: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Observed profile (participant bootstrap) vs map-only / model-of-record / tally-only sims.

    Returns (summary dict for readout_profile.json, arrays for the figure).
    """
    cd, sc, nS = D["cd"], D["sc"], D["nS"]
    Y = np.asarray(D["Y"]).astype(int)
    kl, kr, X = design(cd)
    obs = profile(X, Y)
    boot = boot_profile(X, Y, sc, nS)
    prof = {
        "map": np.array(
            [
                profile(X, y)
                for y in simulate_mor(cd, sc, nS, kl, kr, lam_one=True, key=89)
            ]
        ),
        "mor": np.array(
            [profile(X, y) for y in simulate_mor(cd, sc, nS, kl, kr, key=88)]
        ),
        "tally": np.array(
            [
                profile(X, y)
                for y in simulate_mor(cd, sc, nS, kl, kr, lam_zero=True, key=87)
            ]
        ),
    }
    summary = {
        "_what": "beta_j / beta_4 for j = 1..4 levels won by (fig_ms_readout panel A)",
        "_estimator": "pooled Newton IRLS logistic; observed CI = participant bootstrap (500); "
        "simulated bands = 200 posterior draws",
        "observed": {
            "mean": [round(float(v), 4) for v in obs],
            **{k: v for k, v in _band(boot).items() if k != "mean"},
        },
        "map_only": _band(prof["map"]),
        "tally_only": _band(prof["tally"]),
        "model_of_record": _band(prof["mor"]),
    }
    return summary, {"observed": obs, "boot": boot, **prof}


def lift(p: np.ndarray) -> np.ndarray:
    """Tally lift: mean contribution of 1-3 level wins relative to a 4-level win."""
    return np.asarray(p)[..., :3].mean(-1)


def _tertiles(v, X, Y, sc):
    t = np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
    out = []
    for g in range(3):
        subs = np.flatnonzero(t == g)
        rows = np.isin(sc, subs)
        boot = boot_profile(
            X[rows],
            Y[rows],
            np.searchsorted(subs, sc[rows]),
            len(subs),
            B=SLOPE_B,
            seed=g + 7,
        )
        out.append(
            {"n": int(len(subs)), "profile": profile(X[rows], Y[rows]), "boot": boot}
        )
    return out


def _tertile_summary(res):
    d = lift(res[2]["boot"]) - lift(res[0]["boot"])
    return {
        "tertiles": [
            {
                "n": r["n"],
                "profile": [round(float(x), 4) for x in r["profile"]],
                "profile_lo": [
                    round(float(x), 4) for x in np.percentile(r["boot"], 2.5, 0)
                ],
                "profile_hi": [
                    round(float(x), 4) for x in np.percentile(r["boot"], 97.5, 0)
                ],
                "lift": round(float(lift(r["profile"])), 4),
                "lift_lo": round(float(np.percentile(lift(r["boot"]), 2.5)), 4),
                "lift_hi": round(float(np.percentile(lift(r["boot"]), 97.5)), 4),
            }
            for r in res
        ],
        "high_minus_low": {
            "mean": round(float(lift(res[2]["profile"]) - lift(res[0]["profile"])), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4),
            "hi": round(float(np.percentile(d, 97.5)), 4),
            "p_pos": round(float((d > 0).mean()), 4),
        },
    }


def slope_readout_profile(
    D: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """The profile by tertile of the learning slope, the slope residualized on K, and K.

    The slope is map_slope_draws.npy's posterior mean (cactus_paper.features.learning_slope).
    Returns (summary dict for slope_readout_profile.json, arrays for the figure).
    """
    cd, sc = D["cd"], np.asarray(D["sc"])
    Y = np.asarray(D["Y"]).astype(int)
    _, _, X = design(cd)
    K = np.asarray(D["K"], float)
    slope = np.load(FITS / "map_slope_draws.npy").mean(0)
    slope_resid = slope - np.polyval(np.polyfit(K, slope, 1), K)
    splits = {
        "slope": _tertiles(slope, X, Y, sc),
        "slope_resid_K": _tertiles(slope_resid, X, Y, sc),
        "K": _tertiles(K, X, Y, sc),
    }
    summary = {
        "_what": "fig_ms_slope_readout: contribution profile by learning-phase tertile; "
        "lift = mean beta_j/beta_4 over j=1..3 (tally 1, linear map read by size 0.5)",
        "_estimator": f"pooled Newton IRLS logistic; participant bootstrap ({SLOPE_B}) within tertile",
        **{k: _tertile_summary(v) for k, v in splits.items()},
    }
    arrays = {
        f"{k}_{g}_{f}": r[f]
        for k, v in splits.items()
        for g, r in enumerate(v)
        for f in ("profile", "boot")
    }
    return summary, arrays


def _write(summary, arrays, json_path, npz_path):
    os.makedirs(FITS, exist_ok=True)
    json.dump(summary, open(json_path, "w"), indent=2)
    np.savez(npz_path, **arrays)
    print(f"saved -> {json_path}\n         {npz_path}")


def _cli_readout():
    from cactus_paper.design import load_design

    summary, arrays = readout_profile(load_design())
    _write(summary, arrays, READOUT_JSON, READOUT_DRAWS)
    print(f"observed         {arrays['observed'].round(3)}")


def _cli_slope_readout():
    from cactus_paper.design import load_design

    summary, arrays = slope_readout_profile(load_design())
    _write(summary, arrays, SLOPE_JSON, SLOPE_DRAWS)
    for k in SPLITS:
        t, h = summary[k]["tertiles"], summary[k]["high_minus_low"]
        print(
            f"{k:14s} lift "
            + " / ".join(f"{x['lift']:.3f}" for x in t)
            + f"   high-low {h['mean']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}]  P>0 {h['p_pos']:.3f}"
        )


COMMANDS = {"readout": _cli_readout, "slope-readout": _cli_slope_readout}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.analyses.contribution {readout|slope-readout}"
        )
    COMMANDS[sys.argv.pop(1)]()
