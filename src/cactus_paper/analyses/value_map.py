"""Value-map tests on the learning and choice data, the counterbalance check, and controls.

  tests           within-band position -> choice, angular distance and boundary distance in
                  learning -> results/fits/value_map.json
  controls        residualized angular distance; raw angle by mapping direction
                  -> results/fits/value_map_controls.json
  counterbalance  increment profile fit separately by mapping direction
                  -> results/fits/counterbalance.json

Run from repo root: pixi run python -m cactus_paper.analyses.value_map {tests,controls,counterbalance}
"""

import functools
import json
import os
import sys
import zlib
from typing import Any, Dict, List

import arviz as az
import bambi as bmb
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import MCMC, NUTS

from cactus_paper.constants import ARC, BAND_WIDTH, DIMS, FULL_RAMP, POS
from cactus_paper.data.sample import canonical_sample
from cactus_paper.design import load_design
from cactus_paper.features.geometry import (
    band_arrays,
    choice_frame,
    get_directions,
)
from cactus_paper.features.values import raw_values_and_diffs
from cactus_paper.models.comparison import group_map
from cactus_paper.paths import DATA, FITS
from cactus_paper.stats import summarize

OUT = str(FITS)


os.makedirs(OUT, exist_ok=True)


FIT = dict(
    draws=1500, tune=1500, chains=4, inference_method="nutpie", random_seed=20260908
)


def summ(d: np.ndarray) -> Dict[str, Any]:
    return summarize(d, nd=4, direction="pdir")


def learning_angular_distance(su: List[str]) -> Dict[str, Any]:
    """Learning accuracy ~ value distance + angular distance."""
    ld = pd.read_csv(f"{DATA}/learning_data.csv")
    ld = (
        ld[ld.run_id.isin(su)]
        .dropna(
            subset=["left_choice", "right_choice", "left_rew", "right_rew", "correct"]
        )
        .copy()
    )
    ld["vdist"] = (ld.left_rew - ld.right_rew).abs() / 25.0
    ld["adist"] = (ld.left_choice - ld.right_choice).abs()
    r = float(np.corrcoef(ld.vdist, ld.adist)[0, 1])
    print(
        f"[3.1] n={len(ld)}  corr(value dist, angular dist) = {r:+.3f}"
        f"  (VIF = {1 / (1 - r**2):.2f})",
        flush=True,
    )
    ld["vd_z"] = (ld.vdist - ld.vdist.mean()) / ld.vdist.std()
    ld["ad_z"] = (ld.adist - ld.adist.mean()) / ld.adist.std()
    ld["correct"] = ld.correct.astype(int)
    m = bmb.Model(
        "correct ~ vd_z + ad_z + C(dim) + (1 + vd_z + ad_z | run_id)",
        ld,
        family="bernoulli",
    ).fit(**FIT)
    out = {
        "_n": int(len(ld)),
        "collinearity_r": round(r, 3),
        "vif": round(1 / (1 - r**2), 2),
        "value_distance": summ(m.posterior["vd_z"].values),
        "angular_distance": summ(m.posterior["ad_z"].values),
    }
    print(
        f"      value dist {out['value_distance']['mean']:+.3f} "
        f"[{out['value_distance']['lo']:+.3f},{out['value_distance']['hi']:+.3f}]"
    )
    print(
        f"      ANGULAR    {out['angular_distance']['mean']:+.3f} "
        f"[{out['angular_distance']['lo']:+.3f},{out['angular_distance']['hi']:+.3f}]"
    )
    return out


def within_band_choice(
    D: Dict[str, Any], rec: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Any]:
    """Within-band position on NON-DISCRIMINATING dimensions -> choice."""
    Y = np.asarray(D["Y"]).astype(int)
    sc = D["sc"]
    dp_same = np.zeros(len(Y))
    dp_all = np.zeros(len(Y))
    n_same = np.zeros(len(Y))
    vmag = np.zeros(len(Y))
    vtal = np.zeros(len(Y))
    for d in DIMS:
        r = rec[d]
        dv = (r["kR"] - r["kL"]).astype(float)
        vmag += dv
        vtal += np.sign(dv)
        dp = r["pR"] - r["pL"]
        dp_all += dp
        same = dv == 0
        dp_same += np.where(same, dp, 0.0)
        n_same += same
    z = lambda v: (v - v.mean()) / v.std()
    df = pd.DataFrame(
        {
            "y": Y,
            "vmag": z(vmag),
            "vtal": z(vtal),
            "dp_same": z(dp_same),
            "dp_all": z(dp_all),
            "subject": sc.astype(str),
        }
    )
    print(
        f"[3.2a] trials with >=1 non-discriminating dim: {int((n_same > 0).sum())} / {len(Y)}",
        flush=True,
    )
    m = bmb.Model(
        "y ~ vmag + vtal + dp_same + (1 + dp_same | subject)", df, family="bernoulli"
    ).fit(**FIT)
    m2 = bmb.Model(
        "y ~ vmag + vtal + dp_all + (1 + dp_all | subject)", df, family="bernoulli"
    ).fit(**FIT)
    out = {
        "_n_trials": int(len(Y)),
        "_n_trials_with_same_cat_dim": int((n_same > 0).sum()),
        "dp_same_cat_only": summ(m.posterior["dp_same"].values),
        "dp_all_dims": summ(m2.posterior["dp_all"].values),
        "vmag": summ(m.posterior["vmag"].values),
        "vtal": summ(m.posterior["vtal"].values),
    }
    for k in ("dp_same_cat_only", "dp_all_dims"):
        v = out[k]
        print(
            f"      {k:20s} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
            f"P(dir)={v['pdir']:.3f}"
        )
    return out


def boundary_distance(
    D: Dict[str, Any], rec: Dict[str, Dict[str, np.ndarray]], su: List[str]
) -> Dict[str, Any]:
    """Boundary distance: are near-boundary exemplars less reliable?"""
    ld = pd.read_csv(f"{DATA}/learning_data.csv")
    ld = (
        ld[ld.run_id.isin(su)]
        .dropna(
            subset=["left_choice", "right_choice", "left_rew", "right_rew", "correct"]
        )
        .copy()
    )
    _, _, _, _, bL = band_arrays(ld.left_choice.values.astype(float))
    _, _, _, _, bR = band_arrays(ld.right_choice.values.astype(float))
    ld["bmean"] = (bL + bR) / 2.0
    ld["vdist"] = (ld.left_rew - ld.right_rew).abs() / 25.0
    ld["correct"] = ld.correct.astype(int)
    z = lambda v: (v - v.mean()) / v.std()
    ld["bd_z"] = z(ld.bmean)
    ld["vd_z"] = z(ld.vdist)
    m = bmb.Model(
        "correct ~ vd_z + bd_z + C(dim) + (1 + bd_z | run_id)", ld, family="bernoulli"
    ).fit(**FIT)
    out = {
        "_n": int(len(ld)),
        "learning_boundary_distance": summ(m.posterior["bd_z"].values),
    }
    v = out["learning_boundary_distance"]
    print(
        f"[3.3]  learning: boundary distance {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
        f"P(dir)={v['pdir']:.3f}   (positive = MORE accurate further from a boundary)"
    )
    return out


def run_tests() -> None:
    D = load_design()
    rec, su = choice_frame(D)
    res = {
        "_geometry": {
            "band_width_deg": BAND_WIDTH,
            "arc_deg": ARC,
            "full_ramp_benchmark_s": round(FULL_RAMP, 4),
            "_note": "31-wide bands / 19-wide gaps / 1-231 arc",
        },
        "learning_angular_distance": learning_angular_distance(su),
        "within_band_choice": within_band_choice(D, rec),
        "boundary_distance": boundary_distance(D, rec, su),
    }
    json.dump(res, open(f"{OUT}/value_map.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/value_map.json")


def run_controls() -> None:
    D = load_design()
    cd, sc = D["cd"], D["sc"]
    Y = np.asarray(D["Y"]).astype(int)
    su = cd.run_id.drop_duplicates().tolist()
    cat0 = get_directions(su).reindex(su)[DIMS]
    desc = cat0.values > 50

    def side(s, d, field):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s}_{p}_dim"].values
            vl = cd[f"choice_trial_{s}_{p}_{field}"].values
            m = dm == d
            out[m] = vl[m]
        return out

    res = {}

    # ---------- 3.2a control: raw-angle position, split by mapping direction ----------
    # unflipped within-band position differences on non-discriminating dims, split by that
    # participant's direction on that dimension
    dp_raw_asc = np.zeros(len(Y))
    dp_raw_desc = np.zeros(len(Y))
    vmag = np.zeros(len(Y))
    vtal = np.zeros(len(Y))
    for j, d in enumerate(DIMS):
        aL, aR = side("left", d, "angle"), side("right", d, "angle")
        vL, vR = side("left", d, "value"), side("right", d, "value")
        kL, kR = np.round(vL / 25).astype(int), np.round(vR / 25).astype(int)
        _, _, _, pL, _ = band_arrays(aL)
        _, _, _, pR, _ = band_arrays(aR)
        dv = (kR - kL).astype(float)
        vmag += dv
        vtal += np.sign(dv)
        dp = np.where(dv == 0, pR - pL, 0.0)  # RAW, not value-referenced
        is_desc = desc[sc, j]
        dp_raw_desc += np.where(is_desc, dp, 0.0)
        dp_raw_asc += np.where(~is_desc, dp, 0.0)

    z = lambda v: (v - v.mean()) / v.std()
    df = pd.DataFrame(
        {
            "y": Y,
            "vmag": z(vmag),
            "vtal": z(vtal),
            "asc": z(dp_raw_asc),
            "desc": z(dp_raw_desc),
            "subject": sc.astype(str),
        }
    )
    m = bmb.Model(
        "y ~ vmag + vtal + asc + desc + (1 | subject)", df, family="bernoulli"
    ).fit(**FIT)
    A, Dg = summ(m.posterior["asc"].values), summ(m.posterior["desc"].values)
    opposite = (A["mean"] * Dg["mean"]) < 0
    res["raw_angle_by_direction"] = {
        "ascending_group": A,
        "descending_group": Dg,
        "signs_opposite": bool(opposite),
        "verdict": (
            "VALUE-LOCKED — opposite raw-angle signs, so the pooled value-referenced "
            "effect is not a perceptual/motor angle bias"
            if opposite
            else "ANGLE-LOCKED — same raw-angle sign in both groups; the value-referenced "
            "effect is an artifact of the direction flip"
        ),
    }
    print(
        f"[3.2a control] raw-angle position, ascending  {A['mean']:+.4f} "
        f"[{A['lo']:+.4f},{A['hi']:+.4f}]"
    )
    print(
        f"[3.2a control] raw-angle position, descending {Dg['mean']:+.4f} "
        f"[{Dg['lo']:+.4f},{Dg['hi']:+.4f}]"
    )
    print(f"               -> {res['raw_angle_by_direction']['verdict']}")

    # ---------- 3.1: angular distance residualized within value-distance level ----------
    ld = pd.read_csv(f"{DATA}/learning_data.csv")
    ld = (
        ld[ld.run_id.isin(su)]
        .dropna(
            subset=["left_choice", "right_choice", "left_rew", "right_rew", "correct"]
        )
        .copy()
    )
    ld["vdist"] = ((ld.left_rew - ld.right_rew).abs() / 25.0).astype(int)
    ld["adist"] = (ld.left_choice - ld.right_choice).abs()
    ld["ad_resid"] = ld.groupby("vdist").adist.transform(
        lambda x: (x - x.mean()) / x.std()
    )
    ld["correct"] = ld.correct.astype(int)
    r2 = float(np.corrcoef(ld.vdist, ld.ad_resid)[0, 1])
    print(
        f"\n[3.1] corr(value dist, residualized angular dist) = {r2:+.4f}  "
        f"(+0.971 unresidualized)"
    )
    print(
        "             within-value-distance angular SD by level: "
        + ", ".join(
            f"{k}:{v:.1f}deg" for k, v in ld.groupby("vdist").adist.std().items()
        )
    )
    m2 = bmb.Model(
        "correct ~ C(vdist) + ad_resid + C(dim) + (1 + ad_resid | run_id)",
        ld,
        family="bernoulli",
    ).fit(**FIT)
    res["residualized_angular_distance"] = {
        "_n": int(len(ld)),
        "collinearity_after_residualizing": round(r2, 4),
        "angular_distance_within_value_level": summ(m2.posterior["ad_resid"].values),
    }
    v = res["residualized_angular_distance"]["angular_distance_within_value_level"]
    print(
        f"             ANGULAR (within value level) {v['mean']:+.4f} "
        f"[{v['lo']:+.4f},{v['hi']:+.4f}] P(dir)={v['pdir']:.3f}"
    )

    json.dump(res, open(f"{OUT}/value_map_controls.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/value_map_controls.json")


def fit_group(
    sc_g: np.ndarray,
    kl_g: np.ndarray,
    kr_g: np.ndarray,
    nS_g: int,
    Y_g: np.ndarray,
    seed: int,
    warmup: int = 1500,
    samples: int = 1500,
    chains: int = 4,
    target: float = 0.92,
) -> az.InferenceData:
    model = functools.partial(group_map, use_lam=False)
    m = MCMC(
        NUTS(model, target_accept_prob=target),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    m.run(
        jax.random.PRNGKey(seed),
        jnp.array(sc_g),
        jnp.array(kl_g),
        jnp.array(kr_g),
        nS_g,
        y=jnp.array(Y_g),
    )
    return az.from_numpyro(m)


def run_counterbalance() -> None:
    kept = canonical_sample(DATA)
    direction = get_directions(kept)

    D = load_design()
    cd_full, sc_full, nS_full = D["cd"], D["sc"], D["nS"]
    Y_full = np.asarray(D["Y"]).astype(int)
    su_full = cd_full.run_id.drop_duplicates().tolist()
    Vl, Vr, _ = raw_values_and_diffs(cd_full)
    kl_full = np.round(Vl / 25.0).astype(int)
    kr_full = np.round(Vr / 25.0).astype(int)

    result = {}
    for target_dim in DIMS:
        di = DIMS.index(target_dim)
        dirs = direction[target_dim].reindex(su_full)
        groups = {
            "ascending(cat0=0pts)": dirs[dirs == 0.0].index.tolist(),
            "descending(cat0=100pts)": dirs[dirs == 100.0].index.tolist(),
        }
        print(f"\n########## dimension: {target_dim} ##########")
        print(f"group Ns: {[(k, len(v)) for k, v in groups.items()]}")
        result[target_dim] = {
            "group_ns": {k: len(v) for k, v in groups.items()},
            "groups": {},
        }

        for gname, participants in groups.items():
            su_map = {r: i for i, r in enumerate(su_full)}
            keep_idx = np.array([su_map[r] for r in participants])
            trial_mask = np.isin(sc_full, keep_idx)
            remap = {old: new for new, old in enumerate(keep_idx)}
            sc_g = np.array([remap[s] for s in sc_full[trial_mask]])
            kl_g, kr_g, Y_g = (
                kl_full[trial_mask],
                kr_full[trial_mask],
                Y_full[trial_mask],
            )
            nS_g = len(participants)
            print(f"  fitting {gname}: nS={nS_g}, n_trials={trial_mask.sum()}")

            ida_g = fit_group(
                sc_g,
                kl_g,
                kr_g,
                nS_g,
                Y_g,
                seed=zlib.crc32(f"{target_dim}|{gname}".encode()) % 100000,
            )
            po_g = ida_g.posterior
            max_rhat = float(
                az.rhat(ida_g, var_names=["a0", "sda", "w_mu", "w_sd", "psi_free"])
                .to_array()
                .max()
            )
            ndiv = int(ida_g.sample_stats["diverging"].values.sum())
            frac = (
                po_g["frac"].values.reshape(-1, 3, 4).mean(0)[di]
            )  # this dim's increments
            frac_draws = po_g["frac"].values.reshape(-1, 3, 4)[:, di, :]
            hdi_arr = az.hdi(frac_draws, hdi_prob=0.95)  # (4,2): [lo,hi] per increment
            lo, hi = hdi_arr[:, 0], hdi_arr[:, 1]
            print(
                f"    max R-hat {max_rhat:.3f} | div {ndiv} | {target_dim} frac = {frac.round(3).tolist()}"
            )
            result[target_dim]["groups"][gname] = {
                "nS": nS_g,
                "frac_mean": frac.tolist(),
                "frac_hdi_lo": lo.tolist(),
                "frac_hdi_hi": hi.tolist(),
                "max_rhat": max_rhat,
                "n_divergences": ndiv,
            }

    with open(os.path.join(OUT, "counterbalance.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {OUT}/counterbalance.json")


COMMANDS = {
    "tests": run_tests,
    "controls": run_controls,
    "counterbalance": run_counterbalance,
}

if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            "usage: python -m cactus_paper.analyses.value_map {tests|controls|counterbalance} [args]"
        )
    _cmd = sys.argv.pop(1)
    COMMANDS[_cmd]()
