"""The DeltaValue slope: how steeply learning accuracy rises with category distance.

  correct ~ 1 + z(category distance) + (1 + z(category distance) | subject:dimension)

Outputs (results/fits/):
  map_slope_draws.npy     draws x participant slope, mean over the three dimensions
                          (read by cactus_paper.analyses.antecedents)
  map_slope_cells.npz     per participant x dimension posterior means
  learning_slope.json     summary numbers

Run from repo root (~3 min each; run the halves first, the full fit reads them for split_half):
  pixi run python -m cactus_paper.features.learning_slope --half odd
  pixi run python -m cactus_paper.features.learning_slope --half even
  pixi run python -m cactus_paper.features.learning_slope
  pixi run python -m cactus_paper.features.learning_slope --exclude-rev 1      # drop post-reversal trials
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from cactus_paper.constants import DIMS
from cactus_paper.design import load_design
from cactus_paper.paths import DATA, FITS

SEED = 20260826


def summ(x: Any) -> Dict[str, float]:
    x = np.asarray(x).ravel()
    lo, hi = az.hdi(x, hdi_prob=0.95)
    return {
        "mean": round(float(x.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "pdir": round(float(max((x > 0).mean(), (x < 0).mean())), 3),
    }


def show(name: str, d: Dict[str, float]) -> None:
    star = "" if d["lo"] < 0 < d["hi"] else "  *"
    print(
        f"  {name:38s} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
        f"P(dir)={d['pdir']:.3f}{star}"
    )


def build_learning(su: Sequence[int]) -> pd.DataFrame:
    ld = pd.read_csv(f"{DATA}/learning_data.csv", low_memory=False)
    ld = ld.dropna(subset=["left_category", "right_category", "correct", "dim"])
    ld = ld[ld.run_id.isin(su)].copy()
    ld["dist"] = (ld.left_category - ld.right_category).abs()
    ld["correct"] = ld.correct.astype(int)
    ld["cell"] = ld.run_id.astype(str) + "|" + ld.dim.astype(str)
    ld["distz"] = (ld.dist - ld.dist.mean()) / ld.dist.std()
    return ld


def confound(ld: pd.DataFrame) -> Dict[str, Any]:
    ang = ld.left_vs_right_angle_diff_abs.astype(float)
    return {
        "r_category_distance_vs_angular_distance": round(
            float(np.corrcoef(ld.dist, ang)[0, 1]), 4
        ),
        "angle_by_distance": {
            int(k): [round(float(v["mean"]), 1), round(float(v["std"]), 1)]
            for k, v in ang.groupby(ld.dist).agg(["mean", "std"]).iterrows()
        },
        "_what": "category distance is angular distance times a constant; the gradient cannot "
        "separate map precision from sensory acuity in this design",
    }


def write_outputs(
    ld: pd.DataFrame,
    su: List[int],
    K: np.ndarray,
    sl: np.ndarray,
    ic: np.ndarray,
    po: Any,
    slope_s: np.ndarray,
    conf: Dict[str, Any],
    out_dir: Union[str, Path],
    sfx: str = "",
) -> None:
    """sl / ic are draws x nS x 3 per-cell slopes / intercepts; slope_s is draws x nS."""
    draws_out = f"{out_dir}/map_slope_draws{sfx}.npy"
    cells_out = f"{out_dir}/map_slope_cells{sfx}.npz"
    json_out = f"{out_dir}/learning_slope{sfx}.json"
    # reference: the full-trial-set draws; a suffixed fit is compared against them
    ref_path = f"{FITS}/map_slope_draws.npy"
    ref = np.load(ref_path) if os.path.exists(ref_path) else slope_s
    np.save(draws_out, slope_s)
    sd_dist = float(ld.dist.std())
    # dist_mean / dist_sd let a figure rebuild each cell's fitted curve, logit^-1(level + slope * z)
    np.savez(
        cells_out,
        slope=sl.mean(0),
        level=ic.mean(0),
        run_id=np.array(su),
        dims=np.array(DIMS),
        dist_mean=float(ld.dist.mean()),
        dist_sd=sd_dist,
    )
    out = {
        "_what": "learning-task psychometric correct ~ z(category distance) + (z(distance) | "
        "subject:dimension); slopes in logit per SD of category distance",
        "sd_distance_categories": round(sd_dist, 3),
        "sd_distance_points": round(25 * sd_dist, 1),
        "n_trials": int(len(ld)),
        "n_cells": int(sl.shape[1] * sl.shape[2]),
        "group_intercept": summ(po["Intercept"].values),
        "group_slope": summ(po["distz"].values),
        "sd_intercept": summ(po["1|cell_sigma"].values),
        "sd_slope": summ(po["distz|cell_sigma"].values),
        "by_dim": {d: summ(sl[:, :, j].mean(-1)) for j, d in enumerate(DIMS)},
        "acc_by_distance": {
            int(k): round(float(v), 3)
            for k, v in ld.groupby("dist").correct.mean().items()
        },
        "n_cells_slope_le0": int((sl.mean(0) <= 0).sum()),
        "n_subj_slope_le0": int((slope_s.mean(0) <= 0).sum()),
        "corr_slope_K": summ(
            [np.corrcoef(slope_s[d], K)[0, 1] for d in range(len(slope_s))]
        ),
        "r_distance_vs_angle": conf["r_category_distance_vs_angular_distance"],
        "rerun_vs_saved_r": round(
            float(np.corrcoef(slope_s.mean(0), ref.mean(0))[0, 1]), 4
        ),
    }
    # split-half reliability, from the --half fits of the same trial set (run those first)
    halves = [cells_out.replace(".npz", f"_{h}.npz") for h in ("odd", "even")]
    if all(os.path.exists(h) for h in halves):
        o, e = (np.load(h) for h in halves)
        r_s = float(np.corrcoef(o["slope"].mean(1), e["slope"].mean(1))[0, 1])
        r_l = float(np.corrcoef(o["level"].mean(1), e["level"].mean(1))[0, 1])
        tert = lambda v: np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
        out["split_half"] = {
            "_what": "odd vs even trial_per_dim fits; participant = mean over the three dimensions; "
            "spearman_brown = full-length reliability",
            "slope_r": round(r_s, 3),
            "slope_spearman_brown": round(2 * r_s / (1 + r_s), 3),
            "slope_cell_r": round(
                float(np.corrcoef(o["slope"].ravel(), e["slope"].ravel())[0, 1]), 3
            ),
            "level_r": round(r_l, 3),
            "level_spearman_brown": round(2 * r_l / (1 + r_l), 3),
            "tertile_agreement_by_dim": {
                d: round(
                    float((tert(o["slope"][:, j]) == tert(e["slope"][:, j])).mean()), 3
                )
                for j, d in enumerate(DIMS)
            },
        }
    else:
        print("  ! no --half fits for this trial set; split_half omitted")
    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {draws_out}\n   {cells_out}\n   {json_out}")
    for k in ("group_slope", "sd_slope", "corr_slope_K"):
        show(k, out[k])
    for d in DIMS:
        show(f"slope[{d}]", out["by_dim"][d])
    print(
        f"  cells with slope <= 0: {out['n_cells_slope_le0']}/{out['n_cells']};  participants: "
        f"{out['n_subj_slope_le0']}/{len(su)};  r vs saved draws = {out['rerun_vs_saved_r']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--half",
        choices=["odd", "even"],
        help="fit on odd / even trial_per_dim only and write map_slope_cells_<half>.npz "
        "(no JSON). Feeds split_half; Fig 1C cuts tertiles on one half and plots "
        "the other, so the fan-out is not selection on the plotted noise",
    )
    ap.add_argument(
        "--exclude-rev",
        type=int,
        default=0,
        metavar="N",
        help="drop the N trials starting at each prompt reversal (1 = trials 10/20/30; "
        "2 = also 11/21/31). On a reversal trial a participant still following the "
        "old prompt answers 1 - p(distance), so carry-over flattens the slope. "
        "Outputs get a _norev<N> suffix",
    )
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument(
        "--out", default=str(FITS), help="output directory (default results/fits)"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    D = load_design()
    su = D["cd"].run_id.drop_duplicates().tolist()
    K = np.asarray(D["K"])

    ld = build_learning(su)
    if a.exclude_rev:
        drop = [f + k for f in (10, 20, 30) for k in range(a.exclude_rev)]
        ld = ld[~ld.trial_per_dim.isin(drop)].copy()
    if a.half:
        ld = ld[ld.trial_per_dim % 2 == (1 if a.half == "odd" else 0)].copy()
    if a.exclude_rev or a.half:
        ld["distz"] = (ld.dist - ld.dist.mean()) / ld.dist.std()
    sfx = f"_norev{a.exclude_rev}" if a.exclude_rev else ""
    conf = confound(ld)
    print(
        f"r(|d category|, |d angle|) = {conf['r_category_distance_vs_angular_distance']:+.3f}"
    )

    print(
        f"hierarchical psychometric ({len(ld):,} trials, {ld.cell.nunique()} subject x "
        f"dimension cells)",
        flush=True,
    )
    m = bmb.Model("correct ~ distz + (distz | cell)", ld, family="bernoulli")
    idata = m.fit(
        inference_method="nutpie",
        draws=a.draws,
        tune=a.draws,
        chains=4,
        random_seed=SEED,
        progressbar=False,
    )
    po = idata.posterior

    # per-cell slope draws -> participant-level slope (mean over that participant's three dimensions)
    _cdim = [d for d in po["distz|cell"].dims if d not in ("chain", "draw")][0]
    cells = [str(c) for c in po["distz|cell"].coords[_cdim].values]
    sl_cell = po["distz|cell"].values.reshape(-1, len(cells)) + po[
        "distz"
    ].values.reshape(-1, 1)
    ic_cell = po["1|cell"].values.reshape(-1, len(cells)) + po[
        "Intercept"
    ].values.reshape(-1, 1)
    idx = {c: i for i, c in enumerate(cells)}
    take = np.array([[idx[f"{r}|{d}"] for d in DIMS] for r in su])  # nS x 3

    if a.half:
        out = f"{a.out}/map_slope_cells{sfx}_{a.half}.npz"
        np.savez(
            out,
            slope=sl_cell[:, take].mean(0),
            level=ic_cell[:, take].mean(0),
            run_id=np.array(su),
            dims=np.array(DIMS),
        )
        print(f"-> {out}")
        return
    write_outputs(
        ld,
        su,
        K,
        sl_cell[:, take],
        ic_cell[:, take],
        po,
        sl_cell[:, take].mean(-1),
        conf,
        a.out,
        sfx,
    )


if __name__ == "__main__":
    main()
