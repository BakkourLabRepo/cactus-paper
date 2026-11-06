"""Canonical Exp-2 participant sample / exclusions — single source of truth.

Every notebook and analysis script should get its kept run_ids from
``canonical_sample`` instead of re-deriving ``bad_subs`` locally.

Exclusions (in order):
  1. test subject (run_id == 1)
  2. mouse-quality: > ``mouse_none_thresh`` of choice-trial first-visits are 'none'
  3. super-fast: > ``rt_frac`` of a subject's LEARNING trials fall below the global
     ``rt_pctile`` learning-RT (default: 10th pctile, >20% of trials).

"""

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

TEST_RUN_IDS = [1]

# Set CACTUS_SAMPLE_FILE to a newline-separated list of run_ids to run the WHOLE pipeline (fits,
# make_stats, figures) on an alternate sample -- e.g. the preregistered exclusions -- without
# touching any call site. Unset by default, so every unmodified run still gets the canonical sample.
SAMPLE_OVERRIDE_ENV = "CACTUS_SAMPLE_FILE"


def canonical_sample(
    data_dir: Union[str, Path],
    rt_pctile: float = 0.10,
    rt_frac: float = 0.20,
    mouse_none_thresh: float = 0.05,
    return_breakdown: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Return the sorted array of kept run_ids for Exp 2.

    If ``return_breakdown`` is True, also return a dict documenting each
    exclusion set, the RT threshold, and the final N.
    """
    _ov = os.environ.get(SAMPLE_OVERRIDE_ENV)
    if _ov:
        kept = np.sort(np.loadtxt(_ov, dtype=int, ndmin=1))
        if return_breakdown:
            return kept, {
                "n_kept": int(len(kept)),
                "override_file": _ov,
                "test": [],
                "mouse": [],
                "fast": [],
                "rt_threshold_ms": float("nan"),
                "rt_pctile": rt_pctile,
                "rt_frac": rt_frac,
            }
        return kept

    learning = pd.read_csv(os.path.join(data_dir, "learning_data.csv"))
    choice = pd.read_csv(os.path.join(data_dir, "choice_data.csv"))
    fv = pd.read_csv(os.path.join(data_dir, "first_visits.csv"))

    # (2) mouse quality -- data COVERAGE, not attentional dwell: was any AOI ever
    # touched at all, regardless of duration? `any_aoi_contact` is threshold-free by
    # construction, unlike `first_pos`/`first_dim`, which
    # require a >=100ms genuine visit and so is the wrong column for a tracking
    # QC gate -- a real but brief first glance would count as "none" and inflate this
    # exclusion for reasons that have nothing to do with mouse-tracking quality.
    if "any_aoi_contact" in fv.columns:
        fv = fv.assign(_none=~fv["any_aoi_contact"].astype(bool))
    else:
        fv = fv.assign(_none=fv["first_pos"].eq("none"))
    bad_mouse = (
        fv.groupby("run_id")["_none"]
        .mean()
        .loc[lambda s: s > mouse_none_thresh]
        .index.values
    )
    bad0 = np.unique(np.concatenate([bad_mouse, np.array(TEST_RUN_IDS)]))

    # (3) super-fast learning responders (threshold from mouse/test-clean data)
    clean = learning[~learning.run_id.isin(bad0)]
    thr = clean["rt"].quantile(rt_pctile)
    fast_frac = clean.groupby("run_id")["rt"].apply(lambda r: (r < thr).mean())
    bad_fast = fast_frac.loc[fast_frac > rt_frac].index.values

    excluded = np.unique(np.concatenate([bad0, bad_fast]))
    kept = np.setdiff1d(np.sort(choice.run_id.unique()), excluded)

    if return_breakdown:
        breakdown = {
            "n_kept": int(len(kept)),
            "test": sorted(int(x) for x in TEST_RUN_IDS),
            "mouse": sorted(int(x) for x in bad_mouse),
            "fast": sorted(int(x) for x in bad_fast),
            "rt_threshold_ms": float(thr),
            "rt_pctile": rt_pctile,
            "rt_frac": rt_frac,
        }
        return kept, breakdown
    return kept
