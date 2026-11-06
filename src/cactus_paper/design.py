"""cactus_paper.design — the exp2 choice design loader + derived regressor channels.

jax-free, so figure scripts import fast.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cactus_paper.constants import BAND_WIDTH, BANDS, DIMS, LEVELS, POS
from cactus_paper.data.sample import canonical_sample
from cactus_paper.paths import DATA, UDIM_CACHE

zw = lambda s: (s - s.mean()) / s.std()


def per_subject_drivers(
    kept: np.ndarray, ld: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Per-subject choice-model drivers, derived from the processed data.

    Returns a DataFrame with columns run_id / mean_payne / category_knowledge:
      mean_payne          Payne (1976) index — option-wise positive — averaged over the
                          subject's choice trials (orientation from dim_counts.csv).
      category_knowledge  overall learning-task accuracy: the grand mean over trials, not
                          the mean of the three per-attribute accuracies.

    `kept` is the canonical sample; pass `ld` to reuse an already-filtered learning frame.
    """
    if ld is None:
        ld = pd.read_csv(DATA / "learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]
    dcnt = pd.read_csv(DATA / "dim_counts.csv")
    dcnt = dcnt[dcnt.run_id.isin(kept)]
    out = pd.DataFrame(
        {
            "mean_payne": dcnt.groupby("run_id")["payne_index"].mean(),
            "category_knowledge": ld.groupby("run_id")["correct"].mean(),
        }
    )
    missing = out.index[out.isna().any(axis=1)].tolist()
    assert not missing, f"incomplete Payne/knowledge for run_ids {missing}"
    return out.reset_index()


def load_design() -> Dict[str, Any]:
    """Subjective V_cat/V_rel, per-subject K/P, trial crit_know."""
    C = np.load(UDIM_CACHE)
    u_map = {}
    for k in C.files:
        _, r, d = k.split("_", 2)
        u_map[(int(r), d)] = dict(zip(LEVELS, C[k]))
    kept = canonical_sample(str(DATA))
    cd = pd.read_csv(DATA / "choice_data.csv")
    cd = cd[cd.run_id.isin(kept)].reset_index(drop=True)
    ld = pd.read_csv(DATA / "learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]
    dk = ld.groupby(["run_id", "dim"])["correct"].mean()
    fps = per_subject_drivers(kept, ld)

    def dvf(df, side, d):
        out = np.full(len(df), np.nan)
        for p in POS:
            dm = df[f"choice_trial_{side}_{p}_dim"].values
            vl = df[f"choice_trial_{side}_{p}_value"].values
            m = dm == d
            out[m] = [
                u_map.get((r, d), {}).get(v, 0.0)
                for r, v in zip(df.run_id.values[m], vl[m])
            ]
        return out

    VD = np.stack([dvf(cd, "right", d) - dvf(cd, "left", d) for d in DIMS], 1)
    for i, d in enumerate(DIMS):
        cd[f"vd_{d}"] = VD[
            :, i
        ]  # carry per-dim signed value diff through the merge/dropna
    cd["crit_dim"] = [DIMS[i] for i in np.argmax(np.abs(VD), 1)]
    cd["crit_know"] = [dk.get((r, d), np.nan) for r, d in zip(cd.run_id, cd.crit_dim)]
    cd["V_cat"] = VD.sum(1)
    cd["V_rel"] = cd.right_wins - cd.left_wins
    cd["cr"] = cd.response.astype(int)
    cd = cd.merge(fps, on="run_id").dropna(subset=["crit_know"]).reset_index(drop=True)
    for c in ["V_cat", "V_rel"]:
        cd[c + "_z"] = cd.groupby("run_id")[c].transform(zw)
    cd["critk_wc"] = zw(
        cd.groupby("run_id")["crit_know"].transform(lambda g: g - g.mean())
    )
    sc, su = pd.factorize(cd.run_id)
    nS = len(su)
    pers = cd.drop_duplicates("run_id").set_index("run_id").loc[su]
    VD_final = cd[[f"vd_{d}" for d in DIMS]].values  # (N,3) per-dim signed diff
    dacc = np.array(
        [[dk.get((r, d), np.nan) for d in DIMS] for r in su]
    )  # (nS,3) per-subj dim accuracy
    disp = np.nanmax(dacc, 1) - np.nanmean(dacc, 1)  # validity dispersion (TTB driver)
    return dict(
        cd=cd,
        sc=sc.astype(int),
        nS=nS,
        Vc=cd.V_cat_z.values,
        Vr=cd.V_rel_z.values,
        Y=cd.cr.values.astype(int),
        K=zw(pers.category_knowledge).values,
        P=zw(pers.mean_payne).values,
        ck=cd.critk_wc.values,
        div=(cd.trial_type == "divergent").values,
        VD=VD_final,
        dacc=dacc,
        disp=zw(disp),
    )


def mapping_direction(
    kept: np.ndarray, data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """(run_id, dim) -> 0.0 if category 0 is worth 0 points (ascending), 100.0 if 100 (descending).

    The arc->value mapping direction was counterbalanced per participant per dimension, so any
    position along the arc must be value-referenced with this.
    """
    d = Path(data_dir) if data_dir is not None else DATA
    ld = pd.read_csv(d / "learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]
    a = ld[["run_id", "dim", "left_category", "left_rew"]].rename(
        columns={"left_category": "cat", "left_rew": "rew"}
    )
    b = ld[["run_id", "dim", "right_category", "right_rew"]].rename(
        columns={"right_category": "cat", "right_rew": "rew"}
    )
    long = pd.concat([a, b])
    return long[long.cat == 0].groupby(["run_id", "dim"])["rew"].mean().unstack("dim")


def within_band_position(
    cd: pd.DataFrame, sc: np.ndarray, data_dir: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """(N,3) VALUE-REFERENCED within-band position in [-0.5, +0.5] for the left and right option.

    p > 0 always means "toward the higher-value end of this band", for ascending and descending
    participants alike. Taught value is constant within a band. This is the value-map regressor.
    """
    su = cd.run_id.drop_duplicates().tolist()
    cat0 = mapping_direction(su, data_dir).reindex(su)[DIMS]
    assert cat0.notna().all().all(), (
        "mapping direction missing for some participant x dimension"
    )
    desc = cat0.values > 50  # (nS,3) True = descending

    def _pos(side, d):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{side}_{p}_dim"].values
            an = cd[f"choice_trial_{side}_{p}_angle"].values
            m = dm == d
            out[m] = an[m]
        lo = np.full(len(cd), np.nan)
        for l, h, _ in BANDS:
            inb = (out >= l) & (out <= h)
            lo[inb] = l
        return (out - lo) / BAND_WIDTH - 0.5

    flip = np.where(desc[sc], -1.0, 1.0)  # (N,3)
    pL = np.stack([_pos("left", d) for d in DIMS], 1) * flip
    pR = np.stack([_pos("right", d) for d in DIMS], 1) * flip
    assert not np.isnan(pL).any() and not np.isnan(pR).any(), "angle outside every band"
    return pL, pR


def taught_levels(cd: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(N,3) taught LEVEL INDICES (0-4) per dimension for the left and right option.

    The integer-level counterpart of build_objective_per_dim's point values, for models that
    index a fitted value function u_d(k), such as the model of record.
    """

    def _side(s_, d):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s_}_{p}_dim"].values
            vl = cd[f"choice_trial_{s_}_{p}_value"].values
            m = dm == d
            out[m] = vl[m]
        return out

    Vl = np.stack([_side("left", d) for d in DIMS], 1)
    Vr = np.stack([_side("right", d) for d in DIMS], 1)
    assert not np.isnan(Vl).any() and not np.isnan(Vr).any(), "taught values incomplete"
    return np.round(Vl / 25.0).astype(int), np.round(Vr / 25.0).astype(int)


def build_objective_per_dim(D: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(N,3) objective per-attribute magnitude differences and their signs, within-subject scaled.

    The per-dimension counterpart of build_objective_channels, for models that carry a separate
    weight per attribute (see cactus_paper.models.comparison). Scaled by the within-subject
    SD of the SUM so that unit weights reproduce the summed channels exactly.
    """
    cd = D["cd"]
    sc = D["sc"]

    def side(s, d):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s}_{p}_dim"].values
            vl = cd[f"choice_trial_{s}_{p}_value"].values
            m = dm == d
            out[m] = vl[m]
        return out

    VDo = np.stack([side("right", d) - side("left", d) for d in DIMS], 1)
    assert not np.isnan(VDo).any(), "objective per-dim differences incomplete"
    S = np.sign(VDo)
    ctr = lambda c: c - pd.Series(c).groupby(sc).transform("mean").values
    sdw = lambda t: pd.Series(t).groupby(sc).transform("std").values
    M = np.stack([ctr(VDo[:, i]) for i in range(3)], 1) / sdw(VDo.sum(1))[:, None]
    Sg = np.stack([ctr(S[:, i]) for i in range(3)], 1) / sdw(S.sum(1))[:, None]
    return M, Sg


def build_objective_channels(
    D: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OBJECTIVE value channels — the taught point values, no stage-1 utility model.

    Returns (Vc, Vb, Vw), each z-scored within participant exactly as load_design does:
      Vc  summed objective magnitude difference (right - left)   -> replaces the u-derived V_cat
      Vb  sign on the participant's best-known attribute          -> objective V_ttb
      Vw  sign on their worst-known attribute                     -> objective V_worst
    The tally Vr in load_design is already objective (right_wins - left_wins).
    """
    cd = D["cd"]
    sc = D["sc"]

    def side(s, d):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s}_{p}_dim"].values
            vl = cd[f"choice_trial_{s}_{p}_value"].values
            m = dm == d
            out[m] = vl[m]
        return out

    VDo = np.stack([side("right", d) - side("left", d) for d in DIMS], 1)
    assert not np.isnan(VDo).any(), "objective per-dim differences incomplete"
    S = np.sign(VDo)
    dacc = D["dacc"]
    best = np.argmax(np.where(np.isnan(dacc), -np.inf, dacc), 1)
    worst = np.argmin(np.where(np.isnan(dacc), np.inf, dacc), 1)
    ix = np.arange(len(cd))
    zc = lambda v: zw(pd.Series(v).groupby(sc).transform(lambda g: g - g.mean())).values
    return (
        zw(pd.Series(VDo.sum(1)).groupby(sc).transform(zw)).values,
        zc(S[ix, best[sc]]),
        zc(S[ix, worst[sc]]),
    )


def pole_score(D: Dict[str, Any]) -> np.ndarray:
    """Per-participant P(follow the summed-magnitude winner | divergent trial).

    The model-free tally<->magnitude position that fig3_poles colors participants by: 0 = always
    went with the attribute-win tally, 1 = always with the summed magnitude, .5 = chance.
    Computed from the same choices lam is fit to; not an independent measure of lam.
    """
    cd, sc, nS = D["cd"], D["sc"], D["nS"]
    Y = np.asarray(D["Y"]).astype(int)
    div = np.asarray(D["div"])

    def side(s_, d):
        out = np.full(len(cd), np.nan)
        for p in POS:
            dm = cd[f"choice_trial_{s_}_{p}_dim"].values
            vl = cd[f"choice_trial_{s_}_{p}_value"].values
            m = dm == d
            out[m] = vl[m]
        return out

    VDo = np.stack([side("right", d) - side("left", d) for d in DIMS], 1)
    agg = np.sign(VDo.sum(1))
    follow = (((Y == 1) & (agg > 0)) | ((Y == 0) & (agg < 0))).astype(int)
    return np.array([follow[div & (sc == s)].mean() for s in range(nS)])
