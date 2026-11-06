from dataclasses import dataclass
from io import StringIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Minimum dwell (ms) for a contiguous in-AOI run to count as a genuine visit rather
# than a transit pass-through (e.g. a straight cursor path between the top and bottom
# AOI of one option grazes the middle AOI's 100 px radius for a handful of samples).
DWELL_THRESHOLD_MS = 100.0


@dataclass
class AOI:  # All my AOIs are basically circles
    name: str
    cx: float
    cy: float
    coord: np.ndarray
    r_in: float
    r_out: float  # r_out >= r_in
    dim: str  # dimension for the AOI (line, color, shape)
    dim_side: str  # dimension combined with side (left/right)


def extract_mouse_data_and_targets(
    choice_data: pd.DataFrame, trial_num: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mouse_targets = pd.read_json(
        StringIO(choice_data.iloc[trial_num]["mouse_tracking_targets"])
    ).T
    mouse_targets["mid_x"] = mouse_targets["left"] + 192 / 2
    mouse_targets["mid_y"] = mouse_targets["top"] + 192 / 2
    mouse_targets = add_dim_to_targets(choice_data.iloc[trial_num], mouse_targets)
    mouse_data = pd.read_json(
        StringIO(choice_data.iloc[trial_num]["mouse_tracking_data"])
    )
    # Guarantee time-ordering: all downstream measures (first visits, visit
    # sequences, dwell times) assume rows are sorted by t.
    mouse_data = mouse_data.sort_values("t").reset_index(drop=True)
    return mouse_data, mouse_targets


def add_dim_to_targets(
    choice_trial_data: pd.Series, mouse_targets: pd.DataFrame
) -> pd.DataFrame:
    mapping_dict = {
        "#left-top": "left_top",
        "#left-mid": "left_mid",
        "#left-bot": "left_bot",
        "#right-top": "right_top",
        "#right-mid": "right_mid",
        "#right-bot": "right_bot",
    }
    suffix_series = mouse_targets.index.map(mapping_dict)
    target_col_names = "choice_trial_" + suffix_series + "_dim"
    mouse_targets["dim"] = choice_trial_data[target_col_names].values

    return mouse_targets


def generate_aoi_list_from_targets(mouse_targets: pd.DataFrame) -> List[AOI]:
    aois = []
    for position, row in mouse_targets.iterrows():
        position = str(position)  # should be something like #left-top etc
        if "left" in position:
            side = "left"
        elif "right" in position:
            side = "right"
        aois.append(
            AOI(
                name=position,
                cx=row["mid_x"],
                cy=row["mid_y"],
                coord=np.asarray([row["mid_x"], row["mid_y"]]),
                r_in=90,
                r_out=110,
                dim=row["dim"],
                dim_side=f"{side}_{row['dim']}",
            )
        )
    return aois


def _label_samples(
    mouse_data: pd.DataFrame, aois: List[AOI], label_of, radius: float = 100.0
) -> np.ndarray:
    """Per-sample AOI label (via `label_of(aoi)`), or None where outside every AOI."""
    x = mouse_data["x"].to_numpy(float)
    y = mouse_data["y"].to_numpy(float)
    label = np.full(len(x), None, dtype=object)
    min_d = np.full(len(x), np.inf)
    for aoi in aois:
        d = np.sqrt((x - aoi.cx) ** 2 + (y - aoi.cy) ** 2)
        closer = (d <= radius) & (d < min_d)  # AOIs never overlap; guards ties anyway
        label[closer] = label_of(aoi)
        min_d[closer] = d[closer]
    return label


def generate_dwell_filtered_visits(
    mouse_data: pd.DataFrame,
    aois: List[AOI],
    label_of: Callable[[AOI], Optional[str]],
    trial_rt: float,
    min_dwell_ms: float = DWELL_THRESHOLD_MS,
    radius: float = 100.0,
) -> pd.DataFrame:
    """Collapse the raw sample stream into genuine AOI visits.

    A "visit" is a contiguous run of samples inside one AOI whose dwell (exit
    time minus entry time) reaches `min_dwell_ms`; shorter runs are dropped as
    transit pass-throughs rather than reassigned. Dropping a short run can make
    its neighbors adjacent-and-identical (e.g. top -> [short mid] -> bot
    collapses to a single top -> bot occupancy once mid is dropped), so those
    are merged back into one visit. Exit time for the final run in a trial
    falls back to `trial_rt`, or the last sample's timestamp if `trial_rt` is
    not finite.
    """
    md = mouse_data.sort_values("t").reset_index(drop=True)
    t = md["t"].to_numpy(float)
    n = len(t)
    if n == 0:
        return pd.DataFrame({"which_aoi": pd.Series(dtype=object)})

    label = _label_samples(md, aois, label_of, radius=radius)
    t_end = float(trial_rt) if np.isfinite(trial_rt) else t[-1]

    change = np.r_[True, label[1:] != label[:-1]]
    starts = np.flatnonzero(change)
    stops = np.r_[starts[1:], n] - 1  # inclusive end index of each run

    kept = []
    for s, e in zip(starts, stops):
        if label[s] is None:
            continue
        exit_t = t[e + 1] if e + 1 < n else t_end
        if exit_t - t[s] >= min_dwell_ms:
            kept.append(label[s])

    collapsed = [a for i, a in enumerate(kept) if i == 0 or a != kept[i - 1]]
    return pd.DataFrame({"which_aoi": collapsed})


def generate_visits_positional(
    mouse_data: pd.DataFrame, aois: List[AOI], trial_rt: float = np.nan
) -> pd.DataFrame:
    mapping_dict = {
        "#left-top": "A",
        "#left-mid": "B",
        "#left-bot": "C",
        "#right-top": "A'",
        "#right-mid": "B'",
        "#right-bot": "C'",
    }
    return generate_dwell_filtered_visits(
        mouse_data, aois, lambda aoi: mapping_dict[aoi.name], trial_rt
    )


def generate_visits_dimensional(
    mouse_data: pd.DataFrame, aois: List[AOI], trial_rt: float = np.nan
) -> pd.DataFrame:
    mapping_dict = {
        "left_shape": "L_shape",
        "left_color": "L_color",
        "left_line": "L_line",
        "right_shape": "R_shape",
        "right_color": "R_color",
        "right_line": "R_line",
    }
    return generate_dwell_filtered_visits(
        mouse_data, aois, lambda aoi: mapping_dict[aoi.dim_side], trial_rt
    )


def get_first_dim_pos(
    mouse_data: pd.DataFrame, aois: List[AOI], trial_rt: float = np.nan
) -> Dict[str, object]:
    positional_visits = generate_visits_positional(mouse_data, aois, trial_rt)
    dimensional_visits = generate_visits_dimensional(mouse_data, aois, trial_rt)
    first_visit = {}
    if len(positional_visits) > 0:
        first_visit["position"] = positional_visits.loc[0, "which_aoi"]
    else:
        first_visit["position"] = "none"
    if len(dimensional_visits) > 0:
        first_visit["dimension"] = dimensional_visits.loc[0, "which_aoi"]
    else:
        first_visit["dimension"] = "none"
    # Coverage check, independent of the dwell threshold: did the cursor register
    # inside ANY AOI at all, even briefly? This is what canonical_sample's
    # mouse-quality gate should read -- "was this trial tracked" is a data-coverage
    # question, distinct from "did it produce a genuine >=100ms attentional visit".
    md = mouse_data.sort_values("t").reset_index(drop=True)
    label = _label_samples(md, aois, lambda aoi: True)
    first_visit["any_aoi_contact"] = bool((label != None).any())  # noqa: E711 (object array)
    return first_visit


def get_time_totals_by_dim(mouse_data: pd.DataFrame, aois: List[AOI]) -> pd.DataFrame:
    """Total dwell time (ms) per dimensional AOI for one trial.

    Each sample is credited with the time until the *next* sample, attributed
    to the AOI the cursor occupies at that sample. Samples outside every AOI
    are credited to nothing, so inter-AOI travel time is excluded. The final
    sample has no successor and contributes no
    dwell -- unavoidable without a trial-end timestamp.
    """
    mapping_dict = {
        "left_shape": "L_shape",
        "left_color": "L_color",
        "left_line": "L_line",
        "right_shape": "R_shape",
        "right_color": "R_color",
        "right_line": "R_line",
    }
    md = mouse_data.sort_values("t").reset_index(drop=True).copy()
    which_aoi = pd.Series(pd.NA, index=md.index, dtype="object")
    for aoi in aois:  # AOIs never overlap (verified), so no double assignment
        inside = np.sqrt((md["x"] - aoi.cx) ** 2 + (md["y"] - aoi.cy) ** 2) <= 100
        which_aoi[inside.values] = mapping_dict[aoi.dim_side]
    md["which_aoi"] = which_aoi
    md["time_spent_in_AOI"] = md["t"].shift(-1) - md["t"]  # time until next sample
    time_totals_by_dim = (
        md.dropna(subset=["which_aoi"])
        .groupby("which_aoi")["time_spent_in_AOI"]
        .sum()
        .reset_index()
    )  # getting the total amount of time per aoi per side
    return time_totals_by_dim


def generate_aoi_shifts(aoi_visits: pd.DataFrame) -> List[Tuple[str, str]]:
    aoi_visits["shift_aoi"] = aoi_visits["which_aoi"] != aoi_visits["which_aoi"].shift(
        1
    )
    aoi_switches = aoi_visits[aoi_visits["shift_aoi"] == True].reset_index(drop=True)
    return list(zip(aoi_switches["which_aoi"], aoi_switches["which_aoi"].shift(-1)))


def get_option_switch_count(aoi_shifts: List[Tuple[str, str]]) -> Dict[str, object]:
    """Classify AOI shifts as within-option, attribute-wise, or diagonal, and compute
    the chance-corrected Payne index (Bockenholt & Hynan, 1994) from the first two.

    With 3 attributes x 2 options, every AOI has 5 possible destinations: 2
    within-option (same side, other attribute), 1 attribute-wise (other side,
    matching attribute), and 2 diagonal (other side, other attribute). A shift
    counts as "attribute-wise" only when it also matches attribute -- a diagonal
    move like L_shape -> R_color changes both option and attribute, so it is
    neither option-wise nor attribute-wise processing and is excluded from the
    index rather than folded into "between-option".

    Pooling within_option and attribute_wise directly is not centered at 0 under random search, since a random searcher takes an
    attribute-wise-or-diagonal move 3x as often as a within-option move (3/5 vs
    2/5). Dividing each count by its number of possible transitions (2 and 1)
    before differencing recovers a chance baseline of 0:

        payne_index = (within_option/2 - attribute_wise/1) / (within_option/2 + attribute_wise/1)

    Ranges from -1 (all attribute-wise) to +1 (all within-option/option-wise).
    """
    within_option = 0
    attribute_wise = 0
    diagonal = 0

    for shift in aoi_shifts:
        if shift[0] is None or shift[1] is None:
            continue
        from_side, from_dim = (
            shift[0][0],
            shift[0][2:],
        )  # e.g. "L_shape" -> "L", "shape"
        to_side, to_dim = shift[1][0], shift[1][2:]
        if from_side == to_side:
            within_option += 1
        elif from_dim == to_dim:
            attribute_wise += 1
        else:
            diagonal += 1

    n_within_possible, n_attr_possible = 2, 1
    w_rate = within_option / n_within_possible
    a_rate = attribute_wise / n_attr_possible
    total_rate = w_rate + a_rate
    payne_index = (w_rate - a_rate) / total_rate if total_rate > 0 else np.nan

    return {
        "within_option": within_option,
        "attribute_wise": attribute_wise,
        "diagonal": diagonal,
        "between_options": attribute_wise + diagonal,  # every move that changes option
        "payne_index": payne_index,
    }


def get_dim_switch_count(aoi_shifts: List[Tuple[str, str]]) -> Dict[str, int]:
    """Count AOI switches by dimension.

    A switch is within-dimension when source and target share the same
    dimension (e.g. L_shape -> R_shape) and between-dimension otherwise.
    Each switch is classified exactly once by matching the dimension token.
    """
    dims = ("shape", "color", "line")
    dim_counts = {"shape": 0, "color": 0, "line": 0, "between": 0}

    for src, dst in aoi_shifts:
        if src is None or dst is None:
            continue
        dim_src = next((d for d in dims if d in src), None)
        dim_dst = next((d for d in dims if d in dst), None)
        if dim_src is None or dim_dst is None:
            continue
        if dim_src == dim_dst:
            dim_counts[dim_src] += 1
        else:
            dim_counts["between"] += 1

    return dim_counts


def _iter_trials(choice_data: pd.DataFrame):
    """Yield (run_id, trial_idx, mouse_data, aois, rt) for each trial."""
    for i in range(len(choice_data)):
        mouse_data, mouse_targets = extract_mouse_data_and_targets(
            choice_data, trial_num=i
        )
        aois = generate_aoi_list_from_targets(mouse_targets)
        yield (
            choice_data.loc[i, "run_id"],
            choice_data.loc[i, "trial_idx"],
            mouse_data,
            aois,
            choice_data.loc[i, "rt"],
        )


def generate_visit_sequence(aoi_visits: pd.DataFrame) -> str:
    """Collapse consecutive AOI visits into a dash-separated sequence string.

    E.g. "L_shape-R_shape-R_color"
    """
    if len(aoi_visits) == 0:
        return "none"
    # Drop consecutive duplicates (cursor can stay in same AOI across many rows)
    visits = aoi_visits["which_aoi"]
    collapsed = visits[visits != visits.shift(1)]
    return "-".join(collapsed.tolist())


def get_all_visit_sequences(choice_data: pd.DataFrame) -> pd.DataFrame:
    results = []
    for run_id, trial_idx, mouse_data, aois, rt in _iter_trials(choice_data):
        aoi_visits = generate_visits_dimensional(mouse_data, aois, rt)
        results.append(
            {
                "run_id": run_id,
                "trial_idx": trial_idx,
                "visit_sequence": generate_visit_sequence(aoi_visits),
            }
        )
    return pd.DataFrame(results)


def get_all_first_visits(choice_data: pd.DataFrame) -> pd.DataFrame:
    first_visits = []
    for run_id, trial_idx, mouse_data, aois, rt in _iter_trials(choice_data):
        first_visit_dict = get_first_dim_pos(mouse_data, aois, rt)
        first_visits.append(
            {
                "run_id": run_id,
                "trial_idx": trial_idx,
                "first_dim": first_visit_dict["dimension"],
                "first_pos": first_visit_dict["position"],
                "any_aoi_contact": first_visit_dict["any_aoi_contact"],
            }
        )
    return pd.DataFrame(first_visits)


def get_all_dim_counts(choice_data: pd.DataFrame) -> pd.DataFrame:
    dim_counts = []
    for run_id, trial_idx, mouse_data, aois, rt in _iter_trials(choice_data):
        aoi_visits = generate_visits_dimensional(mouse_data, aois, rt)
        aoi_shifts = generate_aoi_shifts(aoi_visits)
        dim_switch_dict = get_dim_switch_count(aoi_shifts)
        option_switch_dict = get_option_switch_count(aoi_shifts)
        dim_counts.append(
            {
                "run_id": run_id,
                "trial_idx": trial_idx,
                "shape_visits": dim_switch_dict["shape"],
                "color_visits": dim_switch_dict["color"],
                "line_visits": dim_switch_dict["line"],
                "between_visits": dim_switch_dict["between"],
                "within_option": option_switch_dict["within_option"],
                "attribute_wise": option_switch_dict["attribute_wise"],
                "diagonal": option_switch_dict["diagonal"],
                "between_options": option_switch_dict["between_options"],
                "payne_index": option_switch_dict["payne_index"],
            }
        )
    return pd.DataFrame(dim_counts)


def get_time_totals(choice_data: pd.DataFrame) -> pd.DataFrame:
    times = []
    for run_id, trial_idx, mouse_data, aois, rt in _iter_trials(choice_data):
        time_total_per_trial = get_time_totals_by_dim(mouse_data, aois)
        time_total_per_trial["run_id"] = run_id
        time_total_per_trial["trial_idx"] = trial_idx
        times.append(time_total_per_trial)
    return pd.concat(times)


# model matrices
