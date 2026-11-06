import argparse
import os

import numpy as np
import pandas as pd


def select_learning(raw_data: pd.DataFrame) -> pd.DataFrame:
    learning_data = (
        raw_data[raw_data["task"] == "learning"]
        .reset_index(drop=True)
        .dropna(how="all", axis=1)
    )
    return learning_data


def add_cols_to_learning(learning_data: pd.DataFrame) -> pd.DataFrame:
    learning_data["left_better"] = (
        learning_data["left_rew"] > learning_data["right_rew"]
    )
    learning_data["left_correct"] = (
        (learning_data["left_better"]) & (learning_data["prompt_dir"] == "bigger")
    ) | ((~learning_data["left_better"]) & (learning_data["prompt_dir"] == "smaller"))
    learning_data["chose_left"] = learning_data["response"].apply(
        lambda x: 1 if x == "f" else 0
    )
    learning_data = learning_data.dropna(subset=["response"]).reset_index(drop=True)
    learning_data["correct"] = (
        learning_data["chose_left"] == learning_data["left_correct"]
    )
    learning_data["left_vs_right_angle_diff"] = (
        learning_data["left_choice"] - learning_data["right_choice"]
    )
    learning_data["left_angle_more_3_oclock"] = (
        learning_data["left_vs_right_angle_diff"] > 0
    )
    learning_data["left_vs_right_rew_diff"] = (
        learning_data["left_rew"] - learning_data["right_rew"]
    )
    for col in ["left_vs_right_angle_diff", "left_vs_right_rew_diff"]:
        learning_data[f"{col}_abs"] = np.abs(learning_data[col])
    learning_data["trial_per_dim"] = learning_data.groupby(["run_id", "dim"]).cumcount()

    return learning_data


def select_choice(raw_data: pd.DataFrame) -> pd.DataFrame:
    choice_data = (
        raw_data[
            ~raw_data["choice_trial_left_bot_angle"].isna()
            & ~(raw_data["choice_trial_left_bot_angle"] >= 290)
            & ~(raw_data["rt"].isna())
        ]
        .dropna(axis=1)
        .reset_index(drop=True)
    )
    choice_data["response"] = choice_data["response"].astype(int)
    choice_data["chose_left"] = choice_data["response"] == 0

    conversion_dict = {
        "choice_trial_left_total_rew": float,
        "choice_trial_right_total_rew": float,
    }

    choice_data = choice_data.astype(conversion_dict)
    return choice_data


def add_cols_to_choice(choice_data: pd.DataFrame) -> pd.DataFrame:
    choice_data["left_vs_right_rew_diff"] = (
        choice_data["choice_trial_left_total_rew"]
        - choice_data["choice_trial_right_total_rew"]
    )
    choice_data["left_vs_right_rew_diff_abs"] = np.abs(
        choice_data["left_vs_right_rew_diff"]
    )
    choice_data["left_better"] = (
        choice_data["choice_trial_left_total_rew"]
        > choice_data["choice_trial_right_total_rew"]
    )
    choice_data["correct"] = (choice_data["left_better"]) == (
        choice_data["response"] == 0
    )
    for pos in ["top", "mid", "bot"]:
        choice_data[f"{pos}_is_same_cat"] = (
            choice_data[f"choice_trial_left_{pos}_value"]
            == choice_data[f"choice_trial_right_{pos}_value"]
        )
        choice_data[f"{pos}_left_vs_{pos}_right_angle_diff"] = np.abs(
            choice_data[f"choice_trial_left_{pos}_angle"]
            - choice_data[f"choice_trial_right_{pos}_angle"]
        )
    choice_data["total_angle_diff"] = (
        choice_data["top_left_vs_top_right_angle_diff"]
        + choice_data["mid_left_vs_mid_right_angle_diff"]
        + choice_data["bot_left_vs_bot_right_angle_diff"]
    )
    choice_data["trial_idx"] = choice_data.groupby("run_id").cumcount()
    return choice_data


def select_freesort(raw_data: pd.DataFrame) -> pd.DataFrame:
    free_sort_data = (
        raw_data[raw_data["trial_type"] == "free-sort"]
        .dropna(axis=1)
        .reset_index(drop=True)
    )
    return free_sort_data


def get_input_devices(raw_data: pd.DataFrame) -> pd.DataFrame:
    survey_resps = raw_data[raw_data["trial_type"] == "survey-text"][
        ["run_id", "response"]
    ].reset_index(drop=True)
    input_devices = {}
    for i, row in survey_resps.iterrows():
        if "trackpad" in row["response"].lower():
            input_devices[row["run_id"]] = "trackpad"
        elif "mouse" in row["response"].lower():
            input_devices[row["run_id"]] = "mouse"
        else:
            input_devices[row["run_id"]] = f"NEITHER: {row['response'].lower()}"
    inputs = (
        pd.DataFrame.from_dict(input_devices, orient="index")
        .reset_index()
        .rename(columns={"index": "run_id", 0: "input_device"})
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path where the raw data is stored",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the processed data should be saved to",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    assert os.path.exists(input_path), "Input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"

    # load the csv and run through it to separate out the files that I need
    raw_data = pd.read_csv(input_path, dtype={"run_id": str})
    sona_data = raw_data[
        ~raw_data["run_id"].str.contains("test", na=False)
    ].reset_index(drop=True)
    sona_data = sona_data[~sona_data["run_id"].isna()].reset_index(drop=True)
    # get learning
    learning_data = select_learning(sona_data)
    learning_data = add_cols_to_learning(learning_data)
    learning_data.to_csv(os.path.join(output_path, "learning_data.csv"))
    # get choice
    choice_data = select_choice(sona_data)
    choice_data = add_cols_to_choice(choice_data)
    choice_data.to_csv(os.path.join(output_path, "choice_data.csv"))
    # get free-sort
    free_sort = select_freesort(sona_data)
    free_sort.to_csv(os.path.join(output_path, "free_sort_data.csv"))
    # get input devices
    input_devices = get_input_devices(sona_data)
    input_devices.to_csv(os.path.join(output_path, "input_devices.csv"))


if __name__ == "__main__":
    main()
