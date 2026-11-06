import argparse
import os

import pandas as pd

from cactus_paper.data.mousetracking_utils import *


def save(input_path: str, df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(input_path, name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path where the choice_data.csv is stored",
    )
    args = parser.parse_args()
    choice_file = os.path.join(args.input_path, "choice_data.csv")
    assert os.path.exists(choice_file), "Cannot find choice file!"
    choice_data = pd.read_csv(choice_file)
    first_visits = get_all_first_visits(choice_data)
    dim_counts = get_all_dim_counts(choice_data)
    visit_sequences = get_all_visit_sequences(choice_data)
    time_totals = get_time_totals(choice_data)
    names = [
        "first_visits.csv",
        "dim_counts.csv",
        "visit_sequences.csv",
        "time_totals.csv",
    ]
    for df, name in zip(
        [first_visits, dim_counts, visit_sequences, time_totals], names
    ):
        save(args.input_path, df, name)


if __name__ == "__main__":
    import time

    print("Running mousetracking data extraction")
    start = time.time()
    main()
    end = time.time()
    print(f"Done! total time: {(end - start):.4f}")
