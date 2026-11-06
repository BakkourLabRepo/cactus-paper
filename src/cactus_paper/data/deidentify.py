"""Build the de-identified copy of the data that is shared in the repo.

Reads data/raw and data/processed (gitignored) and writes gzipped copies to data/shared,
keyed on the platform run_id. Dropped: platform/recruitment IDs (Prolific PID/study/session,
SONA id), IP, user agent, referer URL (carries the SONA id), browser/OS, accept-language.
recorded_at is coarsened to the date. In the raw exports the demographics/medical survey
responses are blanked and the debrief free text is cut down to the input-device answer (Q0),
which process_raw.get_input_devices needs; processed input_devices rows stored as the whole
debrief JSON ("NEITHER: {...}") are cut to q0 the same way, and exp1's (keyed on SONA id) are
re-keyed to run_id. The Prolific completion code is redacted.

    pixi run python -m cactus_paper.data.deidentify
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = DATA / "shared"

RAW = {
    "exp1": DATA / "raw/exp1/continuous-attribute-choice-task.csv",
    "exp2": DATA / "raw/exp2/cactus-v2.csv",
}

DROP = [
    "PROLIFIC_PID",
    "STUDY_ID",
    "SESSION_ID",
    "sona_id",
    "ip",
    "user_agent",
    "referer",
    "accept_language",
    "device",
    "browser",
    "browser_version",
    "platform",
    "platform_version",
]


def _read(path, **kw):
    # every cell as its literal string so untouched columns round-trip byte-for-byte
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, **kw)


def _sona_to_run():
    # some SONA ids restarted and have several runs; the device answer comes from the run
    # that reached the debrief (survey-text), so key on that one
    d = _read(RAW["exp1"], usecols=["sona_id", "run_id", "trial_type"])
    d = d[(d.trial_type == "survey-text") & (d.sona_id != "")][["sona_id", "run_id"]]
    assert not d.sona_id.duplicated().any() and not d.run_id.duplicated().any()
    return d.set_index("sona_id")["run_id"]


def scrub(df: pd.DataFrame) -> pd.DataFrame:
    if "sona_id" in df.columns and "run_id" not in df.columns:
        # exp1 input_devices is keyed on the SONA id; re-key to run_id before dropping it
        run = df["sona_id"].map(_sona_to_run())
        assert run.notna().all()
        df.insert(df.columns.get_loc("sona_id"), "run_id", run)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    if "input_device" in df.columns:
        # unparsed answers are stored as "NEITHER: <whole lowercased debrief JSON>"; keep q0
        df["input_device"] = df["input_device"].map(_neither_q0_only)
    if "recorded_at" in df.columns:
        df["recorded_at"] = df["recorded_at"].str[:10]
    return df


def _q0_only(resp: str) -> str:
    if not resp:
        return resp
    return json.dumps({"Q0": json.loads(resp).get("Q0")}, separators=(",", ":"))


def _neither_q0_only(s: str) -> str:
    if not s.startswith("NEITHER: {"):
        return s
    q0 = json.loads(s[len("NEITHER: ") :]).get("q0")
    return "NEITHER: " + json.dumps({"q0": q0}, separators=(",", ":"))


def scrub_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = scrub(df)
    df.loc[df["trial_type"] == "survey", "response"] = ""
    st = df["trial_type"] == "survey-text"
    df.loc[st, "response"] = df.loc[st, "response"].map(_q0_only)
    # the Prolific completion code on the thank-you screen would let anyone submit a completion
    df["stimulus"] = df["stimulus"].str.replace(
        r"(completion code is:\s*)[A-Z0-9]+", r"\1[redacted]", regex=True
    )
    return df


def main() -> None:
    for exp, path in RAW.items():
        out = OUT / "raw" / exp
        out.mkdir(parents=True, exist_ok=True)
        scrub_raw(_read(path)).to_csv(out / f"{path.name}.gz", index=False)
        print("wrote", out / f"{path.name}.gz")

    for exp in ("exp1", "exp2"):
        src, out = DATA / "processed" / exp, OUT / "processed" / exp
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.csv")):
            scrub(_read(f, index_col=0)).to_csv(out / f"{f.name}.gz")
            print("wrote", out / f"{f.name}.gz")
        for f in src.glob("*.txt"):
            (out / f.name).write_text(f.read_text())


if __name__ == "__main__":
    main()
