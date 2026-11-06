"""cactus_paper.data.device — pointer device (mouse vs trackpad) per participant.

Reads data/processed/exp2/input_devices.csv; rows stored as "NEITHER: {...}" are resolved by
re-parsing q0 (the pointer question) of the debrief JSON. Used by the pointer-device refits in
cactus_paper.analyses.robustness.
"""

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from cactus_paper.paths import DATA

TRACKPAD_TOKENS = ("trackpad", "track pad", "touchpad", "traackpad", "laptop")
MOUSE_TOKENS = ("mouse", "desktop")


def _parse(raw):
    s = str(raw)
    if s in ("mouse", "trackpad"):
        return s
    m = re.search(r"\{.*\}", s)  # "NEITHER: {...}" -> read q0 only
    q0 = ""
    if m:
        try:
            q0 = str(json.loads(m.group(0)).get("q0", "")).lower()
        except json.JSONDecodeError:
            q0 = s.lower()
    if any(t in q0 for t in TRACKPAD_TOKENS):
        return "trackpad"
    if any(t in q0 for t in MOUSE_TOKENS):
        return "mouse"
    return np.nan


def device_series(path: Optional[Union[str, Path]] = None) -> pd.Series:
    """run_id -> "mouse" | "trackpad" (NaN if unparseable)."""
    d = pd.read_csv(path or DATA / "input_devices.csv")[["run_id", "input_device"]]
    return d.assign(device=d.input_device.map(_parse)).set_index("run_id")["device"]


def device_codes(
    run_ids: Sequence[int], path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """Sum-coded device for an ordered run_id list: +0.5 trackpad, -0.5 mouse.

    Other coefficients keep their grand-mean interpretation; the device coefficient reads as
    (trackpad - mouse).
    """
    dev = device_series(path)
    lab = np.array([dev.get(r, None) for r in run_ids], dtype=object)
    if pd.isna(lab).any():
        raise ValueError(
            f"{int(pd.isna(lab).sum())} run_id(s) with unparseable pointer device"
        )
    return np.where(lab == "trackpad", 0.5, -0.5)
