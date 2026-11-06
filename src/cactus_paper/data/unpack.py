"""Unpack the de-identified data in data/shared/ into the paths the analyses read.

data/shared/{raw,processed}/exp*/<name>.csv.gz -> data/{raw,processed}/exp*/<name>.csv
(other files, e.g. prereg_kept_ids.txt, are copied as is). Files that already exist are
never overwritten, so a local copy of the original data stays untouched.

    pixi run unpack-data
"""

import gzip
import shutil
from pathlib import Path
from typing import List

from cactus_paper.paths import ROOT

SHARED = ROOT / "data" / "shared"


def unpack(shared: Path = SHARED, dest: Path = ROOT / "data") -> List[Path]:
    written = []
    for src in sorted(shared.glob("*/exp*/*")):
        rel = src.relative_to(shared)
        out = dest / rel.parent / (src.stem if src.suffix == ".gz" else src.name)
        if out.exists():
            print(f"exists, skipped: {out.relative_to(ROOT)}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".gz":
            with gzip.open(src, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copyfile(src, out)
        written.append(out)
        print(f"wrote {out.relative_to(ROOT)}")
    return written


if __name__ == "__main__":
    unpack()
