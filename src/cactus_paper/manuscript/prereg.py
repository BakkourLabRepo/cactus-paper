"""stats.json `prereg` namespace: the preregistered-confirmatory results.

Reads the two artifacts written by cactus_paper.analyses.prereg; does not refit anything.
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from cactus_paper.paths import FITS

HYPS = ["H1", "H2", "H3a", "H3b"]
GENERATED_BY = "cactus_paper.manuscript.prereg"


def block(fname: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    d = json.load(open(os.path.join(FITS, fname)))
    out = {
        "n_subjects": d["_n_subjects"],
        "n_divergent_trials": d["_n_divergent_trials"],
    }
    for h in HYPS:
        v = d[h]
        t = v["terms"][v["focal"]]
        out[h] = {
            "focal": v["focal"],
            "predicted": v["predicted_direction"],
            "supported": bool(v["supported"]),
            "estimate": t["estimate"],
            "lo": t["lo"],
            "hi": t["hi"],
            "p": t["p"],
            "random_effects": v.get("random_effects", "as preregistered"),
        }
        # singular maximal fit -> stepped down to intercepts; keep the maximal version too
        if "maximal_singular" in v:
            mt = v["maximal_singular"]["terms"][v["focal"]]
            out[h]["maximal_singular"] = {
                k: mt[k] for k in ("estimate", "lo", "hi", "p")
            }
    if "_dispersion" in d:
        out["dispersion"] = d["_dispersion"]
    return out, d.get("_exclusions")


def prereg_namespace() -> Dict[str, Any]:
    """The `prereg` block of stats.json."""
    analyzed, excl = block("prereg_confirmatory.json")
    prereg, _ = block("prereg_confirmatory_preregsample.json")

    # parse "only N of those are below .60 on ALL three dimensions" from the exclusions note
    m = re.search(r"only (\d+) of those are below", excl.get("_note", ""))
    if not m:
        raise SystemExit(
            "could not parse the all-three-dimensions count from _exclusions._note"
        )
    excl["overlap"]["analyzed_only_failing_all_three"] = int(m.group(1))
    out = {
        "_generated_by": GENERATED_BY,
        "_source": "cactus_paper.analyses.prereg artifacts",
        "analyzed": analyzed,
        "prereg_sample": prereg,
        "exclusions": excl,
        "n_supported_analyzed": sum(analyzed[h]["supported"] for h in HYPS),
        "n_supported_prereg": sum(prereg[h]["supported"] for h in HYPS),
    }

    print(
        f"  prereg: analyzed N = {analyzed['n_subjects']}, prereg N = {prereg['n_subjects']}"
    )
    for h in HYPS:
        print(
            f"    {h:4s} analyzed {analyzed[h]['estimate']:+.4f} p={analyzed[h]['p']:.4f} "
            f"{'OK ' if analyzed[h]['supported'] else '   '} | "
            f"prereg {prereg[h]['estimate']:+.4f} p={prereg[h]['p']:.4f} "
            f"{'OK' if prereg[h]['supported'] else ''}"
        )
    print(
        f"    supported: {out['n_supported_analyzed']}/4 analyzed, "
        f"{out['n_supported_prereg']}/4 prereg"
    )
    return out
