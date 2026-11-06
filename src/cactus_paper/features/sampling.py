"""cactus_paper.features.sampling — per-subject sampling-allocation metrics from the mouse stream.

Parses one visit_sequences.csv string (collapsed AOI visit tokens `L_shape`, `R_color`, ...;
'none' = cursor outside all six AOIs, dropped) into per-trial allocation features (depth, first
dimension, per-dim visits, attribute-wise L<->R comparisons); used by make_stats.py.
Counts are raw AOI visit tokens, so per-trial depth here differs microscopically from
the fig_ms5 depth convention (which keeps 'none' tokens).
"""

from typing import Any, Dict, Union

from cactus_paper.constants import DIMS


def trial_visit_feats(seq: Any) -> Dict[str, Union[int, str]]:
    """Per-trial allocation features from one collapsed visit-sequence string."""
    toks = [t for t in str(seq).split("-") if t != "none"]
    if not toks:
        return {
            "depth": 0,
            "first_dim": "none",
            "ndims": 0,
            **{f"v_{d}": 0 for d in DIMS},
            **{f"aw_{d}": 0 for d in DIMS},
        }
    dims = [t.split("_", 1)[1] for t in toks]
    sides = [t[0] for t in toks]
    row = {"depth": len(toks), "first_dim": dims[0], "ndims": len(set(dims))}
    for d in DIMS:
        row[f"v_{d}"] = dims.count(d)
        # attribute-wise comparison: adjacent visits, same dim, opposite side
        row[f"aw_{d}"] = sum(
            d1 == d2 == d and s1 != s2
            for d1, d2, s1, s2 in zip(dims[:-1], dims[1:], sides[:-1], sides[1:])
        )
    return row
