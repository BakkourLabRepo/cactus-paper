"""Assemble stats.json: load the shared context once, run every section in paper order, merge.

Section order is the stats.json key order. Every refit carries its own seed, so the order does not
change any number.
"""

import os
from typing import Any, Dict

from cactus_paper.manuscript import (
    choice,
    knowledge,
    learning,
    model,
    search,
    validation,
)
from cactus_paper.manuscript.context import Context

SECTIONS = [
    model.header,
    model.constructs,
    model.design,
    learning.learning,
    learning.payne,
    choice.fig3,
    knowledge.fig4,
    search.fig5,
    validation.validation,
    validation.continuum,
    choice.readout_profiles,
]


def build(idata: str, quick: bool = False, tag: str = "") -> Dict[str, Any]:
    ctx = Context.load(idata, quick=quick, tag=tag)
    if quick:
        print(
            "*** MAKE_STATS_QUICK: smoke-test sampler scale. Do not ship this stats.json. ***"
        )
    stats = {}
    for section in SECTIONS:
        stats.update(section(ctx))

    if tag:
        stats["_stale"] = {
            "_what": "blocks NOT recomputed for this sample -- they come from precomputed JSON written "
            "by separate CV/LOSO scripts that were not re-run",
            "blocks": ["cv", "recovery", "identifiability", "device"],
            "sample_file": os.environ.get("CACTUS_SAMPLE_FILE", ""),
        }
    # the canonical stats.json also carries the `fits` and `prereg` namespaces (precomputed artifacts
    # for the full sample, so they are not folded into the quick or re-sampled variants)
    if not quick and not tag:
        from cactus_paper.manuscript.fits import fits_namespace
        from cactus_paper.manuscript.prereg import prereg_namespace

        print("\n[fits] folding in results/fits/ ...", flush=True)
        stats["fits"] = fits_namespace()
        print("[prereg] folding in prereg artifacts ...", flush=True)
        stats["prereg"] = prereg_namespace()
    return stats


def report(stats: Dict[str, Any]) -> None:
    """Console summary of the headline numbers."""
    print(f"  design: {stats['design']}")
    print(
        "  w_mu: "
        + "; ".join(f"{d} {v['mean']:+.3f}" for d, v in stats["w_mu"].items())
        + f"   w_sd {stats['w_sd']['mean']:+.3f}"
    )
    print(
        f"  corr(lam, gain) {stats['corr_lam_gain']['mean']:+.3f}   "
        f"corr(lam, conc) {stats['corr_lam_conc']['mean']:+.3f}"
    )
    kn = stats["knowledge"]
    print(
        f"  knowledge (propagated): gain {kn['corr_gain_K']['mean']:+.3f} "
        f"[{kn['corr_gain_K']['lo']:+.3f},{kn['corr_gain_K']['hi']:+.3f}]  "
        f"lam {kn['corr_lam_K']['mean']:+.3f}"
    )
    print(
        f"  knowledge (point est, = the figure): gain {kn['corr_gain_K_pointest']:+.3f}  "
        f"lam {kn['corr_lam_K_pointest']:+.3f}  "
        f"weight(2x-centered) {kn['corr_weight_dacc_double_centered']:+.3f}"
    )
    ld = stats["fig3_ladder"]
    print(
        f"  ladder: {ld['worst']['mean']:.3f} / {ld['middle']['mean']:.3f} / {ld['best']['mean']:.3f}"
        f"   PPC slope {ld['slope_ppc']['mean']:+.3f} "
        f"[{ld['slope_ppc']['lo']:+.3f}, {ld['slope_ppc']['hi']:+.3f}] vs observed {ld['slope_obs']:+.3f}"
    )
    for k in [
        "fig5_style",
        "fig5_amount",
        "fig5_dissociation",
        "fig5_alloc",
        "fig5_alloc_adjK",
    ]:
        print(
            f"  {k}: "
            + "; ".join(
                f"{f}={v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] P={v['pdir']:.2f}"
                for f, v in stats[k].items()
            )
        )
    c = stats["cv"]
    print(
        f"  cv  lambda (matched, participant_map_lambda - participant_map) "
        f"{c['participant_map_lambda_minus_participant_map']['delta']:+.0f}"
        f" +/- {c['participant_map_lambda_minus_participant_map']['se']:.0f}"
        f" | participant-level u (- group_map_lambda) "
        f"{c['participant_map_lambda_minus_group_map_lambda']['delta']:+.0f}"
        f" | vs WL {c['participant_map_lambda_minus_weighted_linear']['delta']:+.0f}"
    )
    r = stats["recovery"]
    print(
        f"  recovery: {r['n_reps']} reps, weights r >= {r['r_min_across_weights']:.3f}, "
        f"coverage {r['coverage_mean']:.2f}  ({r['_caveat']})"
    )
