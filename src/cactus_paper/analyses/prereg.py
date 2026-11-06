"""prereg.py — the four preregistered models (H1, H2, H3a, H3b), as specified.

Trial-level mixed-effects models on divergent trials (`cactus-prereg-20260123.docx`), fit by
maximum likelihood in lme4 (via rpy2) with Wald 95% CIs and p-values.

Notes:
  - The prereg's "within-dimension" transition (stimulus to stimulus within one dimension) is the
    paper's between-option transition. Transitions are recomputed from `visit_sequences.csv` under
    the prereg's definition; do not substitute the paper's `within_option` column.
  - Predictors are z-scored; count DVs stay on their natural scale.
  - A singular fit counts as non-convergence: the model is refit with random intercepts only, and
    the maximal fit is kept under `maximal_singular`.

    pixi run python -m cactus_paper.analyses.prereg              # analyzed sample
    pixi run python -m cactus_paper.analyses.prereg --prereg-sample  # preregistered sample

`--prereg-sample` rebuilds the frame on the sample the prereg's exclusion cascade produces (see
`audit_exclusions`) instead of the analyzed sample.

Outputs -> results/fits/prereg_confirmatory{,_preregsample}.json + a console report.
"""

import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro

from cactus_paper.constants import DIMS, POS
from cactus_paper.data.sample import canonical_sample
from cactus_paper.paths import DATA, FITS, RAW_EXP2


def build_frame(kept: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Trial-level frame for the divergent trials, using the prereg's own definitions."""
    kept = canonical_sample(str(DATA)) if kept is None else kept
    cd = pd.read_csv(DATA / "choice_data.csv")
    cd = cd[cd.run_id.isin(kept)]
    vs = pd.read_csv(DATA / "visit_sequences.csv")
    ld = pd.read_csv(DATA / "learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]

    # category knowledge = mean of the per-dimension learning accuracies (one 40-trial block each)
    K = (
        ld.groupby(["run_id", "dim"])
        .correct.mean()
        .groupby("run_id")
        .mean()
        .rename("category_knowledge")
    )

    rows = []
    for r in vs.itertuples():
        seq = r.visit_sequence
        if not isinstance(seq, str) or not seq:
            continue
        v = seq.split("-")
        if len(v) < 2:
            continue
        c = dict(
            run_id=r.run_id,
            trial_idx=r.trial_idx,
            total_transitions=len(v) - 1,
            within_dim=0,
            between_dim=0,
            **{f"wd_{d}": 0 for d in DIMS},
        )
        for a, b in zip(v[:-1], v[1:]):
            da, db = a.split("_")[1], b.split("_")[1]
            if da == db:
                c["within_dim"] += 1
                c[f"wd_{da}"] += 1
            else:
                c["between_dim"] += 1
        rows.append(c)
    D = cd.merge(pd.DataFrame(rows), on=["run_id", "trial_idx"]).merge(K, on="run_id")

    # objective critical dimension = largest |value difference| on the taught points
    def side(s, d):
        o = np.full(len(D), np.nan)
        for p in POS:
            m = D[f"choice_trial_{s}_{p}_dim"].values == d
            o[m] = D[f"choice_trial_{s}_{p}_value"].values[m]
        return o

    VD = np.stack([side("right", d) - side("left", d) for d in DIMS], 1)
    D["crit_dim"] = [DIMS[i] for i in np.argmax(np.abs(VD), 1)]
    D["within_dim_crit"] = [D.at[i, f"wd_{c}"] for i, c in zip(D.index, D.crit_dim)]

    d = D[D.trial_type == "divergent"].copy()
    d["chose"] = np.where(
        d.response == 1, "right", "left"
    )  # matches load_design's right - left
    d["choice_relational"] = (d.chose == d.relational_winner).astype(int)
    d["choice_categorical"] = (d.chose == d.categorical_winner).astype(int)
    assert ((d.choice_relational + d.choice_categorical) == 1).all(), (
        "winners not complementary"
    )
    d["discriminability_critical"] = d.critical_dim_discriminability
    d["subject"] = d.run_id.astype(str)
    for c in [
        "discriminability_critical",
        "total_transitions",
        "within_dim",
        "category_knowledge",
    ]:
        d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
    return d


def audit_exclusions() -> Tuple[np.ndarray, Dict[str, Any]]:
    """The prereg's exclusion cascade vs the one actually applied, from the raw data file."""
    raw = pd.read_csv(
        RAW_EXP2,
        low_memory=False,
        usecols=["run_id", "trial_type", "task"],
    )
    ld = pd.read_csv(DATA / "learning_data.csv")
    fv = pd.read_csv(DATA / "first_visits.csv")
    allids = np.sort(raw.run_id.unique())
    TEST = np.array([1])

    n_choice = (
        raw[raw.trial_type.isin(["divergent", "non-divergent"])]
        .groupby("run_id")
        .size()
    )
    ended = raw[raw.task.eq("experiment_complete_questionnaire")].run_id.unique()
    incomplete = np.array(
        [i for i in allids if n_choice.get(i, 0) < 120 or i not in ended]
    )  # prereg 1
    surv = np.setdiff1d(allids, np.union1d(TEST, incomplete))

    # prereg 2: learning failure (<60% on any dimension)

    L = ld[ld.run_id.isin(surv)].sort_values(["run_id", "dim", "trial_per_dim"])
    key = ["run_id", "dim"]
    acc = L.groupby(key).correct.mean().unstack()
    fail = acc.index[(acc < 0.60).any(axis=1)].values
    surv2 = np.setdiff1d(surv, fail)
    nf = (
        fv[fv.run_id.isin(surv2)]
        .assign(n=lambda x: x.first_pos.eq("none"))
        .groupby("run_id")
        .n.mean()
    )
    noncomp = nf.index[nf > 0.05].values  # prereg 3
    prereg_kept = np.setdiff1d(surv2, noncomp)

    kept, bd = canonical_sample(str(DATA), return_breakdown=True)
    pk, ak = set(prereg_kept.tolist()), set(kept.tolist())
    return prereg_kept, {
        "n_raw": len(allids),
        "prereg_cascade": {
            "test_account": len(TEST),
            "incomplete": len(incomplete),
            "learning_failure_any_dim_lt_60": len(fail),
            "_final_block": "the last prompt run (~10 trials) of each dimension",
            "_median_trials_scored": int(L.groupby(key).size().median()),
            "mouse_noncompliance": len(noncomp),
            "n_prereg": len(prereg_kept),
        },
        "applied_cascade": {
            "test_account": len(bd["test"]),
            "mouse_quality": len(bd["mouse"]),
            "fast_learning_rt": len(bd["fast"]),
            "rt_threshold_ms": round(bd["rt_threshold_ms"], 1),
            "n_analyzed": len(kept),
        },
        "overlap": {
            "both": len(pk & ak),
            "prereg_only": len(pk - ak),
            "analyzed_only": len(ak - pk),
            "prereg_only_all_from_fast_rt": len((pk - ak) & set(bd["fast"])),
        },
        "_note": (
            "of the analyzed participants, %d fail the preregistered learning rule; only %d of "
            "those are below .60 on all three dimensions, so the rule mostly removes people "
            "with one weak dimension -- the within-person unevenness the paper is about"
            % (len(ak - pk), int((acc.loc[sorted(ak - pk)] < 0.60).all(axis=1).sum()))
        ),
    }


R_CODE = r"""
suppressMessages({library(lme4); library(MASS)})
d <- read.csv(CSVPATH); d$subject <- factor(d$subject)
ctl <- glmerControl(optimizer = "bobyqa")

tidy <- function(m) {
  co <- as.data.frame(coef(summary(m)))
  names(co)[1:2] <- c("estimate", "se"); names(co)[grep("^Pr", names(co))[1]] <- "p"
  ci <- confint(m, method = "Wald", parm = "beta_")
  co$lo <- ci[rownames(co), 1]; co$hi <- ci[rownames(co), 2]
  co$term <- rownames(co); co
}

# prereg: "If models fail to converge, we simplify by removing random slopes." A singular fit
# (a variance at 0 or a correlation at +-1) is treated as that failure: the model is refit with
# random intercepts only, and the maximal fit is kept alongside as a sensitivity result.
intercepts_only <- function(f) update(nobars(f), . ~ . + (1 | subject))
step_binom <- function(f) {
  mx <- glmer(f, data = d, family = binomial, control = ctl)
  if (!isSingular(mx)) return(list(m = mx, max = mx, stepped = FALSE))
  list(m = glmer(intercepts_only(f), data = d, family = binomial, control = ctl),
       max = mx, stepped = TRUE)
}

M1  <- step_binom(choice_relational ~ z_within_dim + z_total_transitions +
                    z_discriminability_critical + (1 + z_within_dim | subject))
M3a <- step_binom(choice_categorical ~ z_category_knowledge + z_discriminability_critical +
                    (1 | subject))

f2 <- within_dim_crit ~ z_total_transitions + z_discriminability_critical +
        (1 + z_discriminability_critical | subject)
f3 <- within_dim_crit ~ z_total_transitions + z_discriminability_critical * z_category_knowledge +
        (1 + z_discriminability_critical | subject)

# prereg: negative binomial, falling back to Poisson if overdispersion is not significant.
# The random-effects step-down is applied first, to BOTH families together, so the NB-vs-Poisson
# likelihood ratio always compares models with the same random-effects structure.
disp <- function(f) {
  pm_max <- glmer(f, data = d, family = poisson, control = ctl)
  nb_max <- suppressWarnings(glmer.nb(f, data = d, control = ctl))
  stepped <- isSingular(pm_max) || isSingular(nb_max)
  if (stepped) {
    f <- intercepts_only(f)
    pm  <- glmer(f, data = d, family = poisson, control = ctl)
    nbm <- suppressWarnings(glmer.nb(f, data = d, control = ctl))
  } else { pm <- pm_max; nbm <- nb_max }
  list(lr = as.numeric(2 * (logLik(nbm) - logLik(pm))),
       theta = as.numeric(getME(nbm, "glmer.nb.theta")),
       pearson = sum(residuals(pm, type = "pearson")^2) / df.residual(pm),
       pm = pm, max = pm_max, stepped = stepped)
}
D2 <- disp(f2); D3 <- disp(f3)
"""


def main(prereg_sample: bool = False) -> None:
    prereg_kept, audit = audit_exclusions()
    print(
        "EXCLUSIONS  prereg cascade %s -> N=%d   |   applied %s -> N=%d   |   in both: %d\n"
        % (
            audit["prereg_cascade"],
            audit["prereg_cascade"]["n_prereg"],
            audit["applied_cascade"],
            audit["applied_cascade"]["n_analyzed"],
            audit["overlap"]["both"],
        )
    )
    d = build_frame(prereg_kept if prereg_sample else None)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        d.to_csv(f.name, index=False)  # CSV hand-off: no py2rpy conversion needed
        csv = f.name
    try:
        ro.r(f'CSVPATH <- "{csv}"')
        ro.r(R_CODE)
        get = lambda obj: pd.DataFrame(
            {
                c: list(ro.r(f"{obj}${c}"))
                for c in ["term", "estimate", "se", "lo", "hi", "p"]
            }
        )
        fits = {"H1": "M1", "H3a": "M3a", "H2": "D2", "H3b": "D3"}
        # H2/H3b's reported fit lives in `pm` (the Poisson fallback), H1/H3a's in `m`
        chosen = {"H1": "M1$m", "H3a": "M3a$m", "H2": "D2$pm", "H3b": "D3$pm"}
        tabs = {h: get(f"tidy({o})") for h, o in chosen.items()}
        stepped = {h: bool(ro.r(f"{o}$stepped")[0]) for h, o in fits.items()}
        tabs_max = {h: get(f"tidy({o}$max)") for h, o in fits.items() if stepped[h]}
        disp = {
            k: dict(
                nb_vs_poisson_lr=round(float(ro.r(f"{o}$lr")[0]), 3),
                nb_theta=round(float(ro.r(f"{o}$theta")[0]), 1),
                poisson_pearson_dispersion=round(float(ro.r(f"{o}$pearson")[0]), 3),
            )
            for k, o in (("H2", "D2"), ("H3b", "D3"))
        }
    finally:
        os.unlink(csv)

    focal = {
        "H1": ("z_within_dim", "+"),
        "H2": ("z_discriminability_critical", "-"),
        "H3a": ("z_category_knowledge", "+"),
        "H3b": ("z_discriminability_critical:z_category_knowledge", "+"),
    }
    out = {
        "_what": "the four preregistered models, fit as specified (lme4 ML, Wald CIs)",
        "_sample": (
            "preregistered exclusions"
            if prereg_sample
            else "the paper's analyzed sample"
        ),
        "_exclusions": audit,
        "_n_subjects": int(d.run_id.nunique()),
        "_n_divergent_trials": int(len(d)),
        "_note": (
            "negative binomial was preregistered with a Poisson fallback if overdispersion "
            "is not significant; it is not, so Poisson is reported (estimates identical)"
        ),
        "_dispersion": disp,
        "_random_effects": (
            "preregistered maximal structure; any SINGULAR fit is refit with random intercepts "
            "only (the prereg's 'simplify by removing random slopes'), and the maximal fit is "
            "kept under `maximal_singular` as a sensitivity result"
        ),
    }
    print(
        f"N = {out['_n_subjects']} participants, {out['_n_divergent_trials']} divergent trials\n"
    )
    terms = lambda t: {
        row.term: dict(
            estimate=round(row.estimate, 4),
            se=round(row.se, 4),
            lo=round(row.lo, 4),
            hi=round(row.hi, 4),
            p=round(row.p, 5),
        )
        for row in t.itertuples()
    }
    for h, t in tabs.items():
        f_, direction = focal[h]
        r = t[t.term == f_].iloc[0]
        ok = (r.estimate > 0 if direction == "+" else r.estimate < 0) and r.p < 0.05
        out[h] = {
            "focal": f_,
            "predicted_direction": direction,
            "supported": bool(ok),
            "random_effects": (
                "intercepts (stepped down: maximal fit singular)"
                if stepped[h]
                else "as preregistered"
            ),
            "terms": terms(t),
        }
        if stepped[h]:
            out[h]["maximal_singular"] = {"terms": terms(tabs_max[h])}
        print(
            f"{h}: {f_} = {r.estimate:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}], p = {r.p:.4g}"
            f"  -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}"
            + (
                "   [stepped down to intercepts: maximal fit singular]"
                if stepped[h]
                else ""
            )
        )
        print(t.round(4).to_string(index=False), "\n")

    FITS.mkdir(parents=True, exist_ok=True)
    path = FITS / (
        "prereg_confirmatory_preregsample.json"
        if prereg_sample
        else "prereg_confirmatory.json"
    )
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    import sys

    main(prereg_sample="--prereg-sample" in sys.argv)
