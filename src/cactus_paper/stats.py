"""cactus_paper.stats — posterior-summary helper.

Direction-probability key names are the contract:
    pdir, p_dir  = max(P(t>0), P(t<0))   two-sided, in [0.5, 1.0]
    p_pos, p_gt0 = P(t > 0)              one-sided, in [0.0, 1.0]
    p_neg        = P(t < 0)              one-sided
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

_DIRECTION = {
    "pdir": lambda s: max(float((s > 0).mean()), float((s < 0).mean())),
    "p_dir": lambda s: max(float((s > 0).mean()), float((s < 0).mean())),
    "p_pos": lambda s: float((s > 0).mean()),
    "p_gt0": lambda s: float((s > 0).mean()),
    "p_neg": lambda s: float((s < 0).mean()),
}


def summarize(
    draws: Any,
    nd: Optional[int] = None,
    direction: Optional[Union[str, Tuple[str, ...]]] = "pdir",
    nd_dir: Optional[int] = 3,
    interval: str = "hdi",
    hdi_prob: float = 0.95,
    drop_nonfinite: bool = False,
    hdi_as_list: bool = False,
) -> Dict[str, Any]:
    """Posterior mean + 95% interval + a probability of direction.

    draws          array-like of posterior draws; flattened.
    nd             decimals for mean/lo/hi. None = full precision.
    direction      "pdir" | "p_dir" | "p_pos" | "p_gt0" | "p_neg", a tuple of those, or None.
                   pdir/p_dir are two-sided; p_pos/p_gt0/p_neg are one-sided.
    nd_dir         decimals for the direction probability. None = full precision.
    interval       "hdi" (arviz highest-density) or "eti" (equal-tailed percentile).
    drop_nonfinite drop NaN/inf before summarizing.
    hdi_as_list    emit {"hdi95": [lo, hi]} instead of {"lo":…, "hi":…}.
    """
    s = np.asarray(draws, dtype=float).ravel()
    if drop_nonfinite:
        s = s[np.isfinite(s)]

    if interval == "hdi":
        import arviz as az

        lo, hi = az.hdi(s, hdi_prob=hdi_prob)
    elif interval == "eti":
        # round so the tail is exactly 2.5 (float arithmetic gives 2.5000000000000022)
        tail = round(50.0 * (1.0 - hdi_prob), 10)
        lo, hi = np.percentile(s, [tail, 100.0 - tail])
    else:
        raise ValueError(f"interval must be 'hdi' or 'eti', got {interval!r}")

    r = (lambda v: float(v)) if nd is None else (lambda v: round(float(v), nd))
    out = {"mean": r(s.mean())}
    if hdi_as_list:
        out["hdi95"] = [r(lo), r(hi)]
    else:
        out["lo"], out["hi"] = r(lo), r(hi)

    if direction:
        rd = (
            (lambda v: float(v))
            if nd_dir is None
            else (lambda v: round(float(v), nd_dir))
        )
        for key in (direction,) if isinstance(direction, str) else direction:
            if key not in _DIRECTION:
                raise ValueError(
                    f"unknown direction key {key!r}; one of {sorted(_DIRECTION)}"
                )
            out[key] = rd(_DIRECTION[key](s))
    return out


def rep(po: Any, name: str, expect: str = "", w: int = 11) -> None:
    """One-line posterior report: mean [95% HDI] P(direction)."""
    import arviz as az

    s = po[name].values.ravel()
    lo, hi = az.hdi(s, hdi_prob=0.95)
    pdir = max((s > 0).mean(), (s < 0).mean())
    print(
        f"  {name:{w}s} {s.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  P(dir)={pdir:.3f}  {expect}"
    )
