"""udim_cache.py — regenerate the subjective-value cache.

Fits the anchored hierarchical-Bayes utility model (cactus_paper.models.subjective; seed 20260618,
1000 warmup / 1000 draws / 4 chains) on the canonical sample and writes the per-(subject, dim)
value shape `u` (centered per subject) to `artifacts/udim_cache.npz`, keys `u_<run_id>_<dim>`.
Also writes `udim_cache_scaled.npz` (beta*u, same keys).

Run from repo root: python -m cactus_paper.features.udim_cache
"""

import os

import numpy as np
import numpyro
import pandas as pd

from cactus_paper.data.sample import canonical_sample
from cactus_paper.models.subjective import fit_u_dim_draws
from cactus_paper.paths import DATA, UDIM_CACHE

OUT = UDIM_CACHE
SEED, CHAINS, DRAWS, WARM = 20260618, 4, 1000, 1000


if __name__ == "__main__":
    numpyro.set_host_device_count(
        4
    )  # parallel chains; must run before jax's first computation
    kept = canonical_sample(str(DATA))
    ld = pd.read_csv(DATA / "learning_data.csv")
    ld = ld[ld.run_id.isin(kept)]
    print(
        f"fitting stage-1 u on canonical sample (N={ld.run_id.nunique()}) ...",
        flush=True,
    )
    post = fit_u_dim_draws(ld, seed=SEED, warmup=WARM, samples=DRAWS, chains=CHAINS)
    arrs, arrs_scaled = {}, {}
    for dim, p in post.items():
        upm = p["u"].mean(0)
        upm = upm - upm.mean(axis=1, keepdims=True)  # center per subject (SHAPE only)
        # beta*u: per-dimension sensitivity times shape, product taken per draw then averaged
        bu = (p["beta"][:, :, None] * p["u"]).mean(0)
        bu = bu - bu.mean(axis=1, keepdims=True)
        for i, rid in enumerate(p["suniq"]):
            arrs[f"u_{rid}_{dim}"] = np.asarray(upm[i], float)
            arrs_scaled[f"u_{rid}_{dim}"] = np.asarray(bu[i], float)
    os.makedirs(os.path.dirname(str(OUT)), exist_ok=True)
    np.savez(OUT, **arrs)
    scaled_path = str(OUT).replace(".npz", "_scaled.npz")
    np.savez(scaled_path, **arrs_scaled)
    print(f"saved -> {OUT}  ({len(arrs)} subject×dim cells)")
    print(f"saved -> {scaled_path}  (beta*u, same keys)")
