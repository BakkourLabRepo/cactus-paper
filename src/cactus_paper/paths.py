"""cactus_paper.paths — repo-anchored paths for every exp2 artifact, so code runs from any cwd.

Everything the paper reads lives in FITS (results/fits/): JSON summaries tracked, idata / npz /
npy draws untracked.
"""

from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "exp2"
RAW_EXP2 = ROOT / "data" / "raw" / "exp2" / "cactus-v2.csv"
FITS = ROOT / "results" / "fits"
UDIM_CACHE = FITS / "udim_cache.npz"


# Model of record (mor): per-participant monotone value function u_sd
# (endpoints pinned) x per-attribute weights x lambda, plus a within-band map slope s_d; the
# ordinal channel reads the taught category difference. Latents w_mu/w_sd/c0/sigma_lam/a0/sda/
# lam_s/W. No frac_s deterministic — reconstruct increments as softmax(psi_mu + sigma_psi * z_psi).
MOR_IDATA = FITS / "mor_idata.nc"

# per-attribute weights x lambda (models.comparison.weighted_linear); read by models.checks
WL_IDATA = FITS / "weighted_linear_idata.nc"


FIGURES = ROOT / "results" / "figures" / "exp2"


def fig(name: str, base: Optional[Union[str, Path]] = None) -> Path:
    """Path for a figure output (mkdir -p the directory); default base = FIGURES."""
    base = Path(base) if base is not None else FIGURES
    base.mkdir(parents=True, exist_ok=True)
    return base / name
