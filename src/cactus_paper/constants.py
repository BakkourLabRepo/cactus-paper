"""cactus_paper.constants — task-design constants, defined once and imported everywhere.

Pure Python, no imports, so the light manuscript scripts can use it without the modeling stack.
Values mirror the task code, experiment_code/exp2/src/config.js.
"""

# --- dimensions, display rows, taught values ----------------------------------------------------
DIMS = ["color", "shape", "line"]
POS = [
    "top",
    "mid",
    "bot",
]  # display rows; dimension-to-row order is randomized per trial
LEVELS = [0.0, 25, 50, 75, 100]  # taught point values, one per category band

# --- stimulus geometry --------------------------------------------------------------------------
# Category bands, per config.js `categories` (min_val/max_val): 31 inclusive integer angles (a 30 deg
# span) with 19 deg gaps over a 1-231 arc, INCLUSIVE bounds. (lo, hi, category index).
BANDS = [(1, 31, 0), (51, 81, 1), (101, 131, 2), (151, 181, 3), (201, 231, 4)]
BAND_WIDTH = 30.0  # within-band span in degrees
CAT_SPACING = 50.0  # band center-to-center, degrees
ARC = 230.0  # stimulus arc in degrees (1-231)
BAND_CENTERS = [(lo + hi) / 2 for lo, hi, _ in BANDS]  # 16, 66, 116, 166, 216

# A locally angle-linear map makes a full within-band traverse worth (30/50) x 0.25 = 0.15, since
# u_d(k) is anchored at the band CENTER and the four increments sum to 1 (mean 0.25).
FULL_RAMP = (BAND_WIDTH / CAT_SPACING) * 0.25

# --- cross-validation ---------------------------------------------------------------------------
CV_SEED = 2026  # fold assignment (rng.permutation(N) then np.array_split) and per-fold PRNG base
