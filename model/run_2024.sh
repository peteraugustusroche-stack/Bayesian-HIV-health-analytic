#!/usr/bin/env bash
# Re-run the 2024 cascade-coverage arm against the CURRENT model.
#
# Only System 2 depends on cascade_year. System 1 is drawn from its priors and
# never fitted, so data/results/system1.npz is reused as-is; because both
# scripts are seeded (seed=1), the 2024 arm resamples the SAME five System 1
# draws as the 2019 base case, and the two arms differ only in the p1/p2/p3
# targets.
#
# Outputs are year-suffixed, so this cannot overwrite the 2019 base case:
#   data/results/system2_sens2024.npz
#   data/results/posterior_analysis_sens2024.npz
#
# Runtime is about 15 minutes (11 for the calibration, 4 for the posterior).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
LOGS="$REPO/logs"; mkdir -p "$LOGS"
STAMP="$(date +%Y%m%d_%H%M)"

# One BLAS thread per process: 4 chains are 4 OS processes, and the hot
# operation is a 107-vector times a 107x107 matrix -- far too small for
# threading to pay for itself. See the long note in run_all.sh.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Without this, Python block-buffers stdout when piped and the per-outer-draw
# progress lines sit invisible in a buffer while PyMC's stderr banner comes
# through immediately -- which is what made the last attempt look hung.
export PYTHONUNBUFFERED=1

if [ -x "$REPO/.venv/bin/python3" ]; then
  PY="$REPO/.venv/bin/python3"
else
  echo "No .venv at $REPO/.venv" >&2; exit 1
fi

echo "python: $($PY -V 2>&1)"
echo "repo:   $REPO"

if [ ! -f "$REPO/data/results/system1.npz" ]; then
  echo "data/results/system1.npz missing -- run calibrate_system1.py first." >&2
  exit 1
fi

# Guard: the retention arm swaps likelihoods.py. If a swap was left in place
# the 2024 run would be retention-anchored without saying so.
if ! grep -q '^USE_RETENTION_LIKELIHOOD = False' "$REPO/likelihoods.py"; then
  echo "likelihoods.py is not in the UNAIDS-anchored state -- check it before running." >&2
  exit 1
fi

echo
echo "[1/2] System 2 calibration, cascade_year=2024  (~11 min)"
"$PY" calibrate_system2.py 2024 2>&1 | tee "$LOGS/s2_2024_${STAMP}.log"

echo
echo "[2/2] Posterior health-economic analysis, cascade_year=2024  (~4 min)"
"$PY" run_posterior_analysis.py 2024 2>&1 | tee "$LOGS/post_2024_${STAMP}.log"

echo
echo "done."
echo "  data/results/system2_sens2024.npz"
echo "  data/results/posterior_analysis_sens2024.npz"
echo "  logs: $LOGS/s2_2024_${STAMP}.log, $LOGS/post_2024_${STAMP}.log"
