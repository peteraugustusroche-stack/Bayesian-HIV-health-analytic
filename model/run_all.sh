#!/usr/bin/env bash
#
# run_all.sh -- regenerate the base case and run the two-arm calibration
# experiment: UNAIDS-anchored vs retention-anchored.
#
# ARM A (unaids)     likelihood = p1, p2, p3, Supp_2L, CD4 at initiation.
#                    Retention and re-engagement reported as diagnostics only.
#                    This is the current base case.
#
# ARM B (retention)  the same, plus Ret_12m / Ret_24m and Bagayoko's
#                    Reeng_1y/2y/3y as likelihood terms. likelihoods.py argues
#                    these cannot be satisfied jointly with UNAIDS coverage;
#                    this arm quantifies what happens when you try.
#
# The two arms answer: how much of the headline DALY figure is a consequence
# of which cascade data you chose to believe?
#
# The script never leaves likelihoods.py modified -- the original is restored
# on exit, including on Ctrl-C or failure.
#
# Stages are skipped when their outputs already exist, so an interrupted run
# can simply be restarted. Use --force to redo everything.
#
#   ./run_all.sh                 both arms, figures, comparison
#   ./run_all.sh --base-only     arm A only (base-case refresh)
#   ./run_all.sh --force         ignore existing outputs, redo from scratch
#   ./run_all.sh --skip-figures  no figure/table regeneration
#
# Expect a couple of hours: each System 2 calibration is four DEMetropolisZ
# chains wrapped in an outer resample of System 1's locked posterior.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

RESULTS="$REPO/data/results"
ARM_A="$RESULTS/arm_unaids"
ARM_B="$RESULTS/arm_retention"
LOGS="$REPO/logs"
STAMP="$(date +%Y%m%d_%H%M)"

FORCE=0; BASE_ONLY=0; SKIP_FIGURES=0
FAILED_STAGES=""
for arg in "$@"; do
  case "$arg" in
    --force)        FORCE=1 ;;
    --base-only)    BASE_ONLY=1 ;;
    --skip-figures) SKIP_FIGURES=1 ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGS" "$ARM_A" "$ARM_B"

# --------------------------------------------------------------------------
# Thread pinning -- MUST be exported before python starts.
#
# calibrate_system2 runs 4 chains as 4 OS processes. Each one imports numpy,
# which on macOS links Accelerate/vecLib and spawns its own thread pool. The
# result is four processes at ~280% CPU each -- eleven cores' worth of work
# for four cores' worth of arithmetic, with the surplus going to thread
# synchronisation and cache contention between the processes.
#
# There is nothing to gain from threaded BLAS here: the hot operation is a
# 107-element vector times a 107x107 matrix, about 11k flops, far below the
# size where threading pays for itself. One thread per process is strictly
# faster, and lets the 4 chains actually run in parallel on 4 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1          # macOS Accelerate
export NUMEXPR_NUM_THREADS=1

# Python block-buffers stdout when piped into tee, so progress messages sat
# invisible in a buffer while stderr (PyMC's) came through immediately -- which
# is why "Running System 2 calibration..." appeared AFTER PyMC's banner.
export PYTHONUNBUFFERED=1

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

# --------------------------------------------------------------------------
# Restore likelihoods.py on any exit path
# --------------------------------------------------------------------------
LK="$REPO/likelihoods.py"
LK_BACKUP="$REPO/.likelihoods.py.orig.$STAMP"
cp "$LK" "$LK_BACKUP"
restore_lk() {
  if [ -f "$LK_BACKUP" ]; then
    cp -f "$LK_BACKUP" "$LK"
    rm -f "$LK_BACKUP"
    note "likelihoods.py restored to its original state"
  fi
}
trap restore_lk EXIT INT TERM

set_flags() {   # $1 = True|False, applied to both retention and re-engagement
  perl -pi -e "s/^USE_RETENTION_LIKELIHOOD\s*=.*/USE_RETENTION_LIKELIHOOD = $1/" "$LK"
  perl -pi -e "s/^USE_REENGAGEMENT_LIKELIHOOD\s*=.*/USE_REENGAGEMENT_LIKELIHOOD = $1/" "$LK"
  local r e
  r=$(grep -m1 '^USE_RETENTION_LIKELIHOOD'   "$LK")
  e=$(grep -m1 '^USE_REENGAGEMENT_LIKELIHOOD' "$LK")
  note "$r"
  note "$e"
}

run_stage() {   # run_stage <output-file-to-test> <logname> <command...>
  local out="$1" logname="$2"; shift 2
  if [ "$FORCE" -eq 0 ] && [ -e "$out" ]; then
    note "skipping ($(basename "$out") already exists; --force to redo)"
    return 0
  fi
  note "-> $* "
  "$@" 2>&1 | tee "$LOGS/${logname}_${STAMP}.log"
}

run_stage_optional() {   # same, but a failure warns instead of aborting
  local out="$1" logname="$2"; shift 2
  if [ "$FORCE" -eq 0 ] && [ -e "$out" ]; then
    note "skipping ($(basename "$out") already exists; --force to redo)"
    return 0
  fi
  note "-> $* "
  if ! "$@" 2>&1 | tee "$LOGS/${logname}_${STAMP}.log"; then
    printf '   \033[33m!! %s failed -- continuing (see %s)\033[0m\n' \
      "$1 $2" "$LOGS/${logname}_${STAMP}.log"
    FAILED_STAGES="${FAILED_STAGES}${logname} "
  fi
}

stash() {       # stash <destination-dir> <file...>
  local dest="$1"; shift
  for f in "$@"; do
    [ -f "$RESULTS/$f" ] && cp -f "$RESULTS/$f" "$dest/$f"
  done
}

# --------------------------------------------------------------------------
say "Environment"
# --------------------------------------------------------------------------
if [ -x "$REPO/.venv/bin/python3" ]; then
  PY="$REPO/.venv/bin/python3"
else
  echo "No .venv found at $REPO/.venv -- create it or edit PY in this script." >&2
  exit 1
fi
note "python: $($PY -V 2>&1)"
note "pymc:   $($PY -c 'import pymc; print(pymc.__version__)' 2>/dev/null || echo 'NOT INSTALLED -- will fall back to the numpy sampler')"

# The writeup scripts hardcode /tmp/hm as both import root and output dir.
ln -sfn "$REPO" /tmp/hm
note "/tmp/hm -> $REPO"

# --------------------------------------------------------------------------
say "Backing up current results"
# --------------------------------------------------------------------------
BACKUP="$REPO/data/results_backup_$STAMP"
cp -R "$RESULTS" "$BACKUP"
note "$BACKUP"

# --------------------------------------------------------------------------
say "System 1 -- natural history draws (shared by both arms)"
# --------------------------------------------------------------------------
note "USE_SYSTEM1_LIKELIHOOD is False, so this draws prior uncertainty rather"
note "than fitting; the three natural-history targets are out-of-sample checks."
run_stage "$RESULTS/system1.npz" s1 "$PY" calibrate_system1.py

# --------------------------------------------------------------------------
say "ARM A -- UNAIDS-anchored (base case)"
# --------------------------------------------------------------------------
set_flags False
run_stage "$ARM_A/system2.npz"           s2_unaids   "$PY" calibrate_system2.py 2019
stash "$ARM_A" system2.npz

run_stage "$ARM_A/prior_analysis.npz"    prior_unaids "$PY" run_prior_analysis.py
stash "$ARM_A" prior_analysis.npz scenario_prior_point.npz

run_stage "$ARM_A/posterior_analysis.npz" post_unaids "$PY" run_posterior_analysis.py
stash "$ARM_A" posterior_analysis.npz

run_stage_optional "$ARM_A/scenario_uncertainty.npz" scen_unaids "$PY" scenario_analysis.py
stash "$ARM_A" scenario_point.npz scenario_uncertainty.npz scenario_prob.npz

if [ "$BASE_ONLY" -eq 1 ]; then
  say "Base case complete (--base-only)"
  [ "$SKIP_FIGURES" -eq 0 ] && { for s in fig_structure fig_inputs fig_results fig_arms tables; do
      note "-> writeup/$s.py"; "$PY" "writeup/$s.py" || note "!! $s failed - continuing"; done; }
  exit 0
fi

# --------------------------------------------------------------------------
say "ARM B -- retention-anchored"
# --------------------------------------------------------------------------
note "Adding Ret_12m/Ret_24m and Reeng_1y/2y/3y to the likelihood."
note "likelihoods.py predicts these fight UNAIDS coverage through mu_ltfu:"
note "  mu_ltfu   0.05   0.11   0.16   0.25"
note "  p2       0.847  0.719  0.637  0.529   (target 0.828)"
note "  Ret_12m  0.951  0.903  0.865  0.802   (target 0.796)"
set_flags True

run_stage "$ARM_B/system2.npz"            s2_reten   "$PY" calibrate_system2.py 2019
stash "$ARM_B" system2.npz

run_stage "$ARM_B/posterior_analysis.npz" post_reten "$PY" run_posterior_analysis.py
stash "$ARM_B" posterior_analysis.npz

# --------------------------------------------------------------------------
say "Restoring arm A as the live base case"
# --------------------------------------------------------------------------
restore_lk
trap - EXIT INT TERM
for f in system2.npz posterior_analysis.npz prior_analysis.npz \
         scenario_point.npz scenario_uncertainty.npz scenario_prob.npz \
         scenario_prior_point.npz; do
  [ -f "$ARM_A/$f" ] && cp -f "$ARM_A/$f" "$RESULTS/$f"
done
note "data/results/ now holds the UNAIDS-anchored base case"

# --------------------------------------------------------------------------
if [ "$SKIP_FIGURES" -eq 0 ]; then
  say "Figures and tables (base case)"
  for s in fig_structure fig_inputs fig_results fig_arms tables; do
    note "-> writeup/$s.py"
    if ! "$PY" "writeup/$s.py" 2>&1 | tee "$LOGS/${s}_${STAMP}.log"; then
      printf '   \033[33m!! writeup/%s.py failed -- continuing\033[0m\n' "$s"
      FAILED_STAGES="${FAILED_STAGES}${s} "
    fi
  done
fi

# --------------------------------------------------------------------------
say "Comparison"
# --------------------------------------------------------------------------
"$PY" compare_arms.py 2>&1 | tee "$LOGS/compare_${STAMP}.log"

if [ -n "$FAILED_STAGES" ]; then
  printf '\n\033[33m== Stages that failed (non-fatal): %s\033[0m\n' "$FAILED_STAGES"
fi

say "Convergence"
"$PY" - <<'PYCHK'
import numpy as np, pathlib, sys
sys.path.insert(0, ".")
from calibrate_system2 import _block_convergence, convergence_report
for arm, label in (("arm_unaids", "UNAIDS-anchored"),
                   ("arm_retention", "Retention-anchored")):
    p = pathlib.Path("data/results") / arm / "system2.npz"
    if not p.exists():
        continue
    d = np.load(p, allow_pickle=True)
    names = [str(n) for n in d["param_names"]]
    ch, extra = d["chains"], d["extra"].item()
    n_outer = int(extra["n_outer_draws"]); per = ch.shape[0] // n_outer
    pieces = [ch[k*per:(k+1)*per] for k in range(n_outer)]
    nh = [extra["nh_draws_by_chain"][k*per:(k+1)*per] for k in range(n_outer)]
    conv = extra.get("convergence") or _block_convergence(pieces, names, nh)
    print(f"\n--- {label} ---")
    print(convergence_report(conv))
PYCHK

say "Done"
note "arm A (UNAIDS-anchored)     $ARM_A"
note "arm B (retention-anchored)  $ARM_B"
note "comparison table            writeup/arm_comparison.md"
note "logs                        $LOGS"
note "pre-run backup              $BACKUP"
