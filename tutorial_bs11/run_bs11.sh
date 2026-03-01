#!/bin/bash
# =============================================================================
# Bai & Stone 2011 (ApJ 736, 144) -- MHD Wave Damping Tests
# =============================================================================
# Runs 9 simulations (3 waves × 1D/2D/3D) and produces the comparison plot.
#
# Prerequisites
# -------------
#   1. Build the AthenaK binary with the linear_wave problem generator:
#        cmake -B build_linwave -DPROBLEM=built_in_pgens
#        cmake --build build_linwave -j4
#
#   2. Run this script from the ATHENAK ROOT directory:
#        bash tutorial_bs11/run_bs11.sh
#
# Physics
# -------
#   rho0=1, P0=0.6, gamma=5/3  -->  cs=1
#   B0=(1,sqrt(2),0.5),  B0^2=3.25,  omega_a=100,  eta_AD=0.0325
#   k=2*pi (lx=1 for all geometries)  -->  identical analytic damping rates:
#     Gamma_A = 2*pi^2/100  ~= 0.197
#     Gamma_f = 5.2*pi^2/100 ~= 0.513
#     Gamma_s = 1.3*pi^2/100 ~= 0.128
#
# Geometries
# ----------
#   1D: wave along x1,  N=32
#   2D: wave along grid diagonal (ang=arctan(2)~63 deg),
#       domain [0,sqrt(5)] x [0,sqrt(5)/2],  64x32 cells
#   3D: wave along cube diagonal (ang_3=45 deg, ang_2~35 deg),
#       domain [0,sqrt(3)]^3,  32^3 cells
# =============================================================================

set -euo pipefail

ATHENA="./build_linwave/src/athena"
INDIR="./tutorial_bs11/inputs"
OUTDIR="./tutorial_bs11/output"

# Check binary
if [[ ! -x "$ATHENA" ]]; then
  echo "ERROR: Binary not found at $ATHENA"
  echo "Build with: cmake -B build_linwave -DPROBLEM=built_in_pgens && cmake --build build_linwave -j4"
  exit 1
fi

echo "============================================================"
echo " Bai & Stone 2011 MHD Wave Damping Tests"
echo " Binary: $ATHENA"
echo "============================================================"

# Helper: run one simulation
run_sim() {
  local label="$1"   # e.g., "1D Alfven"
  local infile="$2"  # input file path
  local rundir="$3"  # output directory (athena writes tab files here)
  echo
  echo "── $label ──────────────────────────────────────"
  mkdir -p "$rundir/tab"
  (cd "$rundir" && "$OLDPWD/$ATHENA" -i "$OLDPWD/$infile" 2>&1 | tail -3)
  echo "   Done: $(ls $rundir/tab/*.tab 2>/dev/null | wc -l | tr -d ' ') snapshots"
}

# ── 1D simulations ─────────────────────────────────────────────────────────────
run_sim "1D  Alfven"  "$INDIR/athinput.bs11_1d_alfven"  "$OUTDIR/alfven_1d"
run_sim "1D  fast"    "$INDIR/athinput.bs11_1d_fast"    "$OUTDIR/fast_1d"
run_sim "1D  slow"    "$INDIR/athinput.bs11_1d_slow"    "$OUTDIR/slow_1d"

# ── 2D simulations ─────────────────────────────────────────────────────────────
run_sim "2D  Alfven"  "$INDIR/athinput.bs11_2d_alfven"  "$OUTDIR/alfven_2d"
run_sim "2D  fast"    "$INDIR/athinput.bs11_2d_fast"    "$OUTDIR/fast_2d"
run_sim "2D  slow"    "$INDIR/athinput.bs11_2d_slow"    "$OUTDIR/slow_2d"

# ── 3D simulations ─────────────────────────────────────────────────────────────
run_sim "3D  Alfven"  "$INDIR/athinput.bs11_3d_alfven"  "$OUTDIR/alfven_3d"
run_sim "3D  fast"    "$INDIR/athinput.bs11_3d_fast"    "$OUTDIR/fast_3d"
run_sim "3D  slow"    "$INDIR/athinput.bs11_3d_slow"    "$OUTDIR/slow_3d"

# ── Plot ──────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " All simulations complete. Generating plots..."
echo "============================================================"
MPLBACKEND=Agg python3 tutorial_bs11/plot_bs11.py

echo
echo "Output plots:"
echo "  $OUTDIR/wave_damping_bs11.pdf"
echo "  $OUTDIR/wave_damping_bs11.png"
