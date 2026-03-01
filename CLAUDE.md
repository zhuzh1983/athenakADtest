# AthenaK — Ambipolar Diffusion Module

## Overview

This repository (`hotfix/diffusion` branch) implements an ambipolar diffusion module
for the AthenaK MHD code. The module follows the naming conventions and dispatch
pattern of the existing `Resistivity` class.

**Physics reference:** Bai & Stone 2011 (ApJ 736, 144)

---

## Physics

```
E_AD = η_AD · J_⊥  =  η_AD · (J − (J·B / B²) · B)
```

Unlike Ohmic diffusion (`E = η·J`), ambipolar diffusion acts only on the
**perpendicular** current component, leaving field-aligned diffusion zero.

**CT induction update:** `∂B/∂t = −∇ × (E_ideal + E_Ohm + E_AD)`

**Energy flux:** `S_AD = E_AD × B = η_AD · (J × B)` (since `J_∥ × B = 0`)

**Timestep constraint:** `Δt_AD = C · Δx²_min / η_AD`
- C = 1/6 (3D), 1/4 (2D), 1/2 (1D)

---

## Files Added

### New Source Files

| File | Description |
|------|-------------|
| `src/diffusion/ambipolar_diffusion.hpp` | Class declaration (mirrors `resistivity.hpp`) |
| `src/diffusion/ambipolar_diffusion.cpp` | Physics implementation |
| `src/pgen/field_diffusion.cpp` | Test problem generator |
| `inputs/mhd/athinput.field_diffusion` | Input file for convergence test |

### Modified Files

| File | Change |
|------|--------|
| `src/mhd/mhd.hpp` | Added `AmbipolarDiffusion *pambi = nullptr;` pointer |
| `src/mhd/mhd.cpp` | Instantiate/delete `pambi`; `#include` ambipolar header |
| `src/mhd/mhd_tasks.cpp` | Call `pambi->AddAmbipolarEMFs()` in `EField()`; call `pambi->AddAmbipolarFluxes()` in `Fluxes()` |
| `src/mhd/mhd_newdt.cpp` | Call `pambi->NewTimeStep()` to refresh `dtnew` |
| `src/mesh/mesh.cpp` | Apply `pambi->dtnew` to global CFL timestep |
| `src/CMakeLists.txt` | Added `diffusion/ambipolar_diffusion.cpp` to source list |

---

## Class API

```cpp
class AmbipolarDiffusion {
 public:
  AmbipolarDiffusion(MeshBlockPack *pp, ParameterInput *pin);

  Real dtnew;          // timestep constraint (set in constructor for constant η)
  std::string ambi_type; // "constant"
  Real eta_ambi;       // coefficient read from <mhd>/ambipolar_diffusion

  // Dispatchers (called by MHD task DAG)
  void AddAmbipolarEMFs(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddAmbipolarFluxes(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);

  // Implementations
  void AddEMFConstantAmbi(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantAmbi(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
};
```

---

## Algorithm: AddEMFConstantAmbi

Uses `CurrentDensity()` (from `src/diffusion/current_density.hpp`) to get J at
each edge, then interpolates B by face-averaging, and computes J_⊥:

**3D — flat `par_for` kernels (one per EMF component):**

E1 at (k-½, j-½, i):
```
J1 = dB3/dx2 - dB2/dx3  (direct, correct position)
J2 = 0.25 * avg of J2 at 4 neighbouring (k-½, j, i-½) positions  (interpolated)
J3 = 0.25 * avg of J3 at 4 neighbouring (k, j-½, i-½) positions  (interpolated)
B1 = 0.125*(sum of 8 x1f values) = 4-pt avg of cell-centred B1  (diagonal)
B2 = 0.5*(x2f[k,j,i]  + x2f[k-1,j,i])
B3 = 0.5*(x3f[k,j,i]  + x3f[k,j-1,i])
Bsq   = B1² + B2² + B3² + 1e-300
JdotB = J1·B1 + J2·B2 + J3·B3
e1(m,k,j,i) += η_AD · (J1 − JdotB·B1/Bsq)
```

Same pattern for E2 (B2 diagonal = 8-pt avg) and E3 (B3 diagonal = 8-pt avg).
Matches Athena++ `FieldDiffusion::AmbipolarEMF` (3D branch) exactly.

**1D / 2D:** Team-based (1D) or flat par_for (2D) with reduced stencils, same J_⊥.

---

## Input File Usage

Activate ambipolar diffusion by adding to `<mhd>` block:
```ini
<mhd>
ambipolar_diffusion = 0.01    # η_AD coefficient
```

No `ambipolar_diffusion` key → `pambi = nullptr` → no diffusion (identical to ideal MHD).

---

## Test Problem: field_diffusion.cpp

Controlled by `<problem>/iprob`:

| iprob | Description | Analytic Solution |
|-------|-------------|-------------------|
| 1 | 1D sinusoidal: B2=amp·sin(2πx), B1=B0 | B2(t)=B2(0)·exp(−η·k²·t) |
| 2 | 1D Gaussian: B2=amp·exp(−(x−x0)²/2σ²) | σ(t)²=σ²+2·η·t |
| 3 | 2D: B3=amp·sin(2πx)·sin(2πy), B1=B0 | 2D extension of iprob=1 |

### Verified 2nd-Order Convergence (iprob=1, kinematic mode)

```
N=32:   L1(B2) = 7.79e-7
N=64:   L1(B2) = 1.94e-7  (ratio ≈ 4.0)
N=128:  L1(B2) = 4.85e-8  (ratio ≈ 4.0)
N=256:  L1(B2) = 1.21e-8  (ratio ≈ 4.0)
```

---

## Build Instructions

```bash
cd /Users/zhzhu/Research/ClaudeCode/athenak
git checkout hotfix/diffusion
git submodule update --init --recursive   # initialize Kokkos
cmake -B build -DPROBLEM=field_diffusion
cmake --build build -j4
```

**Note on macOS:** `git` may not be in PATH if Xcode license hasn't been accepted.
Use `/usr/bin/git` directly. Do NOT pass `-DKokkos_ENABLE_OPENMP=ON` on macOS without
OpenMP headers — serial build works without that flag.

### Run Convergence Test
```bash
for N in 32 64 128 256; do
  ./build/src/athena -i inputs/mhd/athinput.field_diffusion \
    mesh/nx1=$N meshblock/nx1=$N
done
```

---

## Implementation Notes

- **Branch:** All work targets `hotfix/diffusion`, not `main`
- **Naming:** Follows existing pattern: `AddAmbipolarEMFs` / `AddEMFConstantAmbi` /
  `AddFluxConstantAmbi` (mirrors `AddResistiveEMFs` / `AddEMFConstantResist` /
  `AddFluxConstantResist`)
- **Timestep architecture:** For constant η_AD, `dtnew` is computed once in the
  constructor. `mhd_newdt.cpp` calls `pambi->NewTimeStep()` to refresh the object's
  `dtnew` member; `mesh.cpp` then reads all per-object `dtnew` values with `std::min`
  to determine the global timestep. Do NOT accumulate `dtnew` inside `mhd_newdt.cpp`
  — that is the job of `mesh.cpp`.
- **Convergence test must use `kinematic` mode** (not `dynamic`) to avoid ideal-MHD
  magnetic pressure perturbations of order amp² that are resolution-independent and
  pollute the error measurement. Use `evolution = kinematic` and `rsolver = advect`
  in the input file.
- **Energy flux:** `AddFluxConstantAmbi` is structurally identical to
  `AddFluxConstantResist` (since J_∥×B=0 means S_AD = η_AD·(J×B)), with
  `eta_ohm` → `eta_ambi`.
- **No STS:** Explicit timestepping only. Super-time-stepping is a future enhancement.
- **TINY_NUMBER equivalent:** Uses `static_cast<Real>(1e-300)` added to Bsq to
  prevent division by zero in vacuum/low-B regions.
- **EMF location in task DAG:** `AddAmbipolarEMFs()` is called from `EField()` in
  `mhd_tasks.cpp` (not from `mhd_corner_e.cpp`). `AddAmbipolarFluxes()` is called
  from `Fluxes()` in `mhd_tasks.cpp`, guarded by `peos->eos_data.is_ideal`.
- **Kokkos submodule:** Must be initialized with
  `git submodule update --init --recursive` before the first cmake configure;
  an empty `kokkos/` directory causes a cryptic CMake error about missing
  `CMakeLists.txt`.

---

## Bug Fix History

### Bug 1: 1D E2 Missing Guide Field (B1) in bsq

In `AddEMFConstantAmbi` 1D branch, the E2 computation originally used
`bsq_e2 = B2² + B3²` (missing B1²). For tests with a guide field along x1
(like the Bai & Stone Alfvén wave: B0x=1), this caused `e2_val=0` exactly (the J·B
cancellation is exact for the Alfvén wave with only transverse J), so Bz received no
AD damping and the measured Γ_A was ~4× too slow.

**Fix:** Include `b1_e2 = b0.x1f(m,ks,js,i)` in `bsq_e2`. E3 already correctly
included B1.

### Bug 2: 3D J Interpolation and B Diagonal Position Errors

The original 3D `AddEMFConstantAmbi` used a `par_for_outer` (team-based) loop
over (m, k, j) pencils and called `CurrentDensity(m, k, j, ...)` to get J. This
caused two classes of error:

**J interpolation bug:** For E1 at (k-½, j-½, i), `CurrentDensity` places J2 at
position (k-½, j, i-½) and J3 at (k, j-½, i-½). For the J·B dot product at E1,
J2 and J3 must be brought to (k-½, j-½, i) by 4-point face averages (as in Athena++
`AmbipolarEMF`). The old team-based code used the single-pencil J2/J3 directly, at
the wrong positions.

**B diagonal bug:** B1 for E1 was averaged to position (k-½, j-½, i-½) via a
4-point x1f average, instead of (k-½, j-½, i) which requires an 8-point average of
x1f values (equivalent to a 4-point average of cell-centred B1). Same error for
B2@E2 and B3@E3.

**Effect:** These errors caused a numerical instability in the 3D N=64 run (amplitude
grew from ~1.2e-4 to ~0.4 by t≈0.8). The N=32 run was stable but would also have had
incorrect physics.

**Fix:** Replaced the team-based loop with three separate flat `par_for` kernels
(one per EMF component: ambi3_e1, ambi3_e2, ambi3_e3). Each kernel:
- Computes J inline with 4-point averages for cross-terms (matching Athena++ exactly)
- Uses 8-point x1f averages (= 4-pt cell-centred B1 avg) for the diagonal B component

**Result:** N=64 3D Alfvén wave test runs stably to t=5.0 (3328 cycles) and gives
Γ_fit = 0.202 (1.022× theory), matching the N=32 result.

**Also fixed:** `NewTimeStep()` was a stub (just `return`). Now recomputes
`dtnew = fac*dx²/eta_ambi` each call, like `Resistivity::NewTimeStep()`.

---

## Bai & Stone 2011 Tutorial Tests

Reproduces Fig. 2 of Bai & Stone 2011 (1D/2D/3D) using `linear_wave` pgen + AD module.
All tests use the **same domain setup**: `lx = 1`, `k_par = 2π`, so analytic Γ values are
identical across geometries (no rescaling needed).

### Directory layout (all self-contained under `tutorial_bs11/`)

```
tutorial_bs11/
  inputs/        # 9 input files: athinput.bs11_{1d,2d,3d}_{alfven,fast,slow}
  output/        # 9 run directories: {alfven,fast,slow}_{1d,2d,3d}/tab/*.tab
  run_bs11.sh    # runs all 9 simulations then plots
  plot_bs11.py   # 3-panel figure; log-linear fit for Γ
```

Run from athenak root: `bash tutorial_bs11/run_bs11.sh`
Plot only: `python3 tutorial_bs11/plot_bs11.py`

### Domain geometry and background B

| Geom | Domain         | Grid  | ang_3    | ang_2    | lx |
|------|----------------|-------|----------|----------|-----|
| 1D   | [0,1]          | 32    | 0°       | 0°       | 1.0 |
| 2D   | [0,√5]×[0,√5/2]| 64×32 | atan(2)≈63° | 0°  | 1.0 |
| 3D   | [0,√3]³        | 32³   | atan(1)=45° | atan(1/√2)≈35° | 1.0 |

Background B in grid frame (used for amplitude extraction):

| Geom | B1_grid    | B2_grid    | B3_grid |
|------|-----------|-----------|---------|
| 1D   | 1.0       | √2        | 0.5     |
| 2D   | (1−2√2)/√5 ≈ −0.818 | (2+√2)/√5 ≈ 1.527 | 0.5 |
| 3D   | ≈ −0.627  | ≈ 1.373   | ≈ 0.986 |

### Results summary (log-linear fit Γ over all 51 snapshots)

All runs below use the **corrected code** (flat par_for 3D kernels with proper J/B interpolation).

| Wave   | Γ_theory | 1D ratio | 2D ratio | 3D N=32 ratio | 3D N=64 ratio |
|--------|----------|----------|----------|---------------|---------------|
| Alfvén | 0.1974   | 1.071    | 1.108    | 1.176         | 1.022         |
| fast   | 0.5132   | 1.050    | 1.039    | 1.065         | 1.010         |
| slow   | 0.1283   | 1.057    | 1.108    | 1.136         | 1.000         |

N=64 systematically improves over N=32. Error ratio N=32/N=64 ≈ 6–8× (consistent with
2nd-order spatial + parabolic CFL temporal convergence). The slow wave at N=64 gives
ratio = 1.000 exactly.

**Note on old N=32 3D results:** the old buggy code (team-based J interpolation) gave
N=32 3D Alfvén = 1.024, fast = 1.061, slow = 1.095. These were artificially close to
theory because the incorrect J·B subtraction altered the beat phase evolution, causing
an upward amplitude trend at t≈5 that made the log-linear slope shallower. The corrected
code shows the true numerical error: ~18% for Alfvén, ~14% for slow at N=32.

**Amplitude extraction — use exact analytical background B:**
The `plot_bs11.py` script uses analytically computed BG_3D via rotation matrix
(COS_A2, SIN_A2, COS_A3, SIN_A3 from domain geometry). Do NOT use rounded values
(−0.6268, 1.3732, 0.9856) for the 3D background — this introduces ~20% error in
amplitude and ~30% error in Γ_fit. The exact values are:
```python
COS_A3 = SIN_A3 = 1/sqrt(2);  COS_A2 = sqrt(2/3);  SIN_A2 = 1/sqrt(3)
B1_bg = BX0*COS_A2*COS_A3 - BY0*SIN_A3 - BZ0*SIN_A2*COS_A3  # ≈ −0.62677
B2_bg = BX0*COS_A2*SIN_A3 + BY0*COS_A3 - BZ0*SIN_A2*SIN_A3  # ≈  1.37323
B3_bg = BX0*SIN_A2         + BZ0*COS_A2                       # ≈  0.98560
```
Alternatively, use the mean B from the t=0 snapshot (gives identical result).

**Note on 3D Alfvén amplitude oscillations:** the measured amplitude oscillates at
period T_beat ≈ 0.5 (beating between right/left-going modes excited in dynamic mode).
A log-linear fit over all snapshots gives Γ = 0.202 ≈ Γ_theory (1.024×), whereas a
naive A(0)→A(5) two-point estimate gives ~0.12–0.14 due to beat phase at t=5.
Always use the log-linear fit for rate extraction.

---

## Potential Future Extensions

- **Variable η_AD:** Add `DvceArray4D<Real> eta_ad_cc` and a user function pointer
  (same pattern as Athena++'s `UserAmbipolarDiff`) for field/density-dependent η
- **Super-time-stepping (STS):** Explicit STS (Alexiades et al. 1996) would relax
  the parabolic CFL constraint substantially for stiff diffusion problems
- **Hall effect:** Next non-ideal MHD term; follows identical CT dispatch pattern
- **Ion-neutral friction:** Directly related to ambipolar diffusion in the
  two-fluid limit; the `ion-neutral/` directory already exists in AthenaK
