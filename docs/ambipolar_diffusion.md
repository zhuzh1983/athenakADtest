# Ambipolar Diffusion in AthenaK

## Table of Contents

1. [Physical Background](#1-physical-background)
2. [Definition of η_AD and Conventions](#2-definition-of-η_ad-and-conventions)
3. [Numerical Implementation](#3-numerical-implementation)
4. [How to Enable Ambipolar Diffusion](#4-how-to-enable-ambipolar-diffusion)
5. [Timestep Constraint](#5-timestep-constraint)
6. [Validation: Bai & Stone 2011 Tests](#6-validation-bai--stone-2011-tests)
7. [Known Limitations](#7-known-limitations)

---

## 1. Physical Background

Ambipolar diffusion (AD) is a non-ideal MHD effect arising in **partially ionized plasma**,
where neutral particles are not directly coupled to the magnetic field. The ions and electrons
carry the magnetic field, while the neutrals drift relative to them. The resulting
ion-neutral friction dissipates magnetic energy and allows field lines to slip through the
neutral gas.

AD is the dominant non-ideal MHD process in a wide range of astrophysical environments:

- **Molecular cloud cores and protostellar collapse** — AD controls the magnetic flux loss
  rate and the transition from magnetically supported to collapsing cores
- **Protoplanetary disk dead zones** — AD suppresses the magnetorotational instability (MRI)
  in the outer disk (r ≳ 10 AU) and affects disk wind launching
- **Low-ionization regions** — anywhere the ionization fraction χ_i ≲ 10⁻⁷

The effect appears in the induction equation as an extra electromotive force (EMF):

```
∂B/∂t = −∇ × (E_ideal + E_AD)
```

where the AD electric field is

```
E_AD = η_AD · J_⊥
```

and **J_⊥** is the component of current density perpendicular to **B**:

```
J_⊥ = J − (J·B / B²) · B
```

Unlike Ohmic resistivity (`E_Ohm = η_Ohm · J`, which damps all current), AD acts **only on
the perpendicular current**, leaving field-aligned currents unchanged. The physical
consequence is that AD damps waves that bend field lines (Alfvén and slow magnetosonic)
but does not damp fast magnetosonic waves (which compress without bending field lines).

The associated energy flux (Poynting-like heating term) is:

```
S_AD = E_AD × B = η_AD · (J × B)
```

Note that `J_∥ × B = 0`, so `S_AD = η_AD · J_⊥ × B = η_AD · (J × B)` — structurally
identical to the Ohmic Poynting flux.

---

## 2. Definition of η_AD and Conventions

### Physical definition

The ambipolar diffusion coefficient η_AD has units of **[length² / time]** (same as
magnetic diffusivity). It is related to the underlying plasma microphysics by:

```
η_AD = B² / (γ_AD · ρ_i · ρ_n)
```

where γ_AD is the ion-neutral coupling coefficient (momentum transfer rate per unit
ion-neutral density product), ρ_i is the ion mass density, and ρ_n is the neutral
mass density. For a weakly ionized medium:

```
η_AD  ≈  B² / (μ_in · ρ_n²)   [in appropriate units]
```

### Convention used in AthenaK (this implementation)

In the AthenaK AD module, the input parameter `ambipolar_diffusion` is **η_AD directly**:

```
E_AD = η_AD · J_⊥
```

The parabolic timestep constraint is:

```
Δt_AD = C · Δx² / η_AD
```

with C = 1/6 (3D), 1/4 (2D), 1/2 (1D).

### Convention used in Athena++ (DIFFERENT — beware)

Athena++ uses a **different parameterization** (see [GitHub issue #369](https://github.com/PrincetonUniversity/athena/issues/369)).
It stores `eta_AD_pp = η_AD / B²` (units of [time / length²]) and internally computes
`etaB = eta_AD_pp · B²`. The relation is:

```
η_AD (AthenaK)  =  etaB (Athena++)  =  eta_AD_pp (Athena++) · B²
```

For the same physical problem, the numerical values differ by a factor of B²:

| Code | Parameter | Value for BS11 test (B₀²=3.25, ω_a=100) |
|------|-----------|------------------------------------------|
| **AthenaK** | `ambipolar_diffusion` = η_AD | **0.0325** = B₀²/ω_a |
| **Athena++** | `ambipolar_diffusion` = η_AD/B² | **0.01** = 1/ω_a |

**Both give identical physics** for uniform-B problems; the difference is purely
in how the coefficient is parameterized.

### Connection to physical quantities (Bai & Stone 2011 notation)

Bai & Stone 2011 (ApJ 736, 144) define the "Alfvén wave damping frequency":

```
ω_A = |k_∥|² · B₀² / (ρ · η_AD)   →   η_AD = B₀² / ω_A
```

For their fiducial test: B₀² = 3.25, ω_A = 100 → **η_AD = 0.0325**.

The analytic Alfvén wave damping rate is:

```
Γ_A = (1/2) · η_AD · |k_∥|²  =  π² · η_AD   [for k_∥ = 2π, single-mode]
     ≈ 0.197   [for η_AD = 0.0325]
```

---

## 3. Numerical Implementation

### Algorithm: constrained transport (CT) with inline J interpolation

AthenaK uses a **constrained transport** (CT) scheme for the induction equation.
The AD electric field is added to the CT edge EMFs:

```
e1(k-½, j-½, i) += η_AD · [J1 − (J·B / B²) · B1]   at each edge
```

For **3D**, three flat Kokkos `par_for` kernels compute E1, E2, E3 separately.
Each kernel computes J and B at the exact edge position following
**Athena++ `FieldDiffusion::AmbipolarEMF`** (Stone & Gardiner 2005 style):

- **J (own component)**: direct finite-difference curl of B at the edge position
- **J (cross components)**: 4-point face averages of J evaluated at neighbouring positions
- **B (diagonal component)**: 8-point average of face-B values = 4-point average of
  cell-centred B, bringing it to the exact edge position

The `CurrentDensity()` helper from `src/diffusion/current_density.hpp` is used in the
1D branch only; 2D and 3D branches compute J inline to achieve the correct interpolation
stencils for cross-components.

### Energy update

When `eos = ideal`, the AD Poynting flux `S_AD = η_AD · (J × B)` is added to the
energy flux at each cell face:

```
F_energy,x1(i+½) += η_AD · (J₂·B₃ − J₃·B₂)|_{i+½}
```

This is implemented in `AddFluxConstantAmbi()` and is structurally identical to
`AddFluxConstantResist()` in the Ohmic resistivity module.

### Interaction with Ohmic resistivity

Both Ohmic and AD contributions are additive to the CT edge EMF array:

```cpp
// In EField() task (mhd_tasks.cpp):
if (presist != nullptr) presist->AddResistiveEMFs(b0, efld);   // E += η_Ohm · J
if (pambi   != nullptr) pambi->AddAmbipolarEMFs(b0, efld);     // E += η_AD · J_⊥
```

Each module computes J independently from the same face-B values — no shared mutable
state. They can be used simultaneously without interference.

---

## 4. How to Enable Ambipolar Diffusion

### Minimal input file

Add the `ambipolar_diffusion` key to the `<mhd>` block:

```ini
<mhd>
eos              = ideal
reconstruct      = plm
rsolver          = hlld
ambipolar_diffusion = 0.0325    # η_AD in units of [length² / time]
```

If the key is absent, `pambi = nullptr` and the run is identical to ideal MHD.

### Combined Ohmic + AD

```ini
<mhd>
eos                  = ideal
reconstruct          = plm
rsolver              = hlld
resistivity          = 0.001    # η_Ohm
ambipolar_diffusion  = 0.0325   # η_AD
```

Both are applied; the parabolic timestep uses the minimum of the two constraints.

### Isothermal EOS

```ini
<mhd>
eos                  = isothermal
iso_sound_speed      = 1.0
ambipolar_diffusion  = 0.0325
```

The energy flux `S_AD` is automatically disabled for isothermal EOS (the code checks
`eos_data.is_ideal` in `Fluxes()`).

### Complete example: 1D Alfvén wave damping

```ini
<comment>
1D Alfven wave with ambipolar diffusion (Bai & Stone 2011, iprob=1D)

<job>
basename  = alfven_1d

<problem>
pgen_name = linear_wave
wave_flag = 5          # rightgoing Alfven wave
amp       = 1.0e-4
dens      = 1.0
pgas      = 0.6
bx0       = 1.0
by0       = 1.41421356237309504
bz0       = 0.5
along_x1  = true

<mesh>
nghost = 2
nx1    = 32     x1min = 0.0   x1max = 1.0   ix1_bc = periodic   ox1_bc = periodic
nx2    = 1      x2min = 0.0   x2max = 1.0
nx3    = 1      x3min = 0.0   x3max = 1.0

<meshblock>
nx1 = 32

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.4
tlim       = 5.0

<mhd>
eos                 = ideal
gamma               = 1.6666666666666667
reconstruct         = plm
rsolver             = hlld
ambipolar_diffusion = 0.0325   # = B0²/omega_a = 3.25/100

<output1>
file_type   = tab
variable    = mhd_bcc
data_format = %14.8e
dt          = 0.1
```

---

## 5. Timestep Constraint

The AD operator is **parabolic** (diffusion-like), so it imposes a CFL-type stability
constraint much more restrictive than the hyperbolic ideal-MHD constraint at high
resolution:

```
Δt_AD ≤ C · Δx² / η_AD
```

where C is the dimensionality factor applied on top of the user's `cfl_number`:

| Dimension | C |
|-----------|---|
| 1D | 1/2 |
| 2D | 1/4 |
| 3D | 1/6 |

For a 3D uniform grid with Δx = L/N:

```
Δt_AD = cfl_number · (1/6) · (L/N)² / η_AD
```

This scales as N⁻², so doubling the resolution requires **4× more timesteps** (compared
to 2× for hyperbolic). For stiff problems (η_AD large or N large), AD can dominate
the runtime. Super-time-stepping (STS) is a future enhancement.

### Example: N=64³, η_AD = 0.0325, L = √3

```
Δx = √3/64 ≈ 0.027
Δt_AD = 0.4 × (1/6) × (0.027)² / 0.0325 ≈ 1.5 × 10⁻³
```

This is comparable to the ideal-MHD CFL for v_fast ≈ 2, which confirms the AD
timestep dominates at N=64.

---

## 6. Validation: Bai & Stone 2011 Tests

The `tutorial_bs11/` directory provides a self-contained reproduction of Fig. 2 of
Bai & Stone 2011. All tests use:

```
ρ₀ = 1,  P₀ = 0.6,  γ = 5/3  →  c_s = 1
B₀ = (1, √2, 0.5)  →  B₀² = 3.25,  v_A = √(3.25) ≈ 1.80
ω_A = 100  →  η_AD = B₀²/ω_A = 0.0325
k_∥ = 2π (one wavelength per domain in all geometries)
```

Analytic damping rates (from Bai & Stone 2011, Table 1):

| Wave | Γ_analytic |
|------|-----------|
| Alfvén | 2π²/100 ≈ 0.197 |
| fast | 5.2π²/100 ≈ 0.513 |
| slow | 1.3π²/100 ≈ 0.128 |

### Measured ratios Γ_fit / Γ_analytic (PLM, corrected code)

| Wave | 1D (N=32) | 2D (N=64×32) | 3D (N=32³) | 3D (N=64³) |
|------|-----------|--------------|------------|------------|
| Alfvén | 1.071 | 1.108 | 1.176 | **1.022** |
| fast | 1.050 | 1.039 | 1.065 | **1.010** |
| slow | 1.057 | 1.108 | 1.136 | **1.000** |

All results are [OK] (within 18% of analytic at N=32³, within 2.5% at N=64³).
The N=64³ results converge from above, consistent with 2nd-order PLM.

### Running the tutorial

```bash
# Build with built-in pgens
cmake -B build_linwave -DPROBLEM=built_in_pgens
cmake --build build_linwave -j4

# Run all 9 simulations and generate figure (from athenak root)
bash tutorial_bs11/run_bs11.sh

# Or plot only (if data already exists)
python3 tutorial_bs11/plot_bs11.py
```

Output: `tutorial_bs11/output/wave_damping_bs11.pdf`

### Amplitude extraction note

The plot script computes wave amplitude as
`A(t) = max_{x₁} √(δB₁² + δB₂² + δB₃²)` where `δBᵢ = Bᵢ - Bᵢ_background`.
For 3D oblique tests, **use the analytically computed background** (rotation of the
wave-frame B₀ into grid frame), not rounded values — a ~1e-4 error in the background
B introduces ~30% error in the fitted Γ. The `plot_bs11.py` script computes this
correctly using the exact rotation matrix.

For the 3D Alfvén wave, the amplitude oscillates with period T_beat ≈ 0.5 (beating
between right- and left-going modes excited by the initial condition). A log-linear fit
over all 51 snapshots averages out these oscillations correctly.

---

## 7. Known Limitations

### Constant η_AD only

The current implementation supports only a **spatially and temporally constant** η_AD
(read from the input file). Variable η_AD — needed for realistic ISM/disk simulations
where the ionization fraction varies — is not yet implemented.

To add variable η_AD:
1. Allocate `DvceArray4D<Real> eta_ad_cc` in the `AmbipolarDiffusion` class
2. Add a user function pointer `UserAmbipolarDiff` (same pattern as Athena++'s
   `UserAmbipolarDiff`)
3. Call it before `AddAmbipolarEMFs()` to populate `eta_ad_cc` per cell
4. In the EMF kernels, interpolate `eta_ad_cc` to each edge (4-point average)

### No super-time-stepping (STS)

The parabolic CFL constraint (Δt ∝ Δx²) makes high-resolution AD runs expensive.
STS (Alexiades et al. 1996, as implemented in Athena++) would relax this by taking
N_STS substeps, reducing the effective CFL cost from O(N²) to O(N^{3/2}).

### No Hall effect

The Hall EMF (`E_Hall = η_H · (J × B̂)`) is the next non-ideal MHD term. It follows
the same CT dispatch pattern and would be added as a `HallEffect` class alongside
`AmbipolarDiffusion`.

### Explicit timestepping only

This implementation uses standard explicit RK2 timestepping. Implicit or
operator-split methods are not currently supported.

---

## References

- Bai, X.-N. & Stone, J. M. 2011, ApJ, 736, 144 —
  *"Effect of Ambipolar Diffusion on the Nonlinear Evolution of Magnetorotational Instability"*
  (physics derivation and test problem setup)
- Stone, J. M. & Gardiner, T. A. 2005, NewA, 10, 229 —
  (constrained transport algorithm)
- Alexiades, V., Amiez, G., & Ladevant, P.-A. 1996, CNME, 12, 31 —
  (super-time-stepping for parabolic operators)
- [Athena++ issue #369](https://github.com/PrincetonUniversity/athena/issues/369) —
  (discussion of η_AD convention ambiguity)
