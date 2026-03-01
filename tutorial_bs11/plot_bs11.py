"""
Bai & Stone 2011 (ApJ 736, 144) -- Fig. 2: Damping of MHD Waves by Ambipolar Diffusion
Reproduces both panels and extends to 3D.

Three-panel figure:
  Left   : 1D  (wave along x1, N=32)
  Center : 2D  (wave along grid diagonal, ang=arctan(2)~63 deg, 64x32 cells)
  Right  : 3D  (wave along cube diagonal, ang_3=45 deg / ang_2~35 deg, 32^3 cells)

All tests use the same physics and k_par = 2*pi  so analytic damping rates are identical.

Physics
-------
  rho0=1, P0=0.6, gamma=5/3  -->  cs=1
  B0 = (1, sqrt(2), 0.5),  B0^2=3.25,  vA=sqrt(3.25),  vAx=1.0,  omega_a=100
  eta_AD = B0^2/omega_a = 0.0325

Analytic damping rates (Bai & Stone 2011, eqs. 16-17):
  Gamma_A = 2   * pi^2/100 ~= 0.197
  Gamma_f = 5.2 * pi^2/100 ~= 0.513
  Gamma_s = 1.3 * pi^2/100 ~= 0.128

Amplitude extraction
--------------------
  All simulations output mhd_bcc (cell-centered B) along a 1D x1-slice
  (slice_x2=0, slice_x3=0).  The perturbation amplitude is
      A(t) = max_x1  sqrt( dB1^2 + dB2^2 + dB3^2 )
  where dBi = Bi_sim - Bi_bg and Bi_bg is the background in grid coordinates.
  For 1D (along_x1=true): Bi_bg = (bx0, by0, bz0).
  For 2D / 3D (oblique):  Bi_bg is the rotated background (computed below).

  The decay rate Gamma is extracted via log-linear least-squares over all snapshots.
  This correctly handles oscillations from counter-propagating mode interference.

Usage
-----
  cd /Users/zhzhu/Research/ClaudeCode/athenak
  python3 tutorial_bs11/plot_bs11.py
"""

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")

WAVES = {
    "alfven": dict(color="black", label="Alfvén"),
    "fast":   dict(color="blue",  label="fast"),
    "slow":   dict(color="red",   label="slow"),
}

# ── Physics constants ──────────────────────────────────────────────────────────
BX0, BY0, BZ0 = 1.0, np.sqrt(2.0), 0.5
B0SQ    = BX0**2 + BY0**2 + BZ0**2   # = 3.25
OMEGA_A = 100.0
ETA_AD  = B0SQ / OMEGA_A             # = 0.0325

K    = 2.0 * np.pi     # wavenumber (lx=1 for all tests)
VA2  = B0SQ            # rho0=1
CS2  = 1.0             # (5/3)*P0/rho0 = (5/3)*0.6 = 1.0
VAX2 = BX0**2          # = 1.0

VF2 = 0.5*((VA2+CS2) + np.sqrt((VA2-CS2)**2 + 4*CS2*(VA2-VAX2)))
VS2 = 0.5*((VA2+CS2) - np.sqrt((VA2-CS2)**2 + 4*CS2*(VA2-VAX2)))
VF  = np.sqrt(VF2)     # = 2.0
VS  = np.sqrt(VS2)     # = 0.5

cos2_theta = VAX2 / VA2

GAMMA_A = 0.5 * K**2 * VA2 * cos2_theta / OMEGA_A   # = 2*pi^2/100
GAMMA_F = 0.5 * (VF2 - CS2) / (VF2 - VS2) * K**2 * VA2 / OMEGA_A   # = 5.2*pi^2/100
GAMMA_S = 0.5 * (CS2 - VS2) / (VF2 - VS2) * K**2 * VA2 / OMEGA_A   # = 1.3*pi^2/100
GAMMAS = {"alfven": GAMMA_A, "fast": GAMMA_F, "slow": GAMMA_S}

print(f"Wave speeds:  vA={np.sqrt(VA2):.4f},  vf={VF:.4f},  vs={VS:.4f},  cs={np.sqrt(CS2):.4f}")
print(f"Analytic damping rates (k=2*pi, B0^2={B0SQ}, omega_a={OMEGA_A}):")
print(f"  Gamma_A = 2*pi^2/100  = {GAMMA_A:.6f}  ({2*np.pi**2/100:.6f})")
print(f"  Gamma_f = 5.2*pi^2/100 = {GAMMA_F:.6f}  ({5.2*np.pi**2/100:.6f})")
print(f"  Gamma_s = 1.3*pi^2/100 = {GAMMA_S:.6f}  ({1.3*np.pi**2/100:.6f})")
print()

# ── Background B in grid frame for each geometry ───────────────────────────────
# 1D (along_x1=true): wave frame = grid frame
BG_1D = np.array([BX0, BY0, BZ0])

# 2D: ang_3 = atan(x1size/x2size) = atan(sqrt(5)/(sqrt(5)/2)) = atan(2)
#     cos_a2=1 (2D), sin_a2=0
S5 = np.sqrt(5.0)
COS_A3_2D = 1.0/S5;   SIN_A3_2D = 2.0/S5
# B_grid = R * B_wave  where R uses only the 2D rotation (a2=0)
BG_2D = np.array([
    BX0*COS_A3_2D - BY0*SIN_A3_2D,           #  (1-2sqrt2)/sqrt5 ~ -0.8177
    BX0*SIN_A3_2D + BY0*COS_A3_2D,           #  (2+sqrt2)/sqrt5  ~  1.5269
    BZ0,                                       #  0.5
])

# 3D: ang_3=atan(1)=45 deg, ang_2=atan(1/sqrt(2))~35.26 deg
#     domain [0,sqrt(3)]^3 -> lx=1
COS_A3_3D = SIN_A3_3D = 1.0/np.sqrt(2.0)
COS_A2_3D = np.sqrt(2.0/3.0);   SIN_A2_3D = 1.0/np.sqrt(3.0)
BG_3D = np.array([
    BX0*COS_A2_3D*COS_A3_3D - BY0*SIN_A3_3D - BZ0*SIN_A2_3D*COS_A3_3D,  # ~-0.6268
    BX0*COS_A2_3D*SIN_A3_3D + BY0*COS_A3_3D - BZ0*SIN_A2_3D*SIN_A3_3D,  # ~ 1.3732
    BX0*SIN_A2_3D                            + BZ0*COS_A2_3D,             # ~ 0.9856
])
print(f"Grid-frame backgrounds:")
print(f"  1D: {BG_1D}")
print(f"  2D: {BG_2D.round(5)}")
print(f"  3D: {BG_3D.round(5)}")
print()

# Map geometry label to (output subdir, background array, tab col indices)
GEOM_CFG = {
    "1d":     dict(suffix="_1d",     bg=BG_1D, label="1D  (along $x_1$, $N=32$)"),
    "2d":     dict(suffix="_2d",     bg=BG_2D,
                   label=r"2D oblique ($\theta\!=\!\arctan2\!\approx\!63°$, $N_1\!=\!64, N_2\!=\!32$)"),
    "3d":     dict(suffix="_3d",     bg=BG_3D,
                   label=r"3D oblique ($\theta_{12}\!=\!45°,\,\theta_{23}\!\approx\!35°$, $N\!=\!32^3$)"),
    "3d_n64": dict(suffix="_3d_n64", bg=BG_3D,
                   label=r"3D oblique ($N\!=\!64^3$)"),
}


# ── Amplitude loader ───────────────────────────────────────────────────────────
def load_amplitude(wave, geom):
    """Return (t_array, amplitude_array) for given wave and geometry.

    Amplitude = max over x1-slice of |delta_B| = sqrt(dB1^2+dB2^2+dB3^2).
    Tab file columns: gid(0), i(1), x1v(2), bcc1(3), bcc2(4), bcc3(5).
    """
    cfg    = GEOM_CFG[geom]
    suffix = cfg["suffix"]
    bg     = cfg["bg"]

    tab_dir = os.path.join(OUTDIR, f"{wave}{suffix}", "tab")
    files   = sorted(glob.glob(os.path.join(tab_dir, "*.tab")))
    if not files:
        raise FileNotFoundError(f"No .tab files in {tab_dir}")

    t_arr, A_arr = [], []
    for fpath in files:
        with open(fpath) as fh:
            t = float(fh.readline().split("time=")[1].split()[0])
            fh.readline()          # skip column header
            rows = []
            for line in fh:
                cols = line.split()
                if len(cols) >= 6:
                    rows.append([float(cols[3]), float(cols[4]), float(cols[5])])
        if rows:
            arr = np.array(rows)
            dB  = arr - bg
            amp = np.max(np.sqrt((dB**2).sum(axis=1)))
            t_arr.append(t)
            A_arr.append(amp)

    return np.array(t_arr), np.array(A_arr)


def fit_decay_rate(t, A):
    """Log-linear least-squares fit of A(t) = A0 * exp(-Gamma * t).

    Returns (Gamma, A0).  Uses all points with A>0.
    The log-linear fit is robust to oscillations from counter-propagating
    mode beating (e.g., 3D Alfven wave shows ~0.5 period beating).
    """
    mask = A > 0
    if mask.sum() < 2:
        return np.nan, np.nan
    log_A   = np.log(A[mask])
    t_fit   = t[mask]
    coeffs  = np.polyfit(t_fit, log_A, 1)
    Gamma   = -coeffs[0]
    A0      = np.exp(coeffs[1])
    return Gamma, A0


# ── Load all data ──────────────────────────────────────────────────────────────
data = {}   # data[(wave, geom)] = (t, A)

print("Loading simulation data...")
for geom in ("1d", "2d", "3d", "3d_n64"):
    for wave in WAVES:
        try:
            t, A = load_amplitude(wave, geom)
            data[(wave, geom)] = (t, A)
            Gam, _ = fit_decay_rate(t, A)
            Gam_th = GAMMAS[wave]
            flag = "OK" if abs(Gam/Gam_th - 1) < 0.25 else "CHECK"
            print(f"  {geom:8s} {wave:6s}: {len(t):2d} snapshots  "
                  f"Gamma_fit={Gam:.4f}  theory={Gam_th:.4f}  ratio={Gam/Gam_th:.3f}  [{flag}]")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
print()


# ── Summary table ──────────────────────────────────────────────────────────────
print(f"{'Wave':8s} {'Geom':8s} {'Gamma_theory':>12s} {'Gamma_fit':>10s} {'ratio':>6s}")
print("-" * 50)
for geom in ("1d", "2d", "3d", "3d_n64"):
    for wave in WAVES:
        if (wave, geom) in data:
            t, A   = data[(wave, geom)]
            Gam, _ = fit_decay_rate(t, A)
            Gam_th = GAMMAS[wave]
            print(f"{wave:8s} {geom:8s} {Gam_th:12.4f} {Gam:10.4f} {Gam/Gam_th:6.3f}")
print()


# ── Plot ───────────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
t_fine = np.linspace(0, 5, 1000)

# Common y-axis upper limit from N=32 runs (ignore N=64 which has same A0)
base_geoms = ("1d", "2d", "3d")
all_A0 = [data[(w, g)][1][0] for (w, g) in data
          if g in base_geoms and len(data[(w, g)][1]) > 0]
y_max  = max(all_A0) * 1.10 if all_A0 else 5e-5

GEOM_PANEL_LABELS = {"1d": "1D", "2d": "2D", "3d": "3D"}

for ax_idx, (ax, geom) in enumerate(zip(axes, ("1d", "2d", "3d"))):
    analytic_plotted = set()

    for wave, props in WAVES.items():
        col    = props["color"]
        Gam_th = GAMMAS[wave]

        # ── N=32 curve ─────────────────────────────────────────────────────
        if (wave, geom) in data:
            t, A    = data[(wave, geom)]
            A0_plot = A[0]
            ax.plot(t, A, "-", color=col, lw=1.0, alpha=0.45)

            # Analytic curve (dashed), drawn once per wave
            if wave not in analytic_plotted:
                ax.plot(t_fine, A0_plot * np.exp(-Gam_th * t_fine),
                        "--", color=col, lw=1.5)
                analytic_plotted.add(wave)

        # ── N=64 overlay (3D panel only) ────────────────────────────────────
        if geom == "3d" and (wave, "3d_n64") in data:
            t64, A64 = data[(wave, "3d_n64")]
            ax.plot(t64, A64, "-", color=col, lw=2.0)

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("wave amplitude", fontsize=12)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, y_max)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(10)

    # Panel label inside plot (upper right, bold)
    ax.text(0.95, 0.95, GEOM_PANEL_LABELS[geom],
            transform=ax.transAxes, ha="right", va="top",
            fontsize=16, fontweight="bold")

    # Wave color legend in left panel
    if ax_idx == 0:
        handles = [Line2D([0], [0], color=WAVES[w]["color"], lw=1.5,
                          label=WAVES[w]["label"])
                   for w in ("fast", "alfven", "slow")]
        ax.legend(handles=handles, fontsize=10, loc="lower left", frameon=False)

    # Resolution legend in 3D panel
    if geom == "3d":
        has_n64 = any((w, "3d_n64") in data for w in WAVES)
        res_handles = [Line2D([0], [0], color="gray", lw=1.0, alpha=0.45, label=r"$N=32^3$"),
                       Line2D([0], [0], color="gray", lw=2.0,              label=r"$N=64^3$"),
                       Line2D([0], [0], color="gray", lw=1.5, ls="--",    label="analytic")]
        if has_n64:
            ax.legend(handles=res_handles, fontsize=9, loc="lower left", frameon=False)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTDIR, exist_ok=True)
for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"wave_damping_bs11.{ext}")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

plt.show()
