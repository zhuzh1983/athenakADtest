//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file field_diffusion.cpp
//! \brief Problem generator for ambipolar diffusion test problems.
//!
//! Three sub-problems controlled by <problem>/iprob:
//!
//! iprob=1: 1D sinusoidal ambipolar diffusion (convergence test)
//!   Domain: x in [0,1], periodic, 1D
//!   B1 = B0 (uniform guide field), B2 = amp*sin(2*pi*x), B3 = 0
//!   Analytic: B2(x,t) = amp*sin(2*pi*x)*exp(-eta_AD*(2*pi)^2*t)
//!   Run at N=32,64,128,256 to verify 2nd-order spatial convergence.
//!
//! iprob=2: 1D Gaussian ambipolar diffusion (Athena++ style test)
//!   B2(x,0) = amp*exp(-(x-x0)^2 / (2*sigma^2))
//!   Analytic: B2(x,t) = amp*(sigma/sigma(t))*exp(-(x-x0)^2/(2*sigma(t)^2))
//!   where sigma(t)^2 = sigma^2 + 2*eta_AD*t
//!
//! iprob=3: 2D field diffusion
//!   Domain: [0,1]x[0,1], periodic
//!   B1 = B0 (uniform), B3 = amp*sin(2*pi*x)*sin(2*pi*y)
//!   Tests 2D edge interpolation and J_perp calculation.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/ambipolar_diffusion.hpp"
#include "pgen/pgen.hpp"

// Prototype for error computation at end of run
void FieldDiffusionErrors(ParameterInput *pin, Mesh *pm);

namespace {
// anonymous namespace for problem-local data
struct FieldDiffVars {
  int iprob;
  Real B0, amp, sigma, x0, B3bg;
  Real eta_ambi;
};
FieldDiffVars fdv;
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for field diffusion tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Register final error-computation function
  pgen_final_func = FieldDiffusionErrors;
  if (restart) return;

  // Read problem parameters
  fdv.iprob = pin->GetOrAddInteger("problem", "iprob", 1);
  fdv.B0    = pin->GetOrAddReal("problem", "B0",   1.0);
  fdv.amp   = pin->GetOrAddReal("problem", "amp",  0.01);
  fdv.sigma = pin->GetOrAddReal("problem", "sigma",0.1);
  fdv.x0    = pin->GetOrAddReal("problem", "x0",   0.5);
  fdv.B3bg  = pin->GetOrAddReal("problem", "B3bg", 0.0);
  // B3bg: uniform background B3. When nonzero, J.B != 0, which
  // makes the ambipolar diffusion test distinct from Ohmic resistivity.
  // The effective diffusivity becomes eta_eff = eta * B0^2 / (B0^2 + B3bg^2).
  // Ohmic uses the full eta, so the two physics diverge when B3bg != 0.

  // Get eta_ambi from input file
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "field_diffusion pgen requires MHD physics." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->pambi == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "field_diffusion pgen requires ambipolar_diffusion in <mhd> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  fdv.eta_ambi = pmbp->pmhd->pambi->eta_ambi;

  // Capture index ranges
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Get EOS data
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto B0_   = fdv.B0;
  auto amp_  = fdv.amp;
  auto sigma_= fdv.sigma;
  auto x0_   = fdv.x0;
  auto B3bg_ = fdv.B3bg;
  int iprob_ = fdv.iprob;

  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  //---- iprob=1: 1D sinusoidal convergence test
  // B1 = B0 (guide field), B2 = amp*sin(2*pi*x), B3 = B3bg (uniform, default 0)
  //
  // When B3bg = 0: J.B = 0, so AD is identical to Ohmic (J_perp = J).
  //   Analytic: B2(x,t) = amp*sin(2*pi*x)*exp(-eta*k^2*t)
  //
  // When B3bg != 0: J.B = J3*B3bg != 0, so J_perp < J for AD but J_perp = J for Ohmic.
  //   Effective AD diffusivity: eta_eff = eta * B0^2 / (B0^2 + B3bg^2)
  //   Analytic: B2(x,t) = amp*sin(2*pi*x)*exp(-eta_eff*k^2*t)
  //   Ohmic would give exp(-eta*k^2*t) -- measurably different, distinguishing the two.
  if (iprob_ == 1) {
    par_for("fd_pgen1", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;

      Real b2_val = amp_*sin(2.0*M_PI*x1v);
      // Total energy: internal + magnetic (B1^2/2 + B2^2/2 + B3^2/2)
      u0(m,IEN,k,j,i) = 1.0/gm1 + 0.5*(SQR(B0_) + SQR(b2_val) + SQR(B3bg_));
    });
    // Face-centered B
    par_for("fd_bx1", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = B0_;  // uniform guide field along x
    });
    par_for("fd_bx2", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      b0.x2f(m,k,j,i) = amp_*sin(2.0*M_PI*x1v);
    });
    par_for("fd_bx3", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) = B3bg_;  // uniform background B3 (default 0)
    });
    return;
  }

  //---- iprob=2: 1D Gaussian diffusion test
  // B2(x,0) = amp * exp(-(x-x0)^2 / (2*sigma^2))
  if (iprob_ == 2) {
    par_for("fd_pgen2", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real b2_val = amp_*exp(-SQR(x1v - x0_)/(2.0*SQR(sigma_)));
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = 1.0/gm1 + 0.5*(SQR(B0_) + SQR(b2_val));
    });
    par_for("fd_bx1_2", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = B0_;
    });
    par_for("fd_bx2_2", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      b0.x2f(m,k,j,i) = amp_*exp(-SQR(x1v - x0_)/(2.0*SQR(sigma_)));
    });
    par_for("fd_bx3_2", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) = 0.0;
    });
    return;
  }

  //---- iprob=3: 2D field diffusion
  // B1 = B0 (guide), B3 = amp*sin(2*pi*x)*sin(2*pi*y)
  if (iprob_ == 3) {
    par_for("fd_pgen3", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real b3_val = amp_*sin(2.0*M_PI*x1v)*sin(2.0*M_PI*x2v);
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = 1.0/gm1 + 0.5*(SQR(B0_) + SQR(b3_val));
    });
    par_for("fd_bx1_3", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = B0_;
    });
    par_for("fd_bx2_3", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x2f(m,k,j,i) = 0.0;
    });
    par_for("fd_bx3_3", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
        ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      b0.x3f(m,k,j,i) = amp_*sin(2.0*M_PI*x1v)*sin(2.0*M_PI*x2v);
    });
    return;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Unrecognized iprob=" << iprob_ << " in field_diffusion pgen." << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn FieldDiffusionErrors()
//! \brief Computes and prints L1 error vs analytic solution at end of run.
//  For iprob=1 (sinusoidal): B2 analytic = amp*sin(2*pi*x)*exp(-eta*(2*pi)^2*t_final)
//  For iprob=2 (Gaussian):   B2 analytic = amp*(s0/s)*exp(-(x-x0)^2/(2*s^2))
//                             where s^2 = sigma^2 + 2*eta*t_final

void FieldDiffusionErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto &indcs = pm->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;

  Real t_final = pm->time;
  auto eta_a   = fdv.eta_ambi;
  auto B0_     = fdv.B0;
  auto amp_    = fdv.amp;
  auto sigma_  = fdv.sigma;
  auto x0_     = fdv.x0;
  auto B3bg_   = fdv.B3bg;
  int iprob_   = fdv.iprob;

  // For iprob=1 with B3bg != 0: the effective AD diffusivity is reduced because
  // part of J is parallel to B and therefore carries no ambipolar EMF.
  //   eta_eff = eta_AD * B0^2 / (B0^2 + B3bg^2)
  // Ohmic resistivity would use the full eta regardless of field direction.
  Real eta_eff = eta_a;
  if (iprob_ == 1 && B3bg_ != 0.0) {
    eta_eff = eta_a * SQR(B0_) / (SQR(B0_) + SQR(B3bg_));
  }

  // Only compute errors for iprob=1 and iprob=2
  if (iprob_ != 1 && iprob_ != 2) return;

  // Get bcc0 from MHD (cell-centered B after ConToPrim)
  auto &bcc0 = pmbp->pmhd->bcc0;

  Real l1_err = 0.0;
  int nx1_tot = indcs.nx1;

  Kokkos::parallel_reduce("fd_err",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pmbp->nmb_thispack * nx1_tot),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx / nx1_tot;
    int i = (idx % nx1_tot) + is;
    int j = js, k = ks;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real b2_analytic = 0.0;
    if (iprob_ == 1) {
      // Sinusoidal: B2 = amp*sin(2*pi*x)*exp(-eta_eff*(2*pi)^2*t)
      // For B3bg=0: eta_eff = eta (AD = Ohmic in this limit).
      // For B3bg!=0: eta_eff = eta*B0^2/(B0^2+B3bg^2) < eta (AD only).
      Real k2 = SQR(2.0*M_PI);
      b2_analytic = amp_*sin(2.0*M_PI*x1v)*exp(-eta_eff*k2*t_final);
    } else if (iprob_ == 2) {
      // Gaussian: B2 = amp*(s0/s)*exp(-(x-x0)^2/(2*s^2))
      Real sig2 = SQR(sigma_) + 2.0*eta_a*t_final;
      b2_analytic = amp_*(sigma_/sqrt(sig2))*exp(-SQR(x1v - x0_)/(2.0*sig2));
    }
    sum += fabs(bcc0(m,IBY,k,j,i) - b2_analytic);
  }, l1_err);

  // Normalize by total number of cells
  int nmb  = pmbp->nmb_thispack;
  l1_err  /= static_cast<Real>(nmb * nx1_tot);

  // Output to stdout
  if (global_variable::my_rank == 0) {
    std::cout << std::scientific;
    std::cout << "# [1] N_x1  [2] L1(B2)" << std::endl;
    std::cout << indcs.nx1 << "  " << l1_err << std::endl;
  }

  return;
}
