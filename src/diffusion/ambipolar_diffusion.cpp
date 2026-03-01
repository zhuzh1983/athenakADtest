//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ambipolar_diffusion.cpp
//  \brief Implements functions for AmbipolarDiffusion class.
//  Physics follows Bai & Stone 2011 (ApJ 736, 144).
//
//  The ambipolar diffusion EMF:
//    E_AD = eta_AD * J_perp  where  J_perp = J - (J.B / B^2) * B
//
//  Unlike Ohmic diffusion (E = eta*J), ambipolar diffusion acts only on the
//  perpendicular component of J, leaving field-aligned diffusion zero.
//  The energy flux is S_AD = E_AD x B = eta_AD * (J x B), identical in form
//  to the Ohmic Poynting flux with eta_ohm -> eta_ambi.
//
//  Convention: eta_ambi (read from input) is the physical ambipolar diffusion
//  coefficient eta_AD with units [length^2/time].  For Bai & Stone 2011 tests,
//  eta_AD = B0^2 / omega_A.  (Athena++ uses a different parameterisation where
//  eta_ad_pp = eta_AD / B^2, so the same test needs eta_ad_pp = 1/omega_A.)
//
//  J interpolation: for E1 at (k-1/2, j-1/2, i), CurrentDensity places J1 at
//  the correct position, but J2 (at k-1/2, j, i-1/2) and J3 (at k, j-1/2, i-1/2)
//  must be brought to (k-1/2, j-1/2, i) by 4-point face-averages, exactly as in
//  Athena++ AmbipolarEMF.  The "diagonal" B component (B1 for E1, B2 for E2,
//  B3 for E3) is likewise brought to the correct edge position via 4-point averages
//  of cell-centred values (= 8-point averages of face values).

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "ambipolar_diffusion.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// Constructor

AmbipolarDiffusion::AmbipolarDiffusion(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read constant ambipolar diffusion coefficient
  eta_ambi = pin->GetReal("mhd", "ambipolar_diffusion");
  ambi_type = "constant";

  // Precompute timestep constraint: dt = C * dx^2 / eta_ambi
  // with C = 1/6 (3D), 1/4 (2D), 1/2 (1D)
  dtnew = std::numeric_limits<float>::max();
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  auto &size = pmy_pack->pmb->mb_size;
  for (int m = 0; m < (pmy_pack->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ambi);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/eta_ambi);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/eta_ambi);
    }
  }
}

//----------------------------------------------------------------------------------------
// Destructor

AmbipolarDiffusion::~AmbipolarDiffusion() {
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddAmbipolarEMFs()
//! \brief Dispatcher that adds ambipolar diffusion electric fields.

void AmbipolarDiffusion::AddAmbipolarEMFs(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  AddEMFConstantAmbi(b0, efld);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddAmbipolarFluxes()
//! \brief Dispatcher that adds ambipolar energy (Poynting) fluxes.

void AmbipolarDiffusion::AddAmbipolarFluxes(const DvceFaceFld4D<Real> &b0,
    DvceFaceFld5D<Real> &flx) {
  AddFluxConstantAmbi(b0, flx);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddEMFConstantAmbi()
//  \brief Adds electric field from ambipolar diffusion to edge-centred EMFs.
//
//  E_AD = eta_AD * J_perp  where  J_perp = J - (J.B / B^2) * B
//
//  Algorithm (follows Athena++ FieldDiffusion::AmbipolarEMF exactly):
//  For each edge:
//    - The "own" J component (J1 for E1, J2 for E2, J3 for E3) is computed
//      directly from the local curl of B at that edge position.
//    - The two "cross" J components are brought to the edge by 4-point
//      face-averages of their natural positions.
//    - The "diagonal" B component (B1 for E1, B2 for E2, B3 for E3) is brought
//      to the edge via 4-point averages of cell-centred values (= 8-point
//      averages of face B values).  The other two B components already land on
//      the edge via 2-point face-averages.

void AmbipolarDiffusion::AddEMFConstantAmbi(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta_a = eta_ambi;

  //---- 1-D problem:
  // In 1D only E2 and E3 are active.  B1 is the guide field; B3=0 for
  // the standard convergence test.  J1=0, so J.B = 0 for the E3 edge and
  // E3_AD = eta_AD * J3 exactly.  We still carry the full J_perp expression
  // for generality (it reduces correctly in all 1D limits).

  if (pmy_pack->pmesh->one_d) {
    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ambi1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      par_for_inner(member, is, ie+1, [&](const int i) {
        // E2 at (ks, js, i): position (ks, js, i-1/2)
        Real b1_e2 = b0.x1f(m,ks,js,i);
        Real b2_e2 = b0.x2f(m,ks,js,i);
        Real b3_e2 = 0.5*(b0.x3f(m,ks,js,i) + b0.x3f(m,ks+1,js,i));
        Real bsq_e2 = b1_e2*b1_e2 + b2_e2*b2_e2 + b3_e2*b3_e2
                      + static_cast<Real>(1e-300);
        Real jdb_e2 = j2(i)*b2_e2 + j3(i)*b3_e2;  // j1=0 in 1D
        Real e2_val = eta_a*(j2(i) - jdb_e2*b2_e2/bsq_e2);

        // E3 at (ks, js, i): position (ks, js, i-1/2)
        Real b1_fc = b0.x1f(m,ks,js,i);
        Real b2_e3 = 0.5*(b0.x2f(m,ks,js,i) + b0.x2f(m,ks,js+1,i));
        Real b3_e3 = b0.x3f(m,ks,js,i);
        Real bsq_e3 = b1_fc*b1_fc + b2_e3*b2_e3 + b3_e3*b3_e3
                      + static_cast<Real>(1e-300);
        Real jdb_e3 = j1(i)*b1_fc + j2(i)*b2_e3 + j3(i)*b3_e3;
        Real e3_val = eta_a*(j3(i) - jdb_e3*b3_e3/bsq_e3);

        e2(m,ks,  js  ,i) += e2_val;
        e2(m,ke+1,js  ,i) += e2_val;
        e3(m,ks  ,js  ,i) += e3_val;
        e3(m,ks  ,je+1,i) += e3_val;
      });
    });
    return;
  }

  //---- 2-D problem:
  // Flat par_for kernels, one per EMF component.
  // J cross-terms and B diagonal use the same interpolation stencils as
  // Athena++ FieldDiffusion::AmbipolarEMF (2D branch).

  if (pmy_pack->pmesh->two_d) {
    // -- E1 at (ks, j-1/2, i) --
    par_for("ambi2_e1", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      Real dx1 = mbsize.d_view(m).dx1;
      Real dx2 = mbsize.d_view(m).dx2;

      // J1 direct: dB3/dx2  (no dB2/dx3 in 2D)
      Real j1 = (b0.x3f(m,ks,j,i) - b0.x3f(m,ks,j-1,i))/dx2;

      // J2: 4-pt avg of J2 at (ks,j,i),(ks,j,i+1),(ks,j-1,i),(ks,j-1,i+1)
      // J2(ks,j,i) = -dB3/dx1  (no dB1/dx3 in 2D)
      Real j2 = 0.25*(
        -(b0.x3f(m,ks,j,  i  ) - b0.x3f(m,ks,j,  i-1))/dx1
        -(b0.x3f(m,ks,j,  i+1) - b0.x3f(m,ks,j,  i  ))/dx1
        -(b0.x3f(m,ks,j-1,i  ) - b0.x3f(m,ks,j-1,i-1))/dx1
        -(b0.x3f(m,ks,j-1,i+1) - b0.x3f(m,ks,j-1,i  ))/dx1);

      // J3: 2-pt avg of J3 at (ks,j,i),(ks,j,i+1)
      // J3(ks,j,i) = dB2/dx1 - dB1/dx2
      Real j3 = 0.5*(
        (b0.x2f(m,ks,j,i  ) - b0.x2f(m,ks,j,i-1))/dx1
       -(b0.x1f(m,ks,j,i  ) - b0.x1f(m,ks,j-1,i  ))/dx2
       +(b0.x2f(m,ks,j,i+1) - b0.x2f(m,ks,j,i  ))/dx1
       -(b0.x1f(m,ks,j,i+1) - b0.x1f(m,ks,j-1,i+1))/dx2);

      // B: B1 = 4-pt avg of cell-centred B1, B2 exact, B3 2-pt avg
      Real b1 = 0.25*(b0.x1f(m,ks,j,  i) + b0.x1f(m,ks,j,  i+1)
                    + b0.x1f(m,ks,j-1,i) + b0.x1f(m,ks,j-1,i+1));
      Real b2 = b0.x2f(m,ks,j,i);
      Real b3 = 0.5*(b0.x3f(m,ks,j,i) + b0.x3f(m,ks,j-1,i));

      Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
      Real jdb = j1*b1 + j2*b2 + j3*b3;
      Real e1_val = eta_a*(j1 - jdb*b1/bsq);
      e1(m,ks,  j,i) += e1_val;
      e1(m,ke+1,j,i) += e1_val;
    });

    // -- E2 at (ks, j, i-1/2) --
    par_for("ambi2_e2", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      Real dx1 = mbsize.d_view(m).dx1;
      Real dx2 = mbsize.d_view(m).dx2;

      // J2 direct: -dB3/dx1  (no dB1/dx3 in 2D)
      Real j2 = -(b0.x3f(m,ks,j,i) - b0.x3f(m,ks,j,i-1))/dx1;

      // J1: 4-pt avg of J1 at (ks,j,i-1),(ks,j,i),(ks,j+1,i-1),(ks,j+1,i)
      Real j1 = 0.25*(
        (b0.x3f(m,ks,j,  i-1) - b0.x3f(m,ks,j-1,i-1))/dx2
       +(b0.x3f(m,ks,j,  i  ) - b0.x3f(m,ks,j-1,i  ))/dx2
       +(b0.x3f(m,ks,j+1,i-1) - b0.x3f(m,ks,j,  i-1))/dx2
       +(b0.x3f(m,ks,j+1,i  ) - b0.x3f(m,ks,j,  i  ))/dx2);

      // J3: 2-pt avg of J3 at (ks,j,i),(ks,j+1,i)
      Real j3 = 0.5*(
        (b0.x2f(m,ks,j,  i) - b0.x2f(m,ks,j,  i-1))/dx1
       -(b0.x1f(m,ks,j,  i) - b0.x1f(m,ks,j-1,i  ))/dx2
       +(b0.x2f(m,ks,j+1,i) - b0.x2f(m,ks,j+1,i-1))/dx1
       -(b0.x1f(m,ks,j+1,i) - b0.x1f(m,ks,j,  i  ))/dx2);

      // B: B1 exact, B2 = 4-pt avg of cell-centred B2, B3 single x3f
      Real b1 = b0.x1f(m,ks,j,i);
      Real b2 = 0.25*(b0.x2f(m,ks,j,  i) + b0.x2f(m,ks,j+1,i)
                    + b0.x2f(m,ks,j,  i-1) + b0.x2f(m,ks,j+1,i-1));
      Real b3 = 0.5*(b0.x3f(m,ks,j,i) + b0.x3f(m,ks,j,i-1));

      Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
      Real jdb = j1*b1 + j2*b2 + j3*b3;
      Real e2_val = eta_a*(j2 - jdb*b2/bsq);
      e2(m,ks,  j,i) += e2_val;
      e2(m,ke+1,j,i) += e2_val;
    });

    // -- E3 at (ks, j-1/2, i-1/2) --
    par_for("ambi2_e3", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      Real dx1 = mbsize.d_view(m).dx1;
      Real dx2 = mbsize.d_view(m).dx2;

      // J3 direct: dB2/dx1 - dB1/dx2
      Real j3 = (b0.x2f(m,ks,j,i) - b0.x2f(m,ks,j,i-1))/dx1
               -(b0.x1f(m,ks,j,i) - b0.x1f(m,ks,j-1,i))/dx2;

      // J1: 2-pt avg of J1 at (ks,j,i-1),(ks,j,i)
      Real j1 = 0.5*(
        (b0.x3f(m,ks,j,i-1) - b0.x3f(m,ks,j-1,i-1))/dx2
       +(b0.x3f(m,ks,j,i  ) - b0.x3f(m,ks,j-1,i  ))/dx2);

      // J2: 2-pt avg of J2 at (ks,j,i),(ks,j-1,i)
      Real j2 = 0.5*(
        -(b0.x3f(m,ks,j,  i) - b0.x3f(m,ks,j,  i-1))/dx1
        -(b0.x3f(m,ks,j-1,i) - b0.x3f(m,ks,j-1,i-1))/dx1);

      // B: B1 2-pt avg, B2 2-pt avg, B3 = 4-pt avg of cell-centred B3
      Real b1 = 0.5*(b0.x1f(m,ks,j,i) + b0.x1f(m,ks,j-1,i));
      Real b2 = 0.5*(b0.x2f(m,ks,j,i) + b0.x2f(m,ks,j,i-1));
      // cell-centred B3(ks,j,i) = 0.5*(x3f(ks,j,i)+x3f(ks+1,j,i))
      Real b3 = 0.125*(b0.x3f(m,ks,  j,  i) + b0.x3f(m,ks+1,j,  i)
                     + b0.x3f(m,ks,  j,  i-1) + b0.x3f(m,ks+1,j,  i-1)
                     + b0.x3f(m,ks,  j-1,i) + b0.x3f(m,ks+1,j-1,i)
                     + b0.x3f(m,ks,  j-1,i-1) + b0.x3f(m,ks+1,j-1,i-1));

      Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
      Real jdb = j1*b1 + j2*b2 + j3*b3;
      e3(m,ks,j,i) += eta_a*(j3 - jdb*b3/bsq);
    });
    return;
  }

  //---- 3-D problem:
  // Three separate flat par_for kernels (one per EMF component).
  // J cross-terms: 4-point averages to the correct edge position.
  // B diagonal: 8-point avg of face values (= 4-pt avg of cell-centred B).
  // This matches Athena++ FieldDiffusion::AmbipolarEMF (3D branch) exactly.

  // -- E1 at (k-1/2, j-1/2, i) --
  par_for("ambi3_e1", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    Real dx3 = mbsize.d_view(m).dx3;

    // J1 at (k-1/2, j-1/2, i): direct
    Real j1 = (b0.x3f(m,k,j,i) - b0.x3f(m,k,j-1,i))/dx2
             -(b0.x2f(m,k,j,i) - b0.x2f(m,k-1,j,i))/dx3;

    // J2 at (k-1/2, j-1/2, i): 4-pt avg of J2 positions
    // J2(k,j,x) = -dB3/dx1 + dB1/dx3  at  (k-1/2, j, x-1/2)
    Real j2 = 0.25*(
      -(b0.x3f(m,k,  j,  i  ) - b0.x3f(m,k,  j,  i-1))/dx1
      +(b0.x1f(m,k,  j,  i  ) - b0.x1f(m,k-1,j,  i  ))/dx3
      -(b0.x3f(m,k,  j,  i+1) - b0.x3f(m,k,  j,  i  ))/dx1
      +(b0.x1f(m,k,  j,  i+1) - b0.x1f(m,k-1,j,  i+1))/dx3
      -(b0.x3f(m,k,  j-1,i  ) - b0.x3f(m,k,  j-1,i-1))/dx1
      +(b0.x1f(m,k,  j-1,i  ) - b0.x1f(m,k-1,j-1,i  ))/dx3
      -(b0.x3f(m,k,  j-1,i+1) - b0.x3f(m,k,  j-1,i  ))/dx1
      +(b0.x1f(m,k,  j-1,i+1) - b0.x1f(m,k-1,j-1,i+1))/dx3);

    // J3 at (k-1/2, j-1/2, i): 4-pt avg of J3 positions
    // J3(k,j,x) = dB2/dx1 - dB1/dx2  at  (k, j-1/2, x-1/2)
    Real j3 = 0.25*(
       (b0.x2f(m,k,  j,  i  ) - b0.x2f(m,k,  j,  i-1))/dx1
      -(b0.x1f(m,k,  j,  i  ) - b0.x1f(m,k,  j-1,i  ))/dx2
      +(b0.x2f(m,k,  j,  i+1) - b0.x2f(m,k,  j,  i  ))/dx1
      -(b0.x1f(m,k,  j,  i+1) - b0.x1f(m,k,  j-1,i+1))/dx2
      +(b0.x2f(m,k-1,j,  i  ) - b0.x2f(m,k-1,j,  i-1))/dx1
      -(b0.x1f(m,k-1,j,  i  ) - b0.x1f(m,k-1,j-1,i  ))/dx2
      +(b0.x2f(m,k-1,j,  i+1) - b0.x2f(m,k-1,j,  i  ))/dx1
      -(b0.x1f(m,k-1,j,  i+1) - b0.x1f(m,k-1,j-1,i+1))/dx2);

    // B at (k-1/2, j-1/2, i):
    // B1: 8-pt avg of x1f = 4-pt avg of cell-centred B1
    Real b1 = 0.125*(b0.x1f(m,k,  j,  i) + b0.x1f(m,k,  j,  i+1)
                   + b0.x1f(m,k,  j-1,i) + b0.x1f(m,k,  j-1,i+1)
                   + b0.x1f(m,k-1,j,  i) + b0.x1f(m,k-1,j,  i+1)
                   + b0.x1f(m,k-1,j-1,i) + b0.x1f(m,k-1,j-1,i+1));
    // B2, B3: already land on edge with 2-pt face avg
    Real b2 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k-1,j,i));
    Real b3 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j-1,i));

    Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
    Real jdb = j1*b1 + j2*b2 + j3*b3;
    e1(m,k,j,i) += eta_a*(j1 - jdb*b1/bsq);
  });

  // -- E2 at (k-1/2, j, i-1/2) --
  par_for("ambi3_e2", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    Real dx3 = mbsize.d_view(m).dx3;

    // J2 at (k-1/2, j, i-1/2): direct
    Real j2 = -(b0.x3f(m,k,j,i) - b0.x3f(m,k,j,i-1))/dx1
              +(b0.x1f(m,k,j,i) - b0.x1f(m,k-1,j,i))/dx3;

    // J1 at (k-1/2, j, i-1/2): 4-pt avg of J1 positions
    // J1(k,j,x) = dB3/dx2 - dB2/dx3  at  (k-1/2, j-1/2, x)
    Real j1 = 0.25*(
       (b0.x3f(m,k,  j,  i-1) - b0.x3f(m,k,  j-1,i-1))/dx2
      -(b0.x2f(m,k,  j,  i-1) - b0.x2f(m,k-1,j,  i-1))/dx3
      +(b0.x3f(m,k,  j,  i  ) - b0.x3f(m,k,  j-1,i  ))/dx2
      -(b0.x2f(m,k,  j,  i  ) - b0.x2f(m,k-1,j,  i  ))/dx3
      +(b0.x3f(m,k,  j+1,i-1) - b0.x3f(m,k,  j,  i-1))/dx2
      -(b0.x2f(m,k,  j+1,i-1) - b0.x2f(m,k-1,j+1,i-1))/dx3
      +(b0.x3f(m,k,  j+1,i  ) - b0.x3f(m,k,  j,  i  ))/dx2
      -(b0.x2f(m,k,  j+1,i  ) - b0.x2f(m,k-1,j+1,i  ))/dx3);

    // J3 at (k-1/2, j, i-1/2): 4-pt avg of J3 positions
    // J3(k,j,x) = dB2/dx1 - dB1/dx2  at  (k, j-1/2, x-1/2)
    Real j3 = 0.25*(
       (b0.x2f(m,k,  j,  i  ) - b0.x2f(m,k,  j,  i-1))/dx1
      -(b0.x1f(m,k,  j,  i  ) - b0.x1f(m,k,  j-1,i  ))/dx2
      +(b0.x2f(m,k,  j+1,i  ) - b0.x2f(m,k,  j+1,i-1))/dx1
      -(b0.x1f(m,k,  j+1,i  ) - b0.x1f(m,k,  j,  i  ))/dx2
      +(b0.x2f(m,k-1,j,  i  ) - b0.x2f(m,k-1,j,  i-1))/dx1
      -(b0.x1f(m,k-1,j,  i  ) - b0.x1f(m,k-1,j-1,i  ))/dx2
      +(b0.x2f(m,k-1,j+1,i  ) - b0.x2f(m,k-1,j+1,i-1))/dx1
      -(b0.x1f(m,k-1,j+1,i  ) - b0.x1f(m,k-1,j,  i  ))/dx2);

    // B at (k-1/2, j, i-1/2):
    // B1: 2-pt face avg (already correct position)
    Real b1 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k-1,j,i));
    // B2: 8-pt avg of x2f = 4-pt avg of cell-centred B2
    Real b2 = 0.125*(b0.x2f(m,k,  j,  i) + b0.x2f(m,k,  j+1,i)
                   + b0.x2f(m,k,  j,  i-1) + b0.x2f(m,k,  j+1,i-1)
                   + b0.x2f(m,k-1,j,  i) + b0.x2f(m,k-1,j+1,i)
                   + b0.x2f(m,k-1,j,  i-1) + b0.x2f(m,k-1,j+1,i-1));
    // B3: 2-pt face avg (already correct position)
    Real b3 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j,i-1));

    Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
    Real jdb = j1*b1 + j2*b2 + j3*b3;
    e2(m,k,j,i) += eta_a*(j2 - jdb*b2/bsq);
  });

  // -- E3 at (k, j-1/2, i-1/2) --
  par_for("ambi3_e3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    Real dx3 = mbsize.d_view(m).dx3;

    // J3 at (k, j-1/2, i-1/2): direct
    Real j3 = (b0.x2f(m,k,j,i) - b0.x2f(m,k,j,i-1))/dx1
             -(b0.x1f(m,k,j,i) - b0.x1f(m,k,j-1,i))/dx2;

    // J1 at (k, j-1/2, i-1/2): 4-pt avg of J1 positions
    // J1(k,j,x) = dB3/dx2 - dB2/dx3  at  (k-1/2, j-1/2, x)
    Real j1 = 0.25*(
       (b0.x3f(m,k,  j,  i-1) - b0.x3f(m,k,  j-1,i-1))/dx2
      -(b0.x2f(m,k,  j,  i-1) - b0.x2f(m,k-1,j,  i-1))/dx3
      +(b0.x3f(m,k,  j,  i  ) - b0.x3f(m,k,  j-1,i  ))/dx2
      -(b0.x2f(m,k,  j,  i  ) - b0.x2f(m,k-1,j,  i  ))/dx3
      +(b0.x3f(m,k+1,j,  i-1) - b0.x3f(m,k+1,j-1,i-1))/dx2
      -(b0.x2f(m,k+1,j,  i-1) - b0.x2f(m,k,  j,  i-1))/dx3
      +(b0.x3f(m,k+1,j,  i  ) - b0.x3f(m,k+1,j-1,i  ))/dx2
      -(b0.x2f(m,k+1,j,  i  ) - b0.x2f(m,k,  j,  i  ))/dx3);

    // J2 at (k, j-1/2, i-1/2): 4-pt avg of J2 positions
    // J2(k,j,x) = -dB3/dx1 + dB1/dx3  at  (k-1/2, j, x-1/2)
    Real j2 = 0.25*(
      -(b0.x3f(m,k,  j,  i  ) - b0.x3f(m,k,  j,  i-1))/dx1
      +(b0.x1f(m,k,  j,  i  ) - b0.x1f(m,k-1,j,  i  ))/dx3
      -(b0.x3f(m,k,  j-1,i  ) - b0.x3f(m,k,  j-1,i-1))/dx1
      +(b0.x1f(m,k,  j-1,i  ) - b0.x1f(m,k-1,j-1,i  ))/dx3
      -(b0.x3f(m,k+1,j,  i  ) - b0.x3f(m,k+1,j,  i-1))/dx1
      +(b0.x1f(m,k+1,j,  i  ) - b0.x1f(m,k,  j,  i  ))/dx3
      -(b0.x3f(m,k+1,j-1,i  ) - b0.x3f(m,k+1,j-1,i-1))/dx1
      +(b0.x1f(m,k+1,j-1,i  ) - b0.x1f(m,k,  j-1,i  ))/dx3);

    // B at (k, j-1/2, i-1/2):
    // B1: 2-pt face avg
    Real b1 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j-1,i));
    // B2: 2-pt face avg
    Real b2 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j,i-1));
    // B3: 8-pt avg of x3f = 4-pt avg of cell-centred B3
    Real b3 = 0.125*(b0.x3f(m,k,  j,  i) + b0.x3f(m,k+1,j,  i)
                   + b0.x3f(m,k,  j,  i-1) + b0.x3f(m,k+1,j,  i-1)
                   + b0.x3f(m,k,  j-1,i) + b0.x3f(m,k+1,j-1,i)
                   + b0.x3f(m,k,  j-1,i-1) + b0.x3f(m,k+1,j-1,i-1));

    Real bsq = b1*b1 + b2*b2 + b3*b3 + static_cast<Real>(1e-300);
    Real jdb = j1*b1 + j2*b2 + j3*b3;
    e3(m,k,j,i) += eta_a*(j3 - jdb*b3/bsq);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddFluxConstantAmbi()
//  \brief Adds Poynting flux from ambipolar diffusion to energy flux.
//
//  S_AD = E_AD x B = eta_AD * J_perp x B.
//  Since J_parallel x B = 0, this equals eta_AD * (J x B) --
//  identical in form to the Ohmic Poynting flux (AddFluxConstantResist)
//  with eta_ohm replaced by eta_ambi.

void AmbipolarDiffusion::AddFluxConstantAmbi(const DvceFaceFld4D<Real> &b,
                                             DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real qa = 0.25*eta_ambi;

  //------------------------------
  // energy fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("ambi_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j2k   = -(b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1;
    Real j2kp1 = -(b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j,i-1))/size.d_view(m).dx1;

    Real j3j   = (b.x2f(m,k,j  ,i) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1;
    Real j3jp1 = (b.x2f(m,k,j+1,i) - b.x2f(m,k,j+1,i-1))/size.d_view(m).dx1;

    if (multi_d) {
      j3j   -= (b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2;
      j3jp1 -= (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      j2k   += (b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3;
      j2kp1 += (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // flx1 = (E_AD x B)_1 = eta_AD*(J x B)_1 = eta_AD*(J2*B3 - J3*B2)
    flx1(m,IEN,k,j,i) += qa*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("ambi_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1k   = (b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2;
    Real j1kp1 = (b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j-1,i))/size.d_view(m).dx2;

    Real j3i   = (b.x2f(m,k,j,i  ) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i  ) - b.x1f(m,k,j-1,i  ))/size.d_view(m).dx2;
    Real j3ip1 = (b.x2f(m,k,j,i+1) - b.x2f(m,k,j  ,i  ))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i+1) - b.x1f(m,k,j-1,i+1))/size.d_view(m).dx2;

    if (three_d) {
      j1k   -= (b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3;
      j1kp1 -= (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // flx2 = (E_AD x B)_2 = eta_AD*(J x B)_2 = eta_AD*(J3*B1 - J1*B3)
    flx2(m,IEN,k,j,i) += qa*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("ambi_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    // flx3 = (E_AD x B)_3 = eta_AD*(J x B)_3 = eta_AD*(J1*B2 - J2*B1)
    flx3(m,IEN,k,j,i) += qa*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::NewTimeStep()
//! \brief Recomputes parabolic timestep constraint each step (mirrors Resistivity).

void AmbipolarDiffusion::NewTimeStep(const DvceArray5D<Real> &w0,
    const EOS_Data &eos_data) {
  dtnew = std::numeric_limits<float>::max();
  auto &size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m = 0; m < (pmy_pack->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ambi);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/eta_ambi);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/eta_ambi);
    }
  }
  return;
}
