#ifndef DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
#define DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ambipolar_diffusion.hpp
//  \brief Contains data and functions that implement ambipolar diffusion, a non-ideal
//  MHD process important for star formation, protoplanetary disks, and molecular clouds.
//  Physics follows Bai & Stone 2011 (ApJ 736, 144).
//
//  The ambipolar diffusion EMF acts only on the perpendicular current:
//    E_AD = eta_AD * J_perp = eta_AD * (J - (J.B/B^2)*B)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

//----------------------------------------------------------------------------------------
//! \class AmbipolarDiffusion
//  \brief data and functions that implement ambipolar diffusion

class AmbipolarDiffusion {
 public:
  AmbipolarDiffusion(MeshBlockPack *pp, ParameterInput *pin);
  ~AmbipolarDiffusion();

  // data
  Real dtnew;
  std::string ambi_type;  // "constant" (only type currently implemented)
  Real eta_ambi;          // constant ambipolar diffusion coefficient

  // functions to add ambipolar diffusion E-Field and energy flux
  void AddAmbipolarEMFs(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddAmbipolarFluxes(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);
  void AddEMFConstantAmbi(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantAmbi(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
