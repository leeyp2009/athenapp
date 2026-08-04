//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <iomanip>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>     // vector

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
//#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../outputs/outputs.hpp"
#include "../scalars/scalars.hpp"
//#include "../nbody/nbody.cpp"
#include "../nbody/planet.hpp" //planet class
#include "../nbody/nbutil.hpp" // Embedded7_8

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
void GetSphCoord(Coordinates *pco,Real &rad,Real &theta,Real &phi,int i,int j,int k);
//void GetPlanetAcc(const int order, Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gap(const Real rad, const Real phi, const Real z, const Real diff);
//Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
//                        const Real den_ratio, const Real H_ratio);
//Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
// problem parameters which are useful to make global to this file

Real tdf1(const Real rad, const Real d2);
Real tau_relax, rad_soft, gmstar, gmp, inv_sqrt2gmp, rad_planet, phi_planet_0, z_planet, ecc_planet, inc_planet,
inv_rad_planet, t0_planet, t_end_planet, Pp_time, vk_planet, omega_planet, inv_omega_planet, cs_planet,
gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, beta, gMth, nu_alpha,
dfloor, Omega0, user_dt, sigma0, amp, A_gap;
Real rad_planet1, phi_planet_1, z_planet1,gmp1, ecc_planet1, inc_planet1;
Real hst_next_time, hst_dt;
int planet_output, res_flag;
//RCUT_HILL
Real res, area;
bool FeelDisk_Flag;

//Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], weight_dust[NDUSTFLUIDS];
bool Isothermal_Flag, Accretion_Flag, MassTransfer_Flag, RadiativeConduction_Flag,
		 Gap_Flag, TransferFeedback_Flag;
int Damp_Flag;
bool HAF_DISK;
bool HillCut_Flag;
Real rcut;
bool ThreeD_Force;

Real x1min, x1max, tau_damping, damping_rate, accretion_radius, accretion_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
    inner_width_damping, outer_width_damping;
int PlanetaryGravityOrder;
int nPlanet;
vector<Planet> PS;
int NT;
bool BINARY;
bool FIX_PHI;
int IndirectTerm, feelOthers;
Real binary_orb;
Real ODE_TOL;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void InnerWavedamping2(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void OuterWavedamping2(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar, const int i);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void PlanetAccretion(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons,  AthenaArray<Real> &cons_scalar, const int i);
Real PlanetAccretionHistory(MeshBlock *pmb, int iout);

void update_planet(Real dt, int NT);
void derivs_facc(const Real& x, const Real y[], Real dydx[]);

void PlanetUpdateFromDisk(MeshBlock *pmb, int np);


void WritePlanet(Real time);

// User Stopping time
//void StoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
//    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);

// User-defined condutivity
void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);

// User-defined orbital velocity
Real UserOrbitalVelocity(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
// x1 direction
Real UserOrbitalVelocity_r(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
// x3 direction in Cartesian and cylindrical, x2 direction in spherical polar
Real UserOrbitalVelocity_z(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
int RefinementCondition(MeshBlock *pmb);
int RefinementCondition0(MeshBlock *pmb);
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX1_2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1_2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //BINARY = 0;
  FIX_PHI = 0;
  feelOthers = 0;
  //NT     = 5;
  binary_orb = -1;
  ODE_TOL = 1.0e-9;
  // Get parameters for gravitatonal potential of central point mass
  gm0                      = pin->GetOrAddReal("problem", "GM", 0.0);
  r0                       = pin->GetOrAddReal("problem", "r0", 1.0);
  Accretion_Flag           = pin->GetOrAddBoolean("problem", "Acc_Flag", false);
  Isothermal_Flag          = pin->GetOrAddBoolean("problem", "Iso_Flag", false);
  Gap_Flag                 = pin->GetOrAddBoolean("problem", "Gap_Flag", false);
  MassTransfer_Flag        = pin->GetOrAddBoolean("problem", "MassTransfer_Flag", false);
  RadiativeConduction_Flag = pin->GetOrAddBoolean("problem", "RadiativeConduction_Flag", false);
  TransferFeedback_Flag    = pin->GetOrAddBoolean("problem", "TransferFeedback_Flag", true);
  //Relaxation_Flag = pin->GetBoolean("problem",   "Relaxation_Flag");
  FeelDisk_Flag            = pin->GetOrAddBoolean("problem", "FeelDisk_Flag", false);
  Damp_Flag                = pin->GetOrAddInteger("problem", "Damp_Flag", 0);
  ODE_TOL                  = pin->GetOrAddReal("problem", "ODE_TOL", 1.0e-9);
  NT                       = pin->GetOrAddInteger("problem",  "NT", 5); // control timestep for N-body

  // Get parameters for initial density and velocity
  rho0   = pin->GetReal("problem", "rho0");
  dslope = pin->GetOrAddReal("problem", "dslope", -1.0);
  A_gap  = pin->GetOrAddReal("problem", "A_gap", 0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
    gamma_gas  = pin->GetReal("hydro", "gamma");
    beta       = pin->GetOrAddReal("problem", "beta", 0.0);
    if (beta < 0.0) beta = 0.0;
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
    //pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  nu_alpha       = pin->GetOrAddReal("problem", "nu_alpha",  0.0);
  //dffloor        = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0         = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);

  if (!(orbital_advection))
    Omega0 = 0.0;


  // Get AMR parameters
  res = pin->GetOrAddReal("problem","res",0.0002);
  area = pin->GetOrAddReal("problem","area",0.006);

  HAF_DISK     = pin->GetOrAddBoolean("problem", "HAF_DISK", true);


  // The parameters of one planet
  tau_relax        = pin->GetOrAddReal("hydro",      "tau_relax",    0.01);
  nPlanet          = pin->GetOrAddInteger("problem", "Np",   1); // Number of the planet
  rad_planet       = pin->GetOrAddReal("problem",    "rad",   1.0); // radial position of the planet
  phi_planet_0     = pin->GetOrAddReal("problem",    "phi",   0.0); // azimuthal position of the planet
  //z_planet         = pin->GetOrAddReal("problem",    "z",     0.0); // vertical position of the planet
  ecc_planet       = pin->GetOrAddReal("problem",    "ecc",   0.0); // eccentricity of the planet
  inc_planet       = pin->GetOrAddReal("problem",    "inc",   0.0); // inclination of the planet
  gmp              = pin->GetOrAddReal("problem",    "GMp",          0.0); // GM of the planet
  IndirectTerm     = pin->GetOrAddInteger("problem",    "IndTerm",  1); // indirect term of the planet potential
if (nPlanet==2){
  rad_planet1       = pin->GetOrAddReal("problem",    "rad1",   1.0); // radial position of the planet
  phi_planet_1      = pin->GetOrAddReal("problem",    "phi1",   0.0); // azimuthal position of the planet
  //z_planet1         = pin->GetOrAddReal("problem",    "z1",     0.0); // vertical position of the planet
  ecc_planet1       = pin->GetOrAddReal("problem",    "ecc1",   0.0); // eccentricity of the planet
  inc_planet1       = pin->GetOrAddReal("problem",    "inc1",   0.0); // inclination of the planet
  gmp1              = pin->GetOrAddReal("problem",    "GMp1",          0.0); // GM of the planet
  feelOthers        = pin->GetOrAddInteger("problem",  "feelOthers", 1); // feel gravity from another planet
  BINARY            = pin->GetOrAddInteger("problem",  "Binary", 0); // binary flag
}
  t0_planet        = (pin->GetOrAddReal("problem",   "t0p",    0.0))*TWO_PI; // time to put in the planet
  t_end_planet     = (pin->GetOrAddReal("problem",   "tendp", HUGE_NUMBER))*TWO_PI; // time to disapear the planet
  user_dt          = pin->GetOrAddReal("problem",    "user_dt",      0.0);
  planet_output    = pin->GetOrAddInteger("problem","planet_output",1);
  res_flag         = pin->GetOrAddInteger("problem","restart",0);
  Pp_time          = (pin->GetOrAddReal("problem",   "Pp",    0.0))*TWO_PI; // ram-up timescale for planet mass

  HillCut_Flag     = pin->GetOrAddBoolean("problem", "HillCut_Flag", false);
  rcut             = pin->GetOrAddReal("problem",    "rcut",   1.0); // Torque cutoff in unit of Hill radius

  ThreeD_Force     = pin->GetOrAddBoolean("problem", "TDF_Flag", false); // 3D force flag

  if(BINARY) {
    //Omega0 = Omega0*std::sqrt(1.0+gmp+gmp1);
    Omega0 = Omega0*std::sqrt(1.0+gmp+gmp1)*std::cos(inc_planet);
    FIX_PHI        = pin->GetOrAddInteger("problem",  "FIX_PHI", 1); // feel gravity from another planet
    feelOthers = 1;
  } else {
    //Omega0 = Omega0*std::sqrt(1.0+gmp);
    Omega0 = Omega0*std::sqrt(1.0+gmp)*std::cos(inc_planet);
  }

  //hst_next_time = time;
    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 6, "output") == 0) {
        OutputParameters op;
        std::string outn = pib->block_name.substr(6);
        op.block_number = atoi(outn.c_str());
        op.block_name.assign(pib->block_name);
        op.next_time = pin->GetOrAddReal(op.block_name,"next_time", time);
        op.dt = pin->GetReal(op.block_name,"dt");
        op.file_type = pin->GetString(op.block_name,"file_type");
        if (op.file_type.compare("hst") == 0) {
          hst_dt = op.dt;
          hst_next_time = op.next_time;
        }
      }
      pib = pib->pnext;
    }


  PlanetaryGravityOrder = pin->GetOrAddInteger("problem", "PlanetaryGravityOrder", 2);
  if ((PlanetaryGravityOrder != 2) || (PlanetaryGravityOrder != 4) || (PlanetaryGravityOrder != 6))
    PlanetaryGravityOrder = 2;

  if (NON_BAROTROPIC_EOS)
    cs_planet = std::sqrt(p0_over_r0*std::pow(rad_planet/r0, pslope));
  else
    cs_planet = std::sqrt(p0_over_r0);

  if (gmp != 0.0) inv_sqrt2gmp = 1.0/std::sqrt(2.0*gmp);

  //gMth = gm0*SQR(cs_planet)*cs_planet/(SQR(vk_planet)*vk_planet);

  if (t_end_planet < t0_planet)
    t_end_planet = t0_planet;

  rad_soft  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  accretion_radius  = pin->GetOrAddReal("problem", "racc", 0.3); // Accretion radius of planets


  //accretion_radius *= Hill_radius;

  //if (accretion_radius < rad_soft)
  //  accretion_radius = 1.1*rad_soft;

  accretion_rate = pin->GetOrAddReal("problem", "rate", 1.0); // Removal rate of planets

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);





  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    if ((Damp_Flag==4)||(Damp_Flag==0)) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1_2); // fixed bd
    } else {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
    }
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    if ((Damp_Flag==4)||(Damp_Flag==0)) {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1_2); // fixed bd
    } else {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
    }
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }


  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined timestep: depends on binary separation
 if(nPlanet>1) {
  EnrollUserTimeStepFunction(MyTimeStep);
 }

  if (nPlanet>0) {
    //AllocateUserHistoryOutput(4*nPlanet);
    AllocateUserHistoryOutput(nPlanet);
    for(int n=0;n<nPlanet;n++) { 
     std::stringstream dmass_name, djdot_name, dedot_name;
     //std::stringstream dmass_name;
     //std::stringstream djdot_name;
     //std::stringstream dedot_name;
     dmass_name << "dmass_" << n + 1;
     EnrollUserHistoryOutput(n, PlanetAccretionHistory, dmass_name.str().c_str(),UserHistoryOperation::sum); // default: sum
    }
 }


  Real *mass0 = new Real[nPlanet];
  Real *rp0 = new Real[nPlanet];
  Real *phi0 = new Real[nPlanet];
  Real *theta0 = new Real[nPlanet];
  Real *ecc0 = new Real[nPlanet];
  Real *inc0 = new Real[nPlanet];
  Real *vp0 = new Real[nPlanet];
  Real *vr0 = new Real[nPlanet];
  Real *vt0 = new Real[nPlanet];

  if (nPlanet==1){
    mass0[0] = gmp;
    rp0[0] = rad_planet;
    phi0[0] = phi_planet_0*PI;
    ecc0[0] = ecc_planet;
    inc0[0] = inc_planet;
    theta0[0] = PI/2.0;
  } else if (nPlanet==2) {
    mass0[0] = gmp;
    rp0[0] = rad_planet;
    phi0[0] = phi_planet_0*PI;
    ecc0[0] = ecc_planet;
    inc0[0] = inc_planet;
    theta0[0] = PI/2.0;
    mass0[1] = gmp1;
    rp0[1] = rad_planet1;
    phi0[1] = phi_planet_1*PI;
    ecc0[1] = ecc_planet1;
    inc0[1] = inc_planet1;
    theta0[1] = PI/2.0;
  }
  
  
    //vk_planet        = vp0[0];
    //omega_planet     = vk_planet/rad_planet;
    //inv_omega_planet = 1.0/omega_planet;
    //inv_rad_planet   = 1.0/rad_planet;
  
  if (BINARY) {
     Real rbin = fabs(rp0[1]-rp0[0]);
     Real r_COM = (mass0[0]*rp0[0] + mass0[1]*rp0[1])/(mass0[0]+mass0[1]); // if phi0=phi1
     Real vk = sqrt((mass0[0]+mass0[1]+1.0)/r_COM); //vk at center of mass
     Real vbin0 = sqrt(mass0[1]*mass0[1]/rbin/(mass0[0]+mass0[1]));
     Real vbin1 = sqrt(mass0[0]*mass0[0]/rbin/(mass0[0]+mass0[1]));
     //Omega0 = std::sqrt((1.0+mass0[0]+mass0[1])/r_COM)/r_COM;
  
     vp0[0] = vk + binary_orb*vbin0 - Omega0*rp0[0];
     vp0[1] = vk - binary_orb*vbin1 - Omega0*rp0[1];
     vr0[0] = 0.0;
     vr0[1] = 0.0;
     vt0[0] = 0.0;
     vt0[1] = 0.0;
  } else {
    //Omega0 = Omega0*std::sqrt((1.0+gmp));
    for (int n = 0; n<nPlanet; n++){
      rp0[n] *= (1.0-ecc0[n]);
      vp0[n] = std::sqrt((mass0[n]+1.0)/rp0[n])*std::sqrt(1.0+ecc0[n]);
      vt0[n] = -vp0[n]*std::sin(inc0[0]);
      vp0[n] = vp0[n]*std::cos(inc0[n]);
      vp0[n] = vp0[n] - Omega0*rp0[n];
      vr0[n] = 0.0;
      //vz0[n] = ;
    }
  }

     for (int n = 0; n<nPlanet; n++){
       PS.push_back(Planet(rp0[n],phi0[n],mass0[n]));
       PS[n].setIndex(n);
       PS[n].setTheta(theta0[n]);
       PS[n].setVr(vr0[n]);
       PS[n].setVp(vp0[n]);
       PS[n].setVt(vt0[n]);
       PS[n].setFeelDisk(FeelDisk_Flag);
     }


  // allocateDataField
      AllocateRealUserMeshDataField(6);
  if (gmp>0.0) { 
  //if ((gmp>0.0)&&(!res_flag)) { }
      //AllocateRealUserMeshDataField(3);
      //ruser_mesh_data[0].NewAthenaArray(9*nPlanet);
      ruser_mesh_data[0].NewAthenaArray(6*nPlanet); // planet position and velocity
      ruser_mesh_data[1].NewAthenaArray(3*nPlanet); // planet accelaration
      ruser_mesh_data[2].NewAthenaArray(3*nPlanet); // planet accelartion corrected for half disk
    for (int n=0;n<nPlanet;n++) {
      //if (!res_flag) {
        //std::cout << "inside res_flag:" << res_flag <<std::endl;
        ruser_mesh_data[0](6*n+0) = rp0[n];
        ruser_mesh_data[0](6*n+1) = phi0[n];
        ruser_mesh_data[0](6*n+2) = theta0[n];
        ruser_mesh_data[0](6*n+3) = vr0[n];
        ruser_mesh_data[0](6*n+4) = vp0[n];
        ruser_mesh_data[0](6*n+5) = vt0[n];
        ruser_mesh_data[1](3*n+0) = 0.0;
        ruser_mesh_data[1](3*n+1) = 0.0;
        ruser_mesh_data[1](3*n+2) = 0.0;
        ruser_mesh_data[2](3*n+0) = 0.0;
        ruser_mesh_data[2](3*n+1) = 0.0;
        ruser_mesh_data[2](3*n+2) = 0.0;
      //}
        //std::cout << "InitUserMesh ruser_mesh_data 1:" << ruser_mesh_data[0](0) <<std::endl;
    }

  }
      //Real rad(0.0), phi(0.0), z(0.0);
      //AllocateRealUserMeshDataField(3);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        ruser_mesh_data[3].NewAthenaArray(mesh_size.nx3,mesh_size.nx1); // for outer boundary
        ruser_mesh_data[4].NewAthenaArray(mesh_size.nx3,mesh_size.nx1); // for outer boundary
        ruser_mesh_data[5].NewAthenaArray(mesh_size.nx3,mesh_size.nx1); // for outer boundary
      for (int i=0; i<mesh_size.nx1; i++) {
      	for (int j=0; j<mesh_size.nx3; j++) {
        //GetCylCoord(pcoord,rad,phi,z,i,0,0); // convert to cylindrical coordinates
        //Real cs_square = PoverRho(rad, phi, z);
        //Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        //Real vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vis_vel_r     = -1.5*(nu_alpha*p0_over_r0)*std::pow(x1max/r0,pslope+0.5);
        ruser_mesh_data[3](j,i) = vis_vel_r;
        ruser_mesh_data[4](j,i) = 1.0;
        ruser_mesh_data[5](j,i) = rho0*std::pow(x1max/r0,dslope);
        }
       }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        ruser_mesh_data[3].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
        ruser_mesh_data[4].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
        ruser_mesh_data[5].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
      for (int i=0; i<mesh_size.nx1; i++) {
      	for (int j=0; j<mesh_size.nx2; j++) {
        //GetCylCoord(pcoord,rad,phi,z,i,0,0); // convert to cylindrical coordinates
        //Real cs_square = PoverRho(rad, phi, z);
        //Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        //Real vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vis_vel_r     = -1.5*(nu_alpha*p0_over_r0)*std::pow(x1max/r0,pslope+0.5);
        ruser_mesh_data[3](j,i) = vis_vel_r;
        ruser_mesh_data[4](j,i) = 1.0;
        ruser_mesh_data[5](j,i) = rho0*std::pow(x1max/r0,dslope);
        }
       }
      }





  // Enroll user-defined thermal conduction
  if (!(Isothermal_Flag && NON_BAROTROPIC_EOS) && (beta > 0.0) && (RadiativeConduction_Flag))
    EnrollConductionCoefficient(RadiativeCondution);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  // Enroll user orbital velocity
  //EnrollOrbitalVelocity(UserOrbitalVelocity);
  // x1 direction
  //EnrollOrbitalVelocityDerivative(0, UserOrbitalVelocity_r);
  // x3 direction in Cartesian and cylindrical, x2 direction in spherical polar
  //EnrollOrbitalVelocityDerivative(1, UserOrbitalVelocity_z);

 if (Globals::my_rank == 0) {
  cout << "test: this is InitUserMeshData"<< endl;
  cout << "gm0:"<< gm0<<endl;
  cout << "r0:"<< r0<<endl;
  cout << "Omega0:"<< Omega0<<endl;
  cout << "t0_planet:"<< t0_planet<<endl;
  cout << "Pp_time:"<< Pp_time<<endl;
  cout << "nPlanet:"<< nPlanet<<endl;
  if (nPlanet==2) {
     cout << "BINARY:"<< BINARY<<endl;
     cout << "FIX_PHI:"<< FIX_PHI<<endl;
     cout << "rad_planet:"<< rp0[0]<<", phi_planet:"<<phi0[0]<<", ecc:"<<ecc0[0]<<", theta_planet:"<<theta0[0]<<
	  ", mp:"<<mass0[0]<<endl;
     cout << "rad_planet2:"<< rp0[1]<<", phi_planet2:"<<phi0[1]<<", ecc2:"<<ecc0[1]<<", theta_planet2:"<<theta0[1]<<
	  ", mp2:"<<mass0[1]<<endl;
     cout << "feelOthers:"<< feelOthers<<endl;
     cout << "binary_orb:"<< binary_orb<<endl;
  } else {
     cout << "BINARY:"<< BINARY<<endl;
     cout << "FIX_PHI:"<< FIX_PHI<<endl;
     cout << "rad_planet:"<< rp0[0]<<", phi_planet:"<<phi0[0]<<", ecc:"<<ecc0[0]<<", theta_planet:"<<theta0[0]<<
	  ", mp:"<<mass0[0]<<endl;

  }
  cout << "planet_output:"<< planet_output<<", res_flag:"<<res_flag<<endl;
  cout << "res:"<< res<<", area:"<<area<<endl;
  cout << "FeelDisk_Flag:"<< FeelDisk_Flag<<endl;
  cout << "Isothermal_Flag:"<< Isothermal_Flag<<endl;
  cout << "Accretion_Flag:"<< Accretion_Flag<<endl;
  cout << "MassTansfer_Flag:"<< MassTransfer_Flag<<endl;
  cout << "RadiativeConduction_Flag:"<< RadiativeConduction_Flag<<endl;
  cout << "Gap_Flag:"<< Gap_Flag<<endl;
  cout << "TansferFeedback_Flag:"<< TransferFeedback_Flag<<endl;
  cout << "Damp_Flag:"<< Damp_Flag<<endl;
  cout << "HAF_DISK:"<< HAF_DISK<<endl;
  cout << "HillCut_Flag:"<< HillCut_Flag<<",rcut:"<<rcut<<endl;
  cout << "ThreeD_Force:"<< ThreeD_Force<<endl;
  cout << "accretion_radius:"<< accretion_radius<<", accretion_rate:"<<accretion_rate<<endl;

  cout << "IndirectTerm:"<< IndirectTerm<<endl;
  cout << "ODE_TOL:"<< ODE_TOL<<endl;


  cout << "rad_soft:"<< rad_soft<<endl;
  cout << "nu_alpha:"<< nu_alpha<<endl;
  cout << "rho0:"<< rho0<<endl;
  cout << "dslope:"<< dslope<<endl;
  cout << "p0_over_r0:"<< p0_over_r0<<endl;
  if (NON_BAROTROPIC_EOS) {
    cout << "pslope:"<< pslope<<endl;
    cout << "gamma_gas:"<< gamma_gas<<endl;
    cout << "beta:"<< beta<<endl;
  }
 }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real x1, x2, x3;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  //std::cout << "problemGen time:" << time <<std::endl;



  //Real qvalue = gmp/gm0;
  //bool vis_defined = phydro->hdif.hydro_diffusion_defined;

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        Real cs_square = PoverRho(rad, phi, z);
        Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        Real vel_K     = vK(porb, x1, x2, x3);

        // compute initial conditions in cylindrical coordinates
        Real den_gas   = DenProfileCyl_gas(rad, phi, z);
        Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad);

        Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
        if (porb->orbital_advection_defined)
          vel_gas_phi -= vK(porb, x1, x2, x3);

        Real pre_diff    = (pslope + dslope)*cs_square;

        phydro->u(IDN, k, j, i) = den_gas;
        phydro->u(IM1, k, j, i) = den_gas*vis_vel_r;

        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2, k, j, i) = den_gas*vel_gas_phi;
          phydro->u(IM3, k, j, i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = den_gas*vel_gas_phi;
        }

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                        + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
        }

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n, k, j, i) = den_gas;
          }
        }
      }
    }
  }

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  MeshBlock *pmb = my_blocks(0);
  bool flag = false;
  //static bool first = 1;
  //Real present_time = time + dt;
  Real present_time = time;
  //int cncycle = ncycle + 1;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  //NT = 1;
  //Real hst_next_time = time;
  //cout << "nb_local in Mesh::UserWorkInLoop: " << nblocal << std::endl;
  if (planet_output) {
    // check flag
    if ((present_time < tlim) && (nlim < 0 || (ncycle + 1) < nlim)
        && (present_time > hst_next_time)) {
      flag = true;
      hst_next_time += hst_dt;
    }
    if ((present_time >= tlim) || (nlim >= 0 && (ncycle + 1) >= nlim)) {
      flag = true;
    }
  }


    AthenaArray<Real> &vr_sum = ruser_mesh_data[3];
    AthenaArray<Real> &number = ruser_mesh_data[4];
    AthenaArray<Real> &dens_sum = ruser_mesh_data[5];

   if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
     for (int i=0; i<mesh_size.nx1; i++) {
      for (int j=0; j<mesh_size.nx3; j++) {
       vr_sum(j,i) = 0.0;
       number(j,i) = 0.0;
       dens_sum(j,i) = 0.0;
     }
    }
   } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
     for (int i=0; i<mesh_size.nx1; i++) {
      for (int j=0; j<mesh_size.nx2; j++) {
       vr_sum(j,i) = 0.0;
       number(j,i) = 0.0;
       dens_sum(j,i) = 0.0;
     }
    }
   }

        //std::cout << "before update, ruser_mesh_data: "<<ruser_mesh_data[0](0)<<", " << ruser_mesh_data[0](1)<< std::endl;
//        std::cout << "vr_out: "<<vr_out << std::endl;

// update ruser_mesh_data
    for (int bn=0; bn<nblocal; ++bn) {
      MeshBlock *pmb = my_blocks(bn);
      LogicalLocation &loc = pmb->loc;
      if (loc.level == root_level) { // root level
      //if (true) { // root level }
      int is = pmb->is, ie = pmb->ie;
      int js = pmb->js, je = pmb->je;
      int ks = pmb->ks, ke = pmb->ke;
      //std::cout << "ks, js ,is: "<<ks<<", "<<js<<", "<<is << std::endl;
      //std::cout << "ke, je ,ie: "<<ke<<", "<<je<<", "<<ie << std::endl;
      AthenaArray<Real> &u = pmb->phydro->u;
      AthenaArray<Real> &w = pmb->phydro->w;
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
         for (int k=ks; k<=ke; ++k) {
          Real x3 = pmb->pcoord->x3v(k);

          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);


         //std::cout <<"rad: " << rad<< ", radius_outer_damping: "<<radius_outer_damping << std::endl;
          if (rad >= radius_outer_damping) {
              int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-pmb->is);
              int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js);
              int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            //Real &gas_dens    = cons(IDN, k, j, i);
            //Real &gas_mom1    = cons(IM1, k, j, i);
            //Real inv_dens_gas = 1.0/gas_dens;
            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vr = prim(IM1, k, j, i);
            
            //const Real gas_rho = u(IDN,k,j,i);
            const Real gas_mom1 = u(IM1,k,j,i);
            const Real gas_rho = w(IDN,k,j,i);
            const Real gas_vel1 = w(IM1,k,j,i);
            const Real gas_vel1_0 = gas_mom1/gas_rho;
            //std::cout << "in loop gas_vel1: "<<gas_vel1 << std::endl;
            //std::cout << "in loop k, j ,i: "<<k<<", "<<j<<", "<<i << std::endl;
            //std::cout << "in loop loc.lev: "<<loc.level<< ", root_lev: "<<root_level << std::endl;

            Real vol = pmb->pcoord->GetCellVolume(k,j,i);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            number(tk,ti)        += 1.0*vol; // index i
            vr_sum(tk,ti) +=gas_vel1*vol;
            dens_sum(tk,ti) +=gas_rho*vol;
            //number(ti)        += 1.0; // index i
            //vr_sum(ti) +=gas_vel1;
            //Real mdot = 2.0*PI*vr_sum(ti)/number(ti)*gas_rho*rad;
            Real mdot = 4.0*PI*vr_sum(ti)/number(ti)*gas_rho*std::pow(rad,2); // 3d
            //std::cout << "in loop ti: "<<ti<< ", number: " << number(ti)<<", vr_sum: "<< vr_sum(ti) << std::endl;
            /*
            //if ((ti==mesh_size.nx1-1) and (j==je)) {
            if ((ti==mesh_size.nx1-1)) {
             std::cout << "in loop rad: "<<rad << std::endl;
             std::cout << "in loop gas_vel1: "<<gas_vel1<<", gas_vel1(cons): "<< gas_vel1_0 << std::endl;
             std::cout << "in UserWorkInLoop mdot: "<<mdot << ", vr: "<< vr_sum(ti)/number(ti)<<std::endl;
	     }
	    }
	    */
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            number(tj,ti)        += 1.0*vol; // index i
            vr_sum(tj,ti) +=gas_vel1*vol;
            dens_sum(tj,ti) +=gas_rho*vol;
        }
	  }
         }	
         //gas_vel1_0 = vr_sum/number;
       }
      }
    //} // coordinates
    } // root level
  }
#ifdef MPI_PARALLEL
      //if (Globals::my_rank == 0) { }
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        int ntot = mesh_size.nx1*mesh_size.nx3;
        //std::cout << "before MPI: vr_sum: "<<vr_sum(ntot-1)<<", vr" << vr_sum(ntot-1)/number(ntot-1)<< std::endl;
    	MPI_Allreduce(MPI_IN_PLACE, vr_sum.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(MPI_IN_PLACE, number.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(MPI_IN_PLACE, dens_sum.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        int ntot = mesh_size.nx1*mesh_size.nx2;
        //std::cout << "before MPI: vr_sum: "<<vr_sum(ntot-1)<<", vr" << vr_sum(ntot-1)/number(ntot-1)<< std::endl;
    	MPI_Allreduce(MPI_IN_PLACE, vr_sum.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(MPI_IN_PLACE, number.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(MPI_IN_PLACE, dens_sum.data(), ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
     }
#endif


// for planet motion
// ============================================================
// ===============================================================
  //if ((gmp > 0.0) && (Globals::my_rank == 0)) 
  if (gmp > 0.0) {
    //std::cout << "UserWorkInLoop update planet position when restart ..." <<std::endl;
    //std::cout << "UserWorkInLoop time:" << time << ", dt:" << dt <<std::endl;
    
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[0].data(), 6*nPlanet, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
     //std::cout << "time:" << time <<std::endl;
     //std::cout << "UserWorkInLoop ruser_mesh_data:" << ruser_mesh_data[0](0) <<std::endl;
     //AthenaArray<Real> &pp = ruser_mesh_data[0];
    //if (res_flag) 
     //std::cout << "time:" << time <<std::endl;
     //std::cout << "update planet position when restart ..." <<std::endl;
     //std::cout << "ruser_mesh_data:" << Mesh::ruser_mesh_data[0](0) <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
      
        for (int n=0;n<nPlanet;n++) {
          PS[n].setRad(ruser_mesh_data[0](6*n+0)); 
          PS[n].setPhi(ruser_mesh_data[0](6*n+1)); 
          PS[n].setTheta(ruser_mesh_data[0](6*n+2)); 
          PS[n].setVr(ruser_mesh_data[0](6*n+3)); 
          PS[n].setVp(ruser_mesh_data[0](6*n+4)); 
          PS[n].setVt(ruser_mesh_data[0](6*n+5)); 
          //PS[n].setFr(ruser_mesh_data[2](3*n+0)); 
          //PS[n].setFp(ruser_mesh_data[2](3*n+1)); 
          //PS[n].setFt(ruser_mesh_data[2](3*n+2)); 
//  if ((gmp > 0.0) && (Globals::my_rank == 0)) {
//          cout << "UserWorkInLoop beforeUpdate: " << PS[n].getFr()<<", "<< PS[n].getFp()<<", "
//              << PS[n].getFt() << std::endl;
//          cout << "UserWorkInLoop beforeUpdate: " << ruser_mesh_data[2](3*n+0)<<", "<< ruser_mesh_data[2](3*n+1)<<", "
//              << ruser_mesh_data[2](3*n+2) << std::endl;
//}
        }
     

     //std::cout << "time:" << time << ", dt:" << dt <<std::endl;
     //std::cout << "update planet position when restart ..." <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
     
     //AthenaArray<Real> &y = ruser_mesh_data[0];
     //Real *y = ruser_mesh_data[0];
     //for (int n=0;n<nPlanet;n++){
     //  PS[n].initializeRK(&y(n*4));
     //}

     //WritePlanet(time);
     //update_planet(dt, NT);

     //cout << "time:" << time << "dt:" << dt << endl;
     //std::cout << "update planet position ..." <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
  }


  if (gmp >0.0) {
     if(nPlanet>1) {
     //std::cout << "update NT ..." <<std::endl;
       Real dt_np = 1000.0;
       for (int i=0; i<nPlanet;i++){
       // get the index of planet location
        for (int j=i+1;j<nPlanet;j++){
              Real dij = PS[i].distance(PS[j]);
              //dt_np = std::min(dt_np, 1.0/(400.0)*dij*
              dt_np = std::min(dt_np, TWO_PI/(400.0)*dij*
		    std::sqrt(dij/(PS[i].getMass()+PS[j].getMass())));
        } // for j
       NT = std::max(NT, int(dt/dt_np+0.6));
       } // for i
   //cout << "NT: before update_planet: " << NT << std::endl;
     } // if nPlanet

   //if (Globals::my_rank == 0 && flag) {
   if (Globals::my_rank == 0) {
      //cout << "NT: before update_planet: " << NT << std::endl;
     //std::cout << "my_rank==0,rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
     WritePlanet(time);
     //update_planet(dt, NT);
   }
     //std::cout << "before update_planet,rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
     //std::cout << "after update_planet, rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
   if (time > t0_planet) {
     update_planet(dt, NT);
   }
     //cout << "NT: after update_planet: " << NT << std::endl;

    for (int n=0;n<nPlanet;n++) {
     Real &Rad_p = ruser_mesh_data[0](6*n+0); 
     Real &Phi_p = ruser_mesh_data[0](6*n+1); 
     Real &Theta_p = ruser_mesh_data[0](6*n+2); 
     Real &Vr_p = ruser_mesh_data[0](6*n+3); 
     Real &Vp_p = ruser_mesh_data[0](6*n+4); 
     Real &Vt_p = ruser_mesh_data[0](6*n+5); 
     Rad_p = PS[n].getRad();
     Phi_p = PS[n].getPhi();
     Theta_p = PS[n].getTheta();
     Vr_p = PS[n].getVr();
     Vp_p = PS[n].getVp();
     Vt_p = PS[n].getVt();
     //std::cout << "Rad_p:" << pp(4*n+0) << "\n" <<std::endl;
     
     //std::cout << "UserWorkInLoop updating ruser_mesh_data:" << ruser_mesh_data[0](4*n) <<std::endl;
     //std::cout << "Rad_p:" << Rad_p << "\n" <<std::endl;
    }

    for (int n=0;n<nPlanet;n++) {
       //AthenaArray<Real> term;
       //term.NewAthenaArray(3);
       //cout << "nb_local in Mesh::UserWorkInLoop: " << nblocal << std::endl;
       //cout << "term(:): before UpdateFromDisk: " <<n << ", " << term(0)<<", "<<term(1)<<", "<< term(2) << std::endl;
       ruser_mesh_data[2](3*n+0) = 0.0; 
       ruser_mesh_data[2](3*n+1) = 0.0; 
       ruser_mesh_data[2](3*n+2) = 0.0; 
       for (int bn=0; bn<nblocal; ++bn) {
         MeshBlock *pmb = my_blocks(bn);
         PlanetUpdateFromDisk(pmb,n);
         //term(0) = PS[n].getFr();
         //term(1) = PS[n].getFp();
         //term(2) = PS[n].getFt();
         //cout << "term(:), bn: UserWorkInLoop beforeMPI: "<<bn<<", " << term(0)<<", "<<term(1)<<", "<< term(2) << std::endl;
         if (HAF_DISK) {
	    ruser_mesh_data[2](3*n+0) += 2.0*ruser_mesh_data[1](3*n+0); 
            ruser_mesh_data[2](3*n+1) += 2.0*ruser_mesh_data[1](3*n+1); 
            ruser_mesh_data[2](3*n+2) += 2.0*ruser_mesh_data[1](3*n+2); 
         } else {
	    ruser_mesh_data[2](3*n+0) += ruser_mesh_data[1](3*n+0); 
            ruser_mesh_data[2](3*n+1) += ruser_mesh_data[1](3*n+1); 
            ruser_mesh_data[2](3*n+2) += ruser_mesh_data[1](3*n+2); 
	 }
         //cout << "UserWorkInLoop beforeMPI meshblock: bn:"<< bn<< ", "<<ruser_mesh_data[1](3*n+0)<<", "<<ruser_mesh_data[1](3*n+1)<<", "
         //     << ruser_mesh_data[1](3*n+2) << std::endl;
       }
         //cout << "UserWorkInLoop beforeMPI: " << ruser_mesh_data[2](3*n+0)<<", "<< ruser_mesh_data[2](3*n+1)<<", "
         //     << ruser_mesh_data[2](3*n+2) << std::endl;
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[2].data(), 3*nPlanet, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      PS[n].setFr(ruser_mesh_data[2](3*n+0)); 
      PS[n].setFp(ruser_mesh_data[2](3*n+1)); 
      PS[n].setFt(ruser_mesh_data[2](3*n+2)); 
//  if ((gmp > 0.0) && (Globals::my_rank == 0)) {
//      cout << "UserWorkInLoop afterUpdate: " << PS[n].getFr()<<", "<< PS[n].getFp()<<", "
//          << PS[n].getFt() << std::endl;
//      cout << "UserWorkInLoop afterUpdate: " << ruser_mesh_data[2](3*n+0)<<", "<< ruser_mesh_data[2](3*n+1)<<", "
//         << ruser_mesh_data[2](3*n+2) << std::endl;
//  }

    /*
    if (Globals::nranks > 1) {
      cout << "Globals::nranks: "<< Globals::nranks  << std::endl;
      if (Globals::my_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, term.data(), 3,
                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(term.data(), term.data(), 3,
                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      }
    }
    */
         //cout << "UserWorkInLoop afterMPI: " << ruser_mesh_data[2](3*n+0)<<", "<< ruser_mesh_data[2](3*n+1)<<", "
         //     << ruser_mesh_data[2](3*n+2) << std::endl;
         //PS[n].setFr(ruser_mesh_data[2](3*n+0)); 
         //PS[n].setFp(ruser_mesh_data[2](3*n+1)); 
         //PS[n].setFt(ruser_mesh_data[2](3*n+2)); 
    }
  } // if gmp>0

  //if (gmp > 0.0) {
  //   update_planet(dt, NT);
  //}

}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {



  if (gmp > 0.0) {
//   if (Globals::my_rank == 0) {
//	update_planet(dt, NT);
//        cout << "time:" << time << "dt:" << dt << endl;
//        std::cout << "update planet position ..." <<std::endl;
//        std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
//        WritePlanet(time);
//    }
   if (time > t0_planet) {
    PlanetaryGravity(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
   }
    //for (int i=0;i<nPlanet;i++){
    //PlanetaryGravity(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar, i);
    //cout << "This is from main() function:\n";
    //cout << "main: rp, phi, vr, omega: "<< PS[0].getRad()<<", " 
    //     << PS[0].getPhi() <<", " << PS[0].getVr() << ", " 
    //     << PS[0].getVp() <<endl;
    //}
}

  if (Isothermal_Flag && NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  else if (beta > 0.0)
    ThermalRelaxation(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  if ((gmp > 0.0) && (time > t0_planet) && Accretion_Flag) {
    for (int np=0;np<nPlanet;np++){
    //std::cout << "Accretion_Flag =1: " << Accretion_Flag << std::endl;
      PlanetAccretion(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar, np);
    }
  }


  if (Damp_Flag==1) {
    //std::cout << "Damp_Flag =1: " << Damp_Flag << std::endl;
     InnerWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar); 
  } else if (Damp_Flag==2) {
    //std::cout << "Damp_Flag =2: " << Damp_Flag << std::endl;
     OuterWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  } else if (Damp_Flag==3) {
    //std::cout << "Damp_Flag =3: " << Damp_Flag << std::endl;
     InnerWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
     OuterWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  } else if (Damp_Flag==4) { // damped to initial value
     InnerWavedamping2(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
     OuterWavedamping2(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  }

//  if ((gmp > 0.0) && MassTransfer_Flag)
//    for (int i=0;i<nPlanet;i++){
//        MassTransferWithinHill(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar,i);
//    }

/*
  if (Isothermal_Flag && NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  else if (beta > 0.0)
    ThermalRelaxation(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
*/
  //cout << "this is the end of MySource:" << time << "dt:" << dt << endl;
  return;
}

/*
void StoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

  Real sqrt_gm0 = std::sqrt(gm0);

  return;
}
*/

// Add planet
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //update_planet(dt, NT);
  //Real phi_planet_move    = omega_planet*time + phi_planet_0;

  
  int nc1 = pmb->ncells1;


  
  //cout << "rad_sft1:" << rad_sft1 << ", dt:" << dt << endl;

  //if (pmb->porb->orbital_advection_defined)
  //  phip -= Omega0*time;

  Real planet_gm, cs_planet, gMth, t_growth, t_disapear, distance_square, distance;
  Real x_dis, y_dis, z_dis, rad_dis, phi_dis, theta_dis;
  Real rad_arr, phi_arr, theta_arr, z_arr;
  Real acc_r, acc_phi, acc_z, acc_theta;
  Real acc_r_temp, acc_phi_temp, acc_z_temp, acc_theta_temp;
  Real hc, tmp;


/*
  AthenaArray<Real> planet_gm, cs_planet, gMth, t_growth, t_disapear, distance_square, distance;
  AthenaArray<Real> x_dis, y_dis, z_dis, rad_dis, phi_dis, theta_dis;
  AthenaArray<Real> rad_arr, phi_arr, theta_arr, z_arr;
  AthenaArray<Real> acc_r, acc_phi, acc_z, acc_theta;
  AthenaArray<Real> acc_r_temp, acc_phi_temp, acc_z_temp, acc_theta_temp;

  planet_gm.NewAthenaArray(nc1);
  cs_planet.NewAthenaArray(nc1);
  gMth.NewAthenaArray(nc1);
  t_growth.NewAthenaArray(nc1);
  t_disapear.NewAthenaArray(nc1);
  distance_square.NewAthenaArray(nc1);
  distance.NewAthenaArray(nc1);

  x_dis.NewAthenaArray(nc1);
  y_dis.NewAthenaArray(nc1);
  z_dis.NewAthenaArray(nc1);

  rad_dis.NewAthenaArray(nc1);
  phi_dis.NewAthenaArray(nc1);
  theta_dis.NewAthenaArray(nc1);

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  theta_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  acc_r.NewAthenaArray(nc1);
  acc_phi.NewAthenaArray(nc1);
  acc_theta.NewAthenaArray(nc1);
  acc_z.NewAthenaArray(nc1);
  acc_r_temp.NewAthenaArray(nc1);
  acc_phi_temp.NewAthenaArray(nc1);
  acc_theta_temp.NewAthenaArray(nc1);
  acc_z_temp.NewAthenaArray(nc1);
*/
  //Real igm1 = 1.0/(gamma_gas - 1.0);

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
		if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
//#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //Real rad_arr, phi_arr, z_arr;
        acc_r = 0.0;
        acc_phi = 0.0;
        acc_z = 0.0;
        GetCylCoord(pmb->pcoord, rad_arr, phi_arr, z_arr, i, j, k);

	for (int np=0;np<nPlanet;np++) {
          Real phip    = PS[np].getPhi();
          Real radp    = PS[np].getRad();
          Real thetap      = PS[np].getTheta();
          Real zp = radp*std::cos(thetap);
          Real Rp = radp*std::sin(thetap);
          if (phip>TWO_PI)
            phip += -TWO_PI;
          else if (phip<0.0)
            phip += TWO_PI;
          planet_gm = PS[np].getMass();
          //Real gmp    = PS[np].getMass();
          Real  Hill_radius = (std::pow(planet_gm/gm0*ONE_3RD, ONE_3RD)*radp);
          Real rad_sft1 = rad_soft*Hill_radius; // 0.6 *r_Hill

          x_dis = -rad_arr*std::cos(phi_arr) + Rp*std::cos(phip);
          y_dis = -rad_arr*std::sin(phi_arr) + Rp*std::sin(phip);

          rad_dis = -(rad_arr - Rp*cos(phi_arr - phip));
          phi_dis = -Rp*sin(phi_arr - phip);
          z_dis = -(z_arr - zp);

          distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //distance        = std::sqrt(distance_square);

          if (HillCut_Flag) {
            if (std::sqrt(distance_square)<rcut*Hill_radius) {
               hc = 0.0;
            } else if (std::sqrt(distance_square)>=Hill_radius) {
	       hc = 1.0;
	    } else {
	       hc = SQR(std::sin(0.5*PI/(1.0-rcut)*(std::sqrt(distance_square)/Hill_radius-rcut)));
            }
          } else {
            hc = 1.0;
          }

          if(ThreeD_Force) {
            tmp = tdf1(Rp,distance_square+SQR(rad_sft1));
            //if (distance_square<SQR(rad_sft1))
            //   cout <<"tmp:"<<tmp<<", distance:"<<std::sqrt(distance_square)<<std::endl;
            //acc_r *= tmp;
            //acc_phi *= tmp;
            //acc_z *= tmp;
          } else {
            tmp = 1.0;
          }
            //second order gravity
          if (PlanetaryGravityOrder == 2) {
            Real sec_g = tmp*hc*planet_gm/pow(distance_square+SQR(rad_sft1), 1.5);
            acc_r   += sec_g*rad_dis; // radial acceleration
            acc_phi += sec_g*phi_dis; // azimuthal acceleration
            acc_z   += sec_g*z_dis;   // vertical acceleartion
          }

          //fourth order gravity
          if (PlanetaryGravityOrder == 4) {
            Real forth_g = tmp*hc*planet_gm*(5.0*SQR(rad_sft1)+2.0*distance_square)/
                                    (2.0*pow(SQR(rad_sft1)+distance_square, 2.5));
            acc_r     += forth_g*rad_dis; // radial acceleration
            acc_phi   += forth_g*phi_dis; // azimuthal acceleration
            acc_z     += forth_g*z_dis;   // vertical acceleartion
          }

          //sixth order gravity
          if (PlanetaryGravityOrder == 6) {
            Real sixth_g = tmp*hc*planet_gm*(35.0*SQR(SQR(rad_sft1))+28.0*SQR(rad_sft1)*distance_square+
                                    8.0*distance_square)/(8.0*pow(SQR(rad_sft1)+distance_square, 3.5));
            acc_r     += sixth_g*rad_dis; // radial acceleration
            acc_phi   += sixth_g*phi_dis; // azimuthal acceleration
            acc_z     += sixth_g*z_dis;   // vertical acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = tmp*hc*planet_gm/SQR(radp);
             acc_r   += -temp*std::cos(phi_arr - phip);
	     acc_phi += temp*std::sin(phi_arr - phip);
	     acc_z   += -temp*zp/Rp;
          }
          }

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
//ramp-up the disk effect using time pmass_incr_time
	  if (time < t0_planet+Pp_time) {
             Real frate = std::sin(time/(2.0*Pp_time)*PI);
             acc_r *= (frate*frate);
             acc_phi *= (frate*frate);
             acc_z *= (frate*frate);
          }
          Real delta_mom1 = +dt*gas_rho*(acc_r);
          Real delta_mom2 = +dt*gas_rho*(acc_phi);
          Real delta_mom3 = +dt*gas_rho*(acc_z);

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
        }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
//#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
			//Real rad_arr, theta_arr, phi_arr;
        acc_r = 0.0;
        acc_phi = 0.0;
        acc_theta = 0.0;
        GetSphCoord(pmb->pcoord, rad_arr, theta_arr, phi_arr, i, j, k);

	for (int np=0;np<nPlanet;np++) {
          Real phip    = PS[np].getPhi();
          Real radp    = PS[np].getRad();
          Real thetap  = PS[np].getTheta();
          Real Rp = radp*std::sin(thetap);
          //Real zp      = PS[np].getZ();
          planet_gm = PS[np].getMass();
          if (phip>TWO_PI)
            phip += -TWO_PI;
          else if (phip<0.0)
            phip += TWO_PI;
          //Real gmp    = PS[np].getMass();
          Real  Hill_radius = (std::pow(planet_gm/gm0*ONE_3RD, ONE_3RD)*radp);
          Real rad_sft1 = rad_soft*Hill_radius; // 0.6 *r_Hill

          x_dis = -rad_arr*std::cos(phi_arr)*std::sin(theta_arr) + 
		  radp*std::cos(phip)*std::sin(thetap);
          y_dis = -rad_arr*std::sin(phi_arr)*std::sin(theta_arr) + 
		  radp*std::sin(phip)*std::sin(thetap);
          z_dis = -rad_arr*std::cos(theta_arr)+ radp*std::cos(thetap);

          rad_dis = -(rad_arr - radp*(std::sin(theta_arr)*std::cos(phi_arr - phip)*
		std::sin(thetap)+std::cos(theta_arr)*std::cos(thetap)) );
          theta_dis = (radp*std::cos(theta_arr)*std::cos(phi_arr - phip)*
		std::sin(thetap)-std::sin(theta_arr)*std::cos(thetap));
          phi_dis = -radp*std::sin(phi_arr - phip)*std::sin(thetap);

          distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //distance        = std::sqrt(distance_square);

          if (HillCut_Flag) {
            if (std::sqrt(distance_square)<rcut*Hill_radius) {
               hc = 0.0;
            } else if (std::sqrt(distance_square)>=Hill_radius) {
	       hc = 1.0;
	    } else {
	       hc = SQR(std::sin(0.5*PI/(1.0-rcut)*(std::sqrt(distance_square)/Hill_radius-rcut)));
            }
          } else {
            hc = 1.0;
          }

          if(ThreeD_Force) {
            tmp = tdf1(Rp,distance_square+SQR(rad_sft1));
            //if (distance_square<SQR(rad_sft1))
            //   cout <<"tmp:"<<tmp<<", distance:"<<std::sqrt(distance_square)<<std::endl;
            //acc_r *= tmp;
            //acc_phi *= tmp;
            //acc_z *= tmp;
          } else {
            tmp = 1.0;
          }
            //second order gravity
          if (PlanetaryGravityOrder == 2) {
            Real sec_g = tmp*hc*planet_gm/pow(distance_square+SQR(rad_sft1), 1.5);
            acc_r   += sec_g*rad_dis; // radial acceleration
            acc_theta += sec_g*theta_dis; // polar acceleration
            acc_phi   += sec_g*phi_dis;   // azimuthal acceleartion
          }

          //fourth order gravity
          if (PlanetaryGravityOrder == 4) {
            Real forth_g = tmp*hc*planet_gm*(5.0*SQR(rad_sft1)+2.0*distance_square)/
                                    (2.0*pow(SQR(rad_sft1)+distance_square, 2.5));
            acc_r     += forth_g*rad_dis; // radial acceleration
            acc_theta   += forth_g*theta_dis; // polar acceleration
            acc_phi     += forth_g*phi_dis;   // azimuthal acceleartion
          }

          //sixth order gravity
          if (PlanetaryGravityOrder == 6) {
            Real sixth_g = tmp*hc*planet_gm*(35.0*SQR(SQR(rad_sft1))+28.0*SQR(rad_sft1)*distance_square+
                                    8.0*distance_square)/(8.0*pow(SQR(rad_sft1)+distance_square, 3.5));
            acc_r     += sixth_g*rad_dis; // radial acceleration
            acc_theta   += sixth_g*theta_dis; // polar acceleration
            acc_phi     += sixth_g*phi_dis;   // azimuthal acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = tmp*hc*planet_gm/SQR(radp);
             acc_r   += -temp*(std::cos(phi_arr - phip)*std::sin(theta_arr)*std::sin(thetap)+
			std::cos(theta_arr)*std::cos(thetap));
	     acc_theta += -temp*(std::cos(phi_arr - phip)*std::cos(theta_arr)*std::sin(thetap)-
			std::sin(theta_arr)*std::cos(thetap));
	     acc_phi   += temp*std::sin(phi_arr - phip)*std::sin(thetap);
          }
          }

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
//ramp-up the disk effect using time pmass_incr_time
	  if (time < t0_planet+Pp_time) {
             Real frate = std::sin(time/(2.0*Pp_time)*PI);
             acc_r *= (frate*frate);
             acc_theta *= (frate*frate);
             acc_phi *= (frate*frate);
          }

          Real delta_mom1 = +dt*gas_rho*acc_r;
          Real delta_mom2 = +dt*gas_rho*acc_theta;
          Real delta_mom3 = +dt*gas_rho*acc_phi;

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
	}
	
    }
  }
  }
  return;
}


// Mass Remove within Hill
void PlanetAccretion(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar, const int np) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  //if (pmb->porb->orbital_advection_defined)
  //  phi_planet_move -= Omega0*time;
  Real rad(0.0), phi(0.0), z(0.0), theta(PI/2.0);
  //Real pp_orbit_time = Pp_time;
  //Real e2 = SQR(epsilon);
  //if (time < pp_orbit_time)
  //  GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  //Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  //Real rp = 1.0-ecc*std::sin(time);

  Real phip    = PS[np].getPhi();
  Real radp    = PS[np].getRad();
  Real thetap  = PS[np].getTheta();
  Real gmp    = PS[np].getMass();


  //if (time < Pp_time)
  //  gmp *= SQR(std::sin(0.5*PI*time/Pp_time));

  Real GMp = gm0*gmp;
  Real  Hill_radius = (std::pow(gmp*ONE_3RD, ONE_3RD)*radp);
  Real accretion_rad1 = accretion_radius*Hill_radius; // 0.6 *r_Hill

  Real igm1 = 1.0/(gamma_gas - 1.0);
  //Real inv_rad_soft   = 1.0/rad_soft;
  //Real inv_rad_soft_3 = 1.0/(rad_soft*rad_soft*rad_soft);
  //static Real deltaM = 0.0;
  //static Real deltaMp1 = 0.0;
  //static Real deltaMp2 = 0.0;
  //static Real deltaMp3 = 0.0;
  Real deltaM = 0.0;
  Real deltaMp1 = 0.0;
  Real deltaMp2 = 0.0;
  Real deltaMp3 = 0.0;
  Real distance_temp = 1.0e6;
  Real distance;
  //cout << "Omega0:" << Omega0 << endl;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
//#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        //Real d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
        //Real rad, phi, z;
        //GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        //GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
          GetSphCoord(pmb->pcoord,rad,theta,phi,i,j,k); // convert to Spherical coordinates
        // Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= t0_planet) {
        
          Real x_dis = rad*std::cos(phi)*std::sin(theta) - 
		radp*std::cos(phip)*std::sin(thetap);
          Real y_dis = rad*std::sin(phi)*std::sin(theta) - 
		radp*std::sin(phip)*std::sin(thetap);
          Real z_dis = rad*std::cos(theta) - radp*std::cos(thetap);

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          distance        = std::sqrt(distance_square);

          //cout << "d2: "<< d2 << endl;
          //cout << "distance_square: "<< distance_square << endl;

          //if ((distance > rad_soft) && (distance <= accretion_radius)) {
          //distance_temp = std::min(distance,distance_temp);
          if (distance <= accretion_rad1) {
            //Real time_freefall  = distance_square*distance*inv_sqrt2gmp;
            //Real remove_percent = -rate*std::max(dt/time_freefall, 1.0);
            Real omega = std::sqrt(gm0/radp/radp/radp);
            Real frac_acc   = std::min(accretion_rate*dt*omega, 0.8);
            Real vol = pmb->pcoord->GetCellVolume(k, j, i);

            const Real &gas_rho  = prim(IDN, k, j, i);
            const Real &gas_vel1 = prim(IM1, k, j, i);
            const Real &gas_vel2 = prim(IM2, k, j, i);
            const Real &gas_vel3 = prim(IM3, k, j, i);


            Real &gas_dens = cons(IDN, k, j, i);
            Real &gas_mom1 = cons(IM1, k, j, i);
            Real &gas_mom2 = cons(IM2, k, j, i);
            Real &gas_mom3 = cons(IM3, k, j, i);
            //cout << "... before accretion: gas_dens: " <<  cons(IDN, k, j, i)<< endl;
            //cout << "... before accretion: prim gas_dens: " <<  prim(IDN, k, j, i)<< endl;

            Real delta_gas_dens = frac_acc*gas_rho;
            Real delta_gas_mom1 = delta_gas_dens*gas_vel1;
            Real delta_gas_mom2 = delta_gas_dens*gas_vel2;
            Real delta_gas_mom3 = delta_gas_dens*gas_vel3;

            deltaM   += delta_gas_dens*vol;
            deltaMp1 += delta_gas_mom1*vol;
            deltaMp2 += delta_gas_mom2*vol;
            deltaMp3 += delta_gas_mom3*vol;

            //cout << "deltaM: "<< deltaM << endl;
            //cout << "frac_acc: "<< frac_acc << ", gas_rho: "<< gas_rho << ", time"<< time << endl;
            //if (deltaM <= 0.0)
               //cout << "deltaM: "<< deltaM << endl;
            //   cout << "in loop: deltaM: "<< deltaM << ", distance: " << distance << ", racc: " << racc << endl;

            gas_dens -= delta_gas_dens;
            gas_mom1 -= delta_gas_mom1;
            gas_mom2 -= delta_gas_mom2;
            gas_mom3 -= delta_gas_mom3;

            //cout << "... accretion: delta gas_dens: " << delta_gas_dens << ", frac_acc:" << frac_acc << ", gas_rho:" << gas_rho << endl;
            //cout << "... after accretion: gas_dens: " << cons(IDN, k, j, i) << ", " << prim(IDN, k, j, i) << endl;

            if (NON_BAROTROPIC_EOS) {
              Real &gas_erg  = cons(IEN, k, j, i);
              gas_erg       -= (delta_gas_mom1*gas_vel1 + delta_gas_mom2*gas_vel2
                                                        + delta_gas_mom3*gas_vel3);
            }

          } // if distance
        // } // if distance
        // } // if t0
      }
    }
  }

  return;
 }

Real PlanetAccretionHistory(MeshBlock *pmb,  int np)
//const Real time, const Real dt, const AthenaArray<Real> &prim,
//    AthenaArray<Real> &cons 
 {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  //if (pmb->porb->orbital_advection_defined)
  //  phi_planet_move -= Omega0*time;
  Real rad(0.0), phi(0.0), theta(PI/2.0), z(0.0);
  //Real pp_orbit_time = Pp*2.0*PI;
  Real time = pmb->pmy_mesh->time;
  Real dt = pmb->pmy_mesh->dt;
  //Real e2 = SQR(epsilon);
  //Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  //Real rp = 1.0-ecc*std::sin(time);
  //int np = iout;

  Real phip    = PS[np].getPhi();
  Real radp    = PS[np].getRad();
  Real thetap  = PS[np].getTheta();
  Real gmp    = PS[np].getMass();


  //if (time < Pp_time)
  //  gmp *= SQR(std::sin(0.5*PI*time/Pp_time));

  Real GMp = gm0*gmp;
  Real  Hill_radius = (std::pow(gmp*ONE_3RD, ONE_3RD)*radp);
  Real accretion_rad1 = accretion_radius*Hill_radius; // 0.6 *r_Hill

  AthenaArray<Real> &u = pmb->phydro->u;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  //static Real deltaM = 0.0;
  //static Real deltaMp1 = 0.0;
  //static Real deltaMp2 = 0.0;
  //static Real deltaMp3 = 0.0;
  Real deltaM = 0.0;
  Real deltaMp1 = 0.0;
  Real deltaMp2 = 0.0;
  Real deltaMp3 = 0.0;
  Real distance_temp = 1.0e6;
  Real distance;
  //cout << "Omega0:" << Omega0 << endl;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
//#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        //Real d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
        //Real rad, phi, z;
        //GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        // Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= t0_planet) {
        //GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
          GetSphCoord(pmb->pcoord,rad,theta,phi,i,j,k); // convert to Spherical coordinates


          Real x_dis = rad*std::cos(phi)*std::sin(theta) - 
	  	radp*std::cos(phip)*std::sin(thetap);
          Real y_dis = rad*std::sin(phi)*std::sin(theta) - 
	  	radp*std::sin(phip)*std::sin(thetap);
          Real z_dis = rad*std::cos(theta) - radp*std::cos(thetap);

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          distance        = std::sqrt(distance_square);
          //Real cross1 = std::sin(x2)*std::sin(thetap)*std::cos(x3-phip)+std::cos(x2)*std::cos(thetap);
          //distance = SQR(x1)+SQR(radp)-2.0*rp*x1*cross1;

          //distance        = std::sqrt(d2);
          //distance = std::sqrt(SQR(x_dis) + SQR(y_dis));
          //cout << "d2: "<< d2 << endl;
          //cout << "distance_square: "<< distance_square << endl;

          //if ((distance > rad_soft) && (distance <= accretion_radius)) {
          //distance_temp = std::min(distance,distance_temp);
          if (distance <= accretion_rad1) {
            //Real time_freefall  = distance_square*distance*inv_sqrt2gmp;
            //Real remove_percent = -rate*std::max(dt/time_freefall, 1.0);
            Real omega = std::sqrt(gm0/radp/radp/radp);
            Real frac_acc   = std::min(accretion_rate*dt*omega, 0.8);
            Real vol = pmb->pcoord->GetCellVolume(k, j, i);

            const Real gas_rho = u(IDN,k,j,i);
            const Real gas_mom1 = u(IM1,k,j,i);
            const Real gas_mom2 = u(IM2,k,j,i);
            const Real gas_mom3 = u(IM3,k,j,i);
            const Real gas_vel1 = gas_mom1/gas_rho;
            const Real gas_vel2 = gas_mom2/gas_rho;
            const Real gas_vel3 = gas_mom3/gas_rho;
            //const Real &gas_rho  = prim(IDN, k, j, i);
            //const Real &gas_vel1 = prim(IM1, k, j, i);
            //const Real &gas_vel2 = prim(IM2, k, j, i);
            //const Real &gas_vel3 = prim(IM3, k, j, i);


            //Real &gas_dens = cons(IDN, k, j, i);
            //Real &gas_mom1 = cons(IM1, k, j, i);
            //Real &gas_mom2 = cons(IM2, k, j, i);
            //Real &gas_mom3 = cons(IM3, k, j, i);
            //cout << "... before accretion: gas_dens: " <<  cons(IDN, k, j, i)<< endl;
            //cout << "... before accretion: prim gas_dens: " <<  prim(IDN, k, j, i)<< endl;

            Real delta_gas_dens = frac_acc*gas_rho;
            Real delta_gas_mom1 = delta_gas_dens*gas_vel1;
            Real delta_gas_mom2 = delta_gas_dens*gas_vel2;
            Real delta_gas_mom3 = delta_gas_dens*gas_vel3;

            deltaM   += delta_gas_dens*vol;
            deltaMp1 += delta_gas_mom1*vol;
            deltaMp2 += delta_gas_mom2*vol;
            deltaMp3 += delta_gas_mom3*vol;

            //cout << "deltaM: "<< deltaM << endl;
            //cout << "frac_acc: "<< frac_acc << ", gas_rho: "<< gas_rho << ", time"<< time << endl;
            //if (deltaM <= 0.0)
               //cout << "deltaM: "<< deltaM << endl;
            //   cout << "in loop: deltaM: "<< deltaM << ", distance: " << distance << ", racc: " << racc << endl;


            //cout << "... accretion: delta gas_dens: " << delta_gas_dens << ", frac_acc:" << frac_acc << ", gas_rho:" << gas_rho << endl;
            //cout << "... after accretion: gas_dens: " << cons(IDN, k, j, i) << ", " << prim(IDN, k, j, i) << endl;


          } // if distance
        // } // if distance
        // } // if t0
      }
    }
  }

  //if (deltaM >= 0.0)
  //   cout << "refine: "<< refine << endl;
  //   //cout << "deltaM: "<< deltaM << endl;
  //   cout << "... deltaM: "<< deltaM << ", time: " << time << ", ruser0: " << pmb->ruser_meshblock_data[0](0)  << endl;

  return deltaM;
 }


void PlanetUpdateFromDisk(MeshBlock *pmb, int np)
{
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move    = omega_planet*time + phi_planet_0;

  
  int nc1 = pmb->ncells1;
  //Real acc_x1, acc_x2, acc_x3;

  AthenaArray<Real> &u = pmb->phydro->u;

  
  //cout << "rad_sft1:" << rad_sft1 << ", dt:" << dt << endl;

  //if (pmb->porb->orbital_advection_defined)
  //  phip -= Omega0*time;

  Real planet_gm, cs_planet, gMth, t_growth, t_disapear, distance_square, distance;
  Real x_dis, y_dis, z_dis, rad_dis, phi_dis, theta_dis;
  Real rad_arr, phi_arr, theta_arr, z_arr;
  Real hc, tmp;
  //Real acc_r, acc_phi, acc_z, acc_theta;
  //Real acc_r_temp, acc_phi_temp, acc_z_temp, acc_theta_temp;

  //np = nout/3;

  //Real igm1 = 1.0/(gamma_gas - 1.0);
  //AthenaArray<Real> &vs = pmb->pmy_mesh->ruser_mesh_data[0];
/*  
     Real &acc_r = pmb->pmy_mesh->ruser_mesh_data[1](3*np+0); 
     Real &acc_phi = pmb->pmy_mesh->ruser_mesh_data[1](3*np+1); 
     acc_r = 0.0;
     acc_phi = 0.0;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
     Real &acc_z = pmb->pmy_mesh->ruser_mesh_data[1](3*np+2); 
     acc_z = 0.0;
     //std::cout << "Rad_p:" << pp(4*n+0) << "\n" <<std::endl;
   } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
     Real &acc_theta = pmb->pmy_mesh->ruser_mesh_data[1](3*np+2); 
     acc_theta = 0.0;
   }
*/
if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
     Real &acc_r   = pmb->pmy_mesh->ruser_mesh_data[1](3*np+0); 
     Real &acc_phi = pmb->pmy_mesh->ruser_mesh_data[1](3*np+1); 
     Real &acc_z   = pmb->pmy_mesh->ruser_mesh_data[1](3*np+2); 
     acc_r   = 0.0;
     acc_phi = 0.0;
     acc_z   = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
//#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //Real rad_arr, phi_arr, z_arr;
  	const Real rho_gas = u(IDN,k,j,i);
        Real vol = pmb->pcoord->GetCellVolume(k, j, i);
        Real dm_cell = rho_gas*vol;
        //cout << "dm_cell:" << dm_cell << "rho_gas" << rho_gas << "vol:" <<vol << endl;

        GetCylCoord(pmb->pcoord, rad_arr, phi_arr, z_arr, i, j, k);

	//for (int np=0;np<nPlanet;np++) {
          Real phip    = PS[np].getPhi();
          Real radp    = PS[np].getRad();
          Real thetap      = PS[np].getTheta();
          Real zp = radp*std::cos(thetap);
          Real Rp = radp*std::sin(thetap);
          if (phip>TWO_PI)
            phip += -TWO_PI;
          else if (phip<0.0)
            phip += TWO_PI;
          planet_gm = PS[np].getMass();
          //Real gmp    = PS[np].getMass();
          Real  Hill_radius = (std::pow(planet_gm/gm0*ONE_3RD, ONE_3RD)*radp);
          Real rad_sft1 = rad_soft*Hill_radius; // 0.6 *r_Hill

          x_dis = -rad_arr*std::cos(phi_arr) + Rp*std::cos(phip);
          y_dis = -rad_arr*std::sin(phi_arr) + Rp*std::sin(phip);

          rad_dis = -(radp - rad_arr*cos(phi_arr - phip));
          phi_dis = rad_arr*sin(phi_arr - phip);
          z_dis = -(zp - z_arr); // need to update

          distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //distance        = std::sqrt(distance_square);

          if (HillCut_Flag) {
            if (std::sqrt(distance_square)<rcut*Hill_radius) {
               hc = 0.0;
            } else if (std::sqrt(distance_square)>=Hill_radius) {
	       hc = 1.0;
	    } else {
	       hc = SQR(std::sin(0.5*PI/(1.0-rcut)*(std::sqrt(distance_square)/Hill_radius-rcut)));
            }
          } else {
            hc = 1.0;
          }

          if(ThreeD_Force) {
            tmp = tdf1(Rp,distance_square+SQR(rad_sft1));
            //if (distance_square<SQR(rad_sft1))
            //   cout <<"tmp:"<<tmp<<", distance:"<<std::sqrt(distance_square)<<std::endl;
            //acc_r *= tmp;
            //acc_phi *= tmp;
            //acc_z *= tmp;
          } else {
            tmp = 1.0;
          }

            //second order gravity
          if (PlanetaryGravityOrder == 2) {
            Real sec_g = tmp*hc*dm_cell/pow(distance_square+SQR(rad_sft1), 1.5);
            acc_r   += sec_g*rad_dis; // radial acceleration
            acc_phi += sec_g*phi_dis; // azimuthal acceleration
            //acc_z   += sec_g*z_dis;   // vertical acceleartion
            acc_z   += 0.0;   // vertical acceleartion
          }

          //fourth order gravity
          if (PlanetaryGravityOrder == 4) {
            Real forth_g = tmp*hc*dm_cell*(5.0*SQR(rad_sft1)+2.0*distance_square)/
                                    (2.0*pow(SQR(rad_sft1)+distance_square, 2.5));
            acc_r     += forth_g*rad_dis; // radial acceleration
            acc_phi   += forth_g*phi_dis; // azimuthal acceleration
            //acc_z     += forth_g*z_dis;   // vertical acceleartion
            acc_z     += 0.0;   // vertical acceleartion
          }

          //sixth order gravity
          if (PlanetaryGravityOrder == 6) {
            Real sixth_g = tmp*hc*dm_cell*(35.0*SQR(SQR(rad_sft1))+28.0*SQR(rad_sft1)*distance_square+
                                    8.0*distance_square)/(8.0*pow(SQR(rad_sft1)+distance_square, 3.5));
            acc_r     += sixth_g*rad_dis; // radial acceleration
            acc_phi   += sixth_g*phi_dis; // azimuthal acceleration
            //acc_z     += sixth_g*z_dis;   // vertical acceleartion
            acc_z     += 0.0;   // vertical acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = tmp*hc*dm_cell/SQR(rad_arr);
             acc_r   += -temp*std::cos(phi_arr - phip);
	     acc_phi += -temp*std::sin(phi_arr - phip);
	     //acc_z   += -temp*zp/Rp;
	     acc_z   += 0.0;
          }
         // }

        }
       }
      }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
     //if (HAF_DISK==1){
     //	Real &acc_r = 2.0*pmb->pmy_mesh->ruser_mesh_data[1](3*np+0); 
     //	Real &acc_phi = 2*pmb->pmy_mesh->ruser_mesh_data[1](3*np+1); 
     //	Real &acc_theta = 2*pmb->pmy_mesh->ruser_mesh_data[1](3*np+2); 
     //} else {
	Real &acc_r     = pmb->pmy_mesh->ruser_mesh_data[1](3*np+0); 
     	Real &acc_phi   = pmb->pmy_mesh->ruser_mesh_data[1](3*np+1); 
     	Real &acc_theta = pmb->pmy_mesh->ruser_mesh_data[1](3*np+2); 
     //}
     acc_r     = 0.0;
     acc_phi   = 0.0;
     acc_theta = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
//#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
			//Real rad_arr, theta_arr, phi_arr;
        //acc_r = 0.0;
        //acc_phi = 0.0;
        //acc_theta = 0.0;
  	const Real rho_gas = u(IDN,k,j,i);
        Real vol = pmb->pcoord->GetCellVolume(k, j, i);
        Real dm_cell = rho_gas*vol;
        GetSphCoord(pmb->pcoord, rad_arr, theta_arr, phi_arr, i, j, k);

	//for (int np=0;np<nPlanet;np++) {
          Real phip    = PS[np].getPhi();
          Real radp    = PS[np].getRad();
          Real thetap  = PS[np].getTheta();
          Real Rp = radp*std::sin(thetap);
          //Real zp      = PS[np].getZ();
          planet_gm = PS[np].getMass();
          if (phip>TWO_PI)
            phip += -TWO_PI;
          else if (phip<0.0)
            phip += TWO_PI;
          //Real gmp    = PS[np].getMass();
          Real  Hill_radius = (std::pow(planet_gm/gm0*ONE_3RD, ONE_3RD)*radp);
          Real rad_sft1 = rad_soft*Hill_radius; // 0.6 *r_Hill

          x_dis = -rad_arr*std::cos(phi_arr)*std::sin(theta_arr) + 
		  radp*std::cos(phip)*std::sin(thetap);
          y_dis = -rad_arr*std::sin(phi_arr)*std::sin(theta_arr) + 
		  radp*std::sin(phip)*std::sin(thetap);
          z_dis = -rad_arr*std::cos(theta_arr)+ radp*std::cos(thetap);

          rad_dis = -(radp - rad_arr*(std::sin(theta_arr)*std::cos(phi_arr - phip)*
		std::sin(thetap)+std::cos(theta_arr)*std::cos(thetap)) );
          theta_dis = (rad_arr*std::cos(thetap)*std::cos(phi_arr - phip)*
		std::sin(theta_arr)-std::sin(thetap)*std::cos(theta_arr)); // may need to update
          phi_dis = rad_arr*std::sin(phi_arr - phip)*std::sin(theta_arr);

          distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //distance        = std::sqrt(distance_square);
          

          if (HillCut_Flag) {
            if (std::sqrt(distance_square)<rcut*Hill_radius) {
               hc = 0.0;
            } else if (std::sqrt(distance_square)>=Hill_radius) {
	       hc = 1.0;
	    } else {
	       hc = SQR(std::sin(0.5*PI/(1.0-rcut)*(std::sqrt(distance_square)/Hill_radius-rcut)));
            }
          } else {
            hc = 1.0;
          }

          if(ThreeD_Force) {
            tmp = tdf1(Rp,distance_square+SQR(rad_sft1));
            //if (distance_square<SQR(rad_sft1))
            //   cout <<"tmp:"<<tmp<<", distance:"<<std::sqrt(distance_square)<<std::endl;
            //acc_r *= tmp;
            //acc_phi *= tmp;
            //acc_z *= tmp;
          } else {
            tmp = 1.0;
          }

            //second order gravity
          if (PlanetaryGravityOrder == 2) {
            Real sec_g = tmp*hc*dm_cell/pow(distance_square+SQR(rad_sft1), 1.5);
            acc_r   += sec_g*rad_dis; // radial acceleration
            acc_theta += 0.0; // polar acceleration
            acc_phi   += sec_g*phi_dis;   // azimuthal acceleartion
          }

          //fourth order gravity
          if (PlanetaryGravityOrder == 4) {
            Real forth_g = tmp*hc*dm_cell*(5.0*SQR(rad_sft1)+2.0*distance_square)/
                                    (2.0*pow(SQR(rad_sft1)+distance_square, 2.5));
            acc_r     += forth_g*rad_dis; // radial acceleration
            acc_theta   += 0.0; // polar acceleration
            acc_phi     += forth_g*phi_dis;   // azimuthal acceleartion
          }

          //sixth order gravity
          if (PlanetaryGravityOrder == 6) {
            Real sixth_g = tmp*hc*dm_cell*(35.0*SQR(SQR(rad_sft1))+28.0*SQR(rad_sft1)*distance_square+
                                    8.0*distance_square)/(8.0*pow(SQR(rad_sft1)+distance_square, 3.5));
            acc_r     += sixth_g*rad_dis; // radial acceleration
            acc_theta   += 0.0; // polar acceleration
            acc_phi     += sixth_g*phi_dis;   // azimuthal acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = tmp*hc*dm_cell/SQR(rad_arr);
             acc_r   += -temp*(std::cos(phi_arr - phip)*std::sin(theta_arr)*std::sin(thetap)+
			std::cos(theta_arr)*std::cos(thetap));
	     //acc_theta += -temp*(std::cos(phi_arr - phip)*std::cos(theta_arr)*std::sin(thetap)-
             //	         std::sin(theta_arr)*std::cos(thetap));
	     acc_theta += 0.0;
	     acc_phi   += -temp*std::sin(phi_arr - phip)*std::sin(theta_arr);
          }
          //} // nPlanet

	}
	
    }
   }
  }
  //MPI_Allreduce(term, sum, 3, MY_MPI_REAL,
  //              MPI_SUM, MPI_COMM_WORLD);
  //ruser_mesh_data[1][0] = acc_r;
  //ruser_mesh_data[1][1] = acc_phi;
  //ruser_mesh_data[1][2] = acc_theta;

     
  
  /*
  AthenaArray<Real> term;
  term.NewAthenaArray(3);
  term(0) = acc_r;
  term(1) = acc_phi;
  term(2) = 0.0;
  */
//#ifdef MPI_PARALLEL
//    MPI_Allreduce(MPI_IN_PLACE, term.data(), 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
//#endif

   //PS[np].setFr(term(0));
   //PS[np].setFp(term(1));
   //PS[np].setFt(term(2));

   //cout << "PlanetUpdate2:acc_r:" << PS[np].getFr() << "acc_phi:" << PS[np].getFp() << endl;
  /*
   PS[np].fr = term(0);
   PS[np].fp = term(1);
   PS[np].ft = term(2);
  */
  /*
  if (nout%3==0) {
       return acc_r;
  } else if (nout%3==1) {
    if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
       return acc_phi;
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
       return acc_phi;
    }
  } else if (nout%3==2) {
    if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
       return acc_z;
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
       return acc_theta;
    }
  }
  */
}

// Mass Remove within Hill
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar, const int np) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  Real phip    = PS[np].getPhi();
  Real radp    = PS[np].getRad();
  Real thetap  = PS[np].getTheta();
  Real gmp    = PS[np].getMass();

  Real  Hill_radius = (std::pow(gmp*ONE_3RD, ONE_3RD)*radp);
  Real accretion_rad1 = accretion_radius*Hill_radius; // 0.6 *r_Hill
  Real rad_sft1 = rad_soft*Hill_radius; // 0.6 *r_Hill

  if (pmb->porb->orbital_advection_defined)
    phip -= Omega0*time;
  if (phip>TWO_PI)
    phip += -TWO_PI;
  else if (phip<0.0)
    phip += TWO_PI;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_rad_soft   = 1.0/rad_sft1;
  Real inv_rad_soft_3 = 1.0/(rad_sft1*rad_sft1*rad_sft1);

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real rad, phi, theta;
        //GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        GetSphCoord(pmb->pcoord, rad, theta, phi, i, j, k);
        //Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= (t0_planet + t_growth)) 
        if (time >= t0_planet) {

          Real x_dis = rad*std::cos(phi)*std::sin(theta) - 
		radp*std::cos(phip)*std::sin(thetap);
          Real y_dis = rad*std::sin(phi)*std::sin(theta) - 
		radp*std::sin(phip)*std::sin(thetap);
          Real z_dis = rad*std::cos(theta) - radp*std::cos(thetap);

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          Real distance        = std::sqrt(distance_square);

          if ((distance > rad_sft1) && (distance <= accretion_rad1)) {
            Real time_freefall  = std::sqrt(distance_square*distance)*inv_sqrt2gmp;
            Real remove_percent = -accretion_rate*std::max(dt/time_freefall, 1.0);

            const Real &gas_rho  = prim(IDN, k, j, i);
            const Real &gas_vel1 = prim(IM1, k, j, i);
            const Real &gas_vel2 = prim(IM2, k, j, i);
            const Real &gas_vel3 = prim(IM3, k, j, i);

            Real &gas_dens = cons(IDN, k, j, i);
            Real &gas_mom1 = cons(IM1, k, j, i);
            Real &gas_mom2 = cons(IM2, k, j, i);
            Real &gas_mom3 = cons(IM3, k, j, i);


          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------

// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  Real Vel_K, omega_dyn, R_func, damping_tau;

  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  //AthenaArray<Real> Vel_K, omega_dyn, R_func, damping_tau;
  //Vel_K.NewAthenaArray(nc1);
  //omega_dyn.NewAthenaArray(nc1);
  //R_func.NewAthenaArray(nc1);
  //damping_tau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          //std::cout << "Inner_damping, radius_inner_damping:" << radius_inner_damping << std::endl;
          if (rad < radius_inner_damping) {
            //std::cout << "radius_inner_damping:" << radius_inner_damping << std::endl;
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
            //R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            Real xin = (rad - x1min)/inner_width_damping;
            if (xin<=0.5 && xin>=0.0) {
               R_func      =  1.0 - 6.0*xin*xin + 6.0*xin*xin*xin;
            } else if (xin>0.5 && xin<= 1.0) {
               R_func      =  2.0*(1.0-xin)*(1.0-xin)*(1.0-xin);
            }
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
              //gas_pre     = cons(IPR, k, j, i);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func/damping_tau*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func/damping_tau*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func/damping_tau*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func/damping_tau*dt;

            //std::cout << "damping_tau: "<<damping_tau<<", gas_vel1_0: "<< gas_vel1_0 << std::endl;
            //gas_dens += delta_gas_dens; // do not damp density
            gas_vel1 += delta_gas_vel1;
            //gas_vel2 += delta_gas_vel2;
            //gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            //gas_mom2 = gas_dens*gas_vel2; // do not damp vphi or vz
            //gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func/damping_tau*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_vel1) + SQR(gas_vel2)
              //                                   + SQR(gas_vel3))*gas_rho;
            }
          }
        }

      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad >= x1min && rad < radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
            //R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            Real xin = (rad - x1min)/inner_width_damping;
            if (xin<=0.5 && xin>=0.0) {
               R_func      =  1.0 - 6.0*xin*xin + 6.0*xin*xin*xin;
            } else if (xin>0.5 && xin<= 1.0) {
               R_func      =  2.0*(1.0-xin)*(1.0-xin)*(1.0-xin);
            }
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
              //gas_pre     = prim(IPR, k, j, i);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func/damping_tau*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func/damping_tau*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func/damping_tau*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func/damping_tau*dt;

            //gas_dens += delta_gas_dens; // do not damp density
            gas_vel1 += delta_gas_vel1;
            // gas_vel2 += delta_gas_vel2; // do not damp vtheta or vphi
            // gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            // gas_mom2 = gas_dens*gas_vel2;
            // gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func/damping_tau*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_vel1) + SQR(gas_vel2)
              //                                   + SQR(gas_vel3))*gas_rho;
            }
          }
        }

      }
    }
  }
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;
  LogicalLocation &loc = pmb->loc;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  //Real gas_vel1_0  = 0.0;
  Real Vel_K, omega_dyn, R_func, damping_tau;
  Real number_glob, vr_sum_glob;

  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  //AthenaArray<Real> Vel_K, omega_dyn, R_func, damping_tau;
  //Vel_K.NewAthenaArray(nc1);
  //omega_dyn.NewAthenaArray(nc1);
  //R_func.NewAthenaArray(nc1);
  //damping_tau.NewAthenaArray(nc1);
  
         //std::cout << "gas_vel1_0: "<<gas_vel1_0 << std::endl;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
         for (int k=ks; k<=ke; ++k) {
          Real x3 = pmb->pcoord->x3v(k);

          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

            //std::cout << "out of if: rad: "<<rad<<", radius_outer_damping: "<< radius_outer_damping << std::endl;
          if (rad >= radius_outer_damping) {
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-pmb->is);
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js);
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            //std::cout << "lx1: "<<loc.lx1<<", nx1: "<< pmb->block_size.nx1 << ",is: "<<is << std::endl;
            //std::cout << "ti: "<<ti<<", i: "<< i << ",is: "<<is << std::endl;
            //std::cout << "rad: "<<rad<<", radius_outer_damping: "<< radius_outer_damping << std::endl;
  //if (Globals::my_rank == 0) {
            //std::cout << "rad: "<<rad<<", radius_outer_damping: "<< radius_outer_damping << std::endl;
            //std::cout << "damping_tau: "<<damping_tau<<", gas_vel1_0: "<< gas_vel1_0 << std::endl;
  //}
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
            //R_func      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            Real xout = 1.0-(rad - radius_outer_damping)/outer_width_damping;
            if (xout<=0.5 && xout>=0.0) {
               R_func      =  1.0 - 6.0*xout*xout + 6.0*xout*xout*xout;
            } else if (xout>0.5 && xout<= 1.0) {
               R_func      =  2.0*(1.0-xout)*(1.0-xout)*(1.0-xout);
            }
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

            //Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);


            //Real gas_vel1_0 = vis_vel_r;
            //Real gas_vel1_0 = vr_out;
            Real vr0 = pmb->pmy_mesh->ruser_mesh_data[3](tk,ti);
            Real num0 = pmb->pmy_mesh->ruser_mesh_data[4](tk,ti);
            Real dens0 = pmb->pmy_mesh->ruser_mesh_data[5](tk,ti);
            Real gas_rho_0 = dens0/num0;
            Real gas_vel1_0 = vr0/num0;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) { // need correction, use prim!!
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
              //gas_pre           = prim(IPR, k, j, i);
            }

            Real gas_vel1_cons = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;


            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func/damping_tau*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func/damping_tau*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func/damping_tau*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func/damping_tau*dt;

            //gas_dens += delta_gas_dens; // only damp vr, not density or other velocity
            Real gas_vel1_update = gas_vel1 + delta_gas_vel1;
            gas_vel1 += delta_gas_vel1;
            //gas_vel2 += delta_gas_vel2;
            //gas_vel3 += delta_gas_vel3;
            
            //gas_mom1  += gas_rho*delta_gas_vel1;
            gas_mom1 = gas_rho*gas_vel1;
            //gas_mom1 = gas_dens*gas_vel1;
            //gas_mom2 = gas_dens*gas_vel2;
            //gas_mom3 = gas_dens*gas_vel3;
            
           /* 
	    if ((i==ie) and (j==js) and (k==ks)) {
            Real mdot = 2.0*PI*gas_rho*gas_vel1_update*rad;
            Real mdot0 = 2.0*PI*gas_rho_0*vis_vel_r*rad;
            std::cout << "damping rad: "<<rad<<", phi: "<< phi << std::endl;
            std::cout << "damping before gas_vel1: "<<gas_vel1<<", after gas_vel1: "<<gas_vel1_update<< 
		", gas_vel1_cons: "<<gas_vel1_cons<<
		", gas_vel1_0: "<< gas_vel1_0 << std::endl;
            std::cout << "damping gas_den: "<<gas_rho<<", gas_dens(cons): "<< gas_dens << std::endl;
            std::cout << "damping mdot0: "  << mdot0 <<", mdot: "<< mdot << std::endl;
            }
          */
	    
	   
	    

            //gas_mom1 = gas_dens*gas_vel1;
            //gas_mom2 = gas_dens*gas_vel2;
            //gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func/damping_tau*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_vel1) + SQR(gas_vel2)
              //                                        + SQR(gas_vel3))*gas_rho;
            }
          }
        }
       }
      }

    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
         for (int k=ks; k<=ke; ++k) {
          Real x3 = pmb->pcoord->x3v(k);

          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad <= x1max && rad >= radius_outer_damping) {
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-pmb->is);
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js);
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
            //R_func      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            Real xout = 1.0-(rad - radius_outer_damping)/outer_width_damping;
            if (xout<=0.5 && xout>=0.0) {
               R_func      =  1.0 - 6.0*xout*xout + 6.0*xout*xout*xout;
            } else if (xout>0.5 && xout<= 1.0) {
               R_func      =  2.0*(1.0-xout)*(1.0-xout)*(1.0-xout);
            }
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            //Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

            //Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);

            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            //Real gas_vel1_0 = vis_vel_r;
            //Real gas_vel1_0 = vr_out;
            Real vr0 = pmb->pmy_mesh->ruser_mesh_data[3](tj,ti);
            Real num0 = pmb->pmy_mesh->ruser_mesh_data[4](tj,ti);
            Real dens0 = pmb->pmy_mesh->ruser_mesh_data[5](tj,ti);
            Real gas_rho_0 = dens0/num0;
            Real gas_vel1_0 = vr0/num0;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
              //gas_pre           = prim(IPR, k, j, i);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func/damping_tau*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func/damping_tau*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func/damping_tau*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func/damping_tau*dt;

            // gas_dens += delta_gas_dens; // only damp vr, not density or other velocity
            gas_vel1 += delta_gas_vel1;
            
            Real gas_vel1_update = gas_vel1 + delta_gas_vel1;
            // gas_vel2 += delta_gas_vel2;
            // gas_vel3 += delta_gas_vel3;

            //gas_mom1 = gas_rho*gas_vel1;
            gas_mom1 = gas_rho*gas_vel1;
            // gas_mom2 = gas_dens*gas_vel2;
            // gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func/damping_tau*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_vel1) + SQR(gas_vel2)
              //                                        + SQR(gas_vel3))*gas_rho;
            }
          }
        }

      }
    }
  }
  return;
}


void InnerWavedamping2(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, damping_invtau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  damping_invtau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad >= x1min && rad < radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)   = std::sqrt(gm0/(rad*rad*rad));
            //R_func(i)      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            Real xin = (rad - x1min)/inner_width_damping;
            if (xin<=0.5 && xin>=0.0) {
               R_func(i)      =  1.0 - 6.0*xin*xin + 6.0*xin*xin*xin;
            } else if (xin>0.5 && xin<= 1.0) {
               R_func(i)      =  2.0*(1.0-xin)*(1.0-xin)*(1.0-xin);
            }
            damping_invtau(i) = damping_rate*omega_dyn(i);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn(i)/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            gas_mom2 = gas_rho*gas_vel2;
            gas_mom3 = gas_rho*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*damping_invtau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad >= x1min && rad < radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)   = std::sqrt(gm0/(rad*rad*rad));
            //R_func(i)      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            Real xin = (rad - x1min)/inner_width_damping;
            if (xin<=0.5 && xin>=0.0) {
               R_func(i)      =  1.0 - 6.0*xin*xin + 6.0*xin*xin*xin;
            } else if (xin>0.5 && xin<= 1.0) {
               R_func(i)      =  2.0*(1.0-xin)*(1.0-xin)*(1.0-xin);
            }
            damping_invtau(i) = damping_rate*omega_dyn(i);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn(i)/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            gas_mom2 = gas_rho*gas_vel2;
            gas_mom3 = gas_rho*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*damping_invtau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

      }
    }
  }
  return;
}


void OuterWavedamping2(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, damping_invtau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  damping_invtau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad <= x1max && rad >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)   = std::sqrt(gm0/(rad*rad*rad));
            //R_func(i)      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            Real xout = 1.0-(rad - radius_outer_damping)/outer_width_damping;
            if (xout<=0.5 && xout>=0.0) {
               R_func(i)      =  1.0 - 6.0*xout*xout + 6.0*xout*xout*xout;
            } else if (xout>0.5 && xout<= 1.0) {
               R_func(i)      =  2.0*(1.0-xout)*(1.0-xout)*(1.0-xout);
            }
            damping_invtau(i) = damping_rate*omega_dyn(i);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn(i)/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            gas_mom2 = gas_rho*gas_vel2;
            gas_mom3 = gas_rho*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*damping_invtau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad <= x1max && rad >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)   = std::sqrt(gm0/(rad*rad*rad));
            //R_func(i)      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            Real xout = 1.0-(rad - radius_outer_damping)/outer_width_damping;
            if (xout<=0.5 && xout>=0.0) {
               R_func(i)      =  1.0 - 6.0*xout*xout + 6.0*xout*xout*xout;
            } else if (xout>0.5 && xout<= 1.0) {
               R_func(i)      =  2.0*(1.0-xout)*(1.0-xout)*(1.0-xout);
            }
            damping_invtau(i) = damping_rate*omega_dyn(i);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn(i)/rad);

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            Real gas_rho     = prim(IDN, k, j, i);
            Real gas_vel1    = prim(IM1, k, j, i);
            Real gas_vel2    = prim(IM2, k, j, i);
            Real gas_vel3    = prim(IM3, k, j, i);

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            //Real gas_vel1 = gas_mom1*inv_dens_gas;
            //Real gas_vel2 = gas_mom2*inv_dens_gas;
            //Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_rho)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_rho*gas_vel1;
            gas_mom2 = gas_rho*gas_vel2;
            gas_mom3 = gas_rho*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*damping_invtau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

      }
    }
  }
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  // Local Isothermal equation of state
  Real rad, phi, z;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho  = prim(IDN, k, j, i);
        const Real &gas_vel1 = prim(IM1, k, j, i);
        const Real &gas_vel2 = prim(IM2, k, j, i);
        const Real &gas_vel3 = prim(IM3, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        Real press        = PoverRho(rad, phi, z)*gas_dens;
        gas_erg           = press*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  Real rad, phi, z;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta  = 1.0/beta;
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real omega_dyn      = std::sqrt(gm0/(rad*rad*rad));
        Real inv_t_cool     = omega_dyn*inv_beta;
        Real cs_square_init = PoverRho(rad, phi, z);

        Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*inv_t_cool*dt;
        gas_erg        -= delta_erg;
      }
    }
  }
  return;
}


void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {

  Real inv_beta = 1.0/beta;
  Real igm1     = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_pre = w(IPR, k, j, i);

        Real inv_omega_dyn   = std::sqrt((rad*rad*rad)/gm0);
        Real internal_erg    = gas_rho*gas_pre*igm1;
        Real kappa_radiative = internal_erg*inv_omega_dyn*inv_beta;

        phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i) = kappa_radiative;

        //phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) = 1e-4;
      }
    }
  }
  return;
}



void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}


void GetSphCoord(Coordinates *pco,Real &rad,Real &theta,Real &phi,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=std::sqrt(pco->x1v(i)*pco->x1v(i)+pco->x3v(k)*pco->x3v(k));
    theta=std::atan2(pco->x1v(i),pco->x3v(k));
    phi=pco->x2v(j);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad   = pco->x1v(i);
    theta = pco->x2v(j);
    phi   = pco->x3v(k);
  }
  return;
}


//#ifdef THREED_FORCE
Real tdf1(const Real rad_planet, const Real d2) {
  //Real H = H_disk(r0);
  //Real cs_square = PoverRho(rad, phi, z);
  if (NON_BAROTROPIC_EOS)
    cs_planet = std::sqrt(p0_over_r0*std::pow(rad_planet/r0, pslope));
  else
    cs_planet = std::sqrt(p0_over_r0);

  Real omega_dyn = std::sqrt(gm0/(rad_planet*rad_planet*rad_planet));
  Real H = cs_planet/omega_dyn;
  Real twoH = 2.0*H*H;

  if (0.25*d2/(H*H) > 32.0) return 1.0;

  Real dz, aa, z[8], f[8], tdf0, z2, d2s;
  const Real w[]={3.62683783378361983e-01, 3.13706645877887287e-01,
		    2.22381034453374471e-01, 1.01228536290376259e-01};
  const Real x[]={1.83434642495649805e-01, 5.25532409916328986e-01,
		    7.96666477413626740e-01, 9.60289856497536232e-01};
  dz = 3.0*H;
  aa = dz;
  z[0] = aa - x[0]*dz;
  z[1] = aa + x[0]*dz;
  z[2] = aa - x[1]*dz;
  z[3] = aa + x[1]*dz;
  z[4] = aa - x[2]*dz;
  z[5] = aa + x[2]*dz;
  z[6] = aa - x[3]*dz;
  z[7] = aa + x[3]*dz;

  for(int i=0;i<8;i++) {
    z2 = z[i]*z[i];
    d2s = d2 + z2;
    f[i] = std::exp(-z2/twoH)/(d2s*std::sqrt(d2s));
  }

  tdf0 = dz*(w[0]*(f[0]+f[1])+w[1]*(f[2]+f[3])+
	     w[2]*(f[4]+f[5])+w[3]*(f[6]+f[7]));

  // normalization by 2D force
     return (2.0*tdf0/(sqrt(2.0*PI)*H)*d2*sqrt(d2));
  
}
//#endif

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}

/*
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dffloor);
}
*/

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_gap(const Real rad, const Real phi, const Real z, const Real diff) {
  Real vel = std::sqrt(gm0/rad + diff) - rad*Omega0;
  return vel;
}

/*
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}
*/

Real UserOrbitalVelocity(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return std::sqrt(porb->gm/x1)-porb->Omega0*x1;
}

Real UserOrbitalVelocity_r(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return -0.5*std::sqrt(porb->gm/x1)/x1-porb->Omega0;
}

Real UserOrbitalVelocity_z(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return 0.0;
}

int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;
  Real d2;
  Real time = pmb->pmy_mesh->time;
  // planet positions
  //Real px1 = 1.0-ecc*std::sin(time);
  //Real px2 = PI/2.0;
  //Real px3 = fmod(time*(1.0-Omega0)-2.0*ecc*std::cos(time),2.0*PI);
  //Real px1 = 1.0-ecc*std::cos(time);
  //Real px2 = PI/2.0  - inc*std::cos(time);
  //Real px3 = fmod(time*(1.0-Omega0)+2.0*ecc*std::sin(time),2.0*PI);
  Real px1 = PS[0].getRad();
  Real px2 = PS[0].getTheta();
  Real px3 = PS[0].getPhi();
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    px2 = px3;
    px3 = 0.0; 
  }
  // differential arclengths for each cell
  Real ds1, ds2, ds3;
  // projected distance between cell and planet
  Real dp1, dp2, dp3;
  // coodinates and differentials for each cell
  Real x1, x2, x3;
  Real dx1, dx2, dx3;
  for(int k=pmb->ks; k<=pmb->ke; ++k) {
    x3 = pmb->pcoord->x3v(k);
    dx3 = pmb->pcoord->dx3f(k);
    // special behavior for discontinuity at phi=0 in spherical
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
      x3 = fmod(x3,2.0*PI);
      //if (x3<0.0)
      //   x3 += 2*PI;
      if ((px3 < 0.5*PI) && (x3 > 1.5*PI))
	x3 -= 2.0*PI;
      if ((px3 > 1.5*PI) && (x3 < 0.5*PI))
	x3 += 2.0*PI;
    }
    for(int j=pmb->js; j<=pmb->je; ++j) {
      x2 = pmb->pcoord->x2v(j);
      dx2 = pmb->pcoord->dx2f(j);
      // special behavior for discontinuity at phi=0 in cylindrical
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
        x2 = fmod(x2,2.0*PI);
        //if (x2<0.0)
        //   x2 += 2*PI;
	if ((px2 < 0.5*PI) && (x2 > 1.5*PI))
	  x2 -= 2.0*PI;
	if ((px2 > 1.5*PI) && (x2 < 0.5*PI))
	  x2 += 2.0*PI;
      }
      for(int i=pmb->is; i<=pmb->ie; ++i) {
        x1 = pmb->pcoord->x1v(i);
        dx1 = pmb->pcoord->dx1f(i);
        ds1 = dx1;
        dp1 = x1-px1;
        ds2 = x1*dx2;
        dp2 = px1*(x2-px2);
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
          ds3 = dx3;
          dp3 = x3-px3;
        }
        else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
          ds3 = x1*std::sin(x2)*dx3;
          dp3 = px1*std::sin(px2)*(x3-px3);
        }
        // if cell size > refine area, refine cells closest to planet
        if ((ds1 > area) || (ds2 > area) || (ds3 > area)){
          if(std::fabs(x1-px1) < dx1){
            if(std::fabs(x2-px2) < dx2){
              if(std::fabs(x3-px3) < dx3){
                refine = 1;
              }
            }
          }
        }
        // if projected distance from cell to planet < area
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
            d2 = SQR(x1)+SQR(px1)-2.0*px1*x1*std::cos(x2-px2) + SQR(x3); // need to confirm
        }
        else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
       	    Real cross1 = std::sin(x2)*std::sin(px2)*std::cos(x3-px3)+std::cos(x2)*std::cos(px2);
            d2 = SQR(x1)+SQR(px1)-2.0*px1*x1*cross1;
            //Real d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
            //d2 = SQR(x1)+SQR(px1)-2.0*px1*x1*std::sin(x2)*std::cos(x3-px3);
        }
        Real distance = std::sqrt(d2);
         
        if (distance < area) {
           if(ds1 > res)
             refine = 1;
           if(ds2 > res)
             refine = 1;
           if(ds3 > res)
             refine = 1;
        }
/*
        if(std::fabs(dp1) < area){
          if(std::fabs(dp2) < area){
            if(std::fabs(dp3) < area){
              // and cells are too coarse (> res), then refine
              if(ds1 > res)
                refine = 1;
              if(ds2 > res)
                refine = 1;
              if(ds3 > res)
                refine = 1;
            }
          }
        }
*/
      }
    }
  }
  if(refine){
    return 1;
  }
  else{
    return -1;
  }
}


int RefinementCondition0(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  //AthenaArray<Real> &df_prim = pmb->pdustfluids->df_prim;
  Real maxeps = 0.0;
  Real rad(0.0), phi(0.0), z(0.0);
  Real max_rad = 0.0;
  Real min_rad = 3.0;
  int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; j++) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
      Real epsr_g = (std::abs(w(IDN,k,j,i+1) - 2.0*w(IDN,k,j,i) + w(IDN,k,j,i-1))
                   + std::abs(w(IDN,k,j+1,i) - 2.0*w(IDN,k,j,i) + w(IDN,k,j-1,i)))/w(IDN,k,j,i);

      Real epsp = (std::abs(w(IPR,k,j,i+1) - 2.0*w(IPR,k,j,i) + w(IPR,k,j,i-1))
                 + std::abs(w(IPR,k,j+1,i) - 2.0*w(IPR,k,j,i) + w(IPR,k,j-1,i)))/w(IPR,k,j,i);

      Real epsr_d = 0;

      Real eps = std::max(std::max(epsr_g, epsr_d), epsp);
      maxeps   = std::max(maxeps, eps);
      max_rad  = std::max(max_rad, rad);
      min_rad  = std::min(min_rad, rad);
    }
  }
  // refine : curvature > 0.01
  if ((max_rad >= 0.3) && ( min_rad <= 2.9 ) && (maxeps > 0.01)) return 1;
  // derefinement: curvature < 0.005
  if ((max_rad < 0.3) || ( min_rad > 2.9 ) || (maxeps < 0.005)) return -1;
  // otherwise, stay
  return 0;
}

void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}


void update_planet(Real dt, int NT) {
  int neqn = 6*nPlanet;
  Real *y = new Real[neqn+1];
  Real ypc;
  //Real *vp0 = new Real[nPlanet];
  //Real *facc = new Real[nPlanet*2];
 
 //cout << "This is from update_planet() function: outside loop";
 //cout << "ypc: before update: " << ypc << endl;

 if (Globals::my_rank == 0) {
  Real *yout = new Real[neqn];
  Real xt = 0.0;
  Real dtpl = dt/NT;
  int TEST_FIX_PHI =0;

 if (BINARY && FIX_PHI) {
   Real xp0 = PS[0].getRad()*std::cos(PS[0].getPhi())*std::sin(PS[0].getTheta());
   Real xp1 = PS[1].getRad()*std::cos(PS[1].getPhi())*std::sin(PS[1].getTheta());
   Real yp0 = PS[0].getRad()*std::sin(PS[0].getPhi())*std::sin(PS[0].getTheta());
   Real yp1 = PS[1].getRad()*std::sin(PS[1].getPhi())*std::sin(PS[1].getTheta());
   Real x_COM = (PS[0].getMass()*xp0+PS[1].getMass()*xp1)/(PS[0].getMass()+PS[1].getMass());
   Real y_COM = (PS[0].getMass()*yp0+PS[1].getMass()*yp1)/(PS[0].getMass()+PS[1].getMass());
   ypc = atan2(y_COM,x_COM);
   //cout << "ypc: before update: " << ypc << endl;
 //  Real ypc = PS[0].getPhi();
 }

  for (int n=0;n<nPlanet;n++){
    PS[n].initializeRK(&y[n*6]);
  }
  //cout << "This is from update_planet() function: inside loop";
  //cout << "update_planet: r, phi, vr, omega: "<< y[0]<<", " 
  //     << y[1] <<", " << y[2] << ", " 
  //     << y[3] <<endl;
  yout[0] = ODE_TOL;

  for (int i =0; i<NT; i++) {
    int ierr = Embedded_Verner_7_8(y,neqn, xt, dtpl, yout, derivs_facc); 
    if (ierr < 0) {
       cout << " ode solver failed to achieve the accuracy\n";
    }
   for(int j=0; j<neqn;j++) {
      y[j] = yout[j];
   }
   xt += dtpl;
   //cout << "y:" << y[0]<<", "<<y[1]<<", "<<y[2]<<", "<<y[3]<< endl;
   //cout << "xt:" << xt<<", dtpl:"<<dtpl<< endl;
      
  }

  if (BINARY && FIX_PHI) {
   Real omega_save = Omega0;
   //Real Omega0_new = Omega0 + 0.5*(y[3] + y[7]);
   //Omega0 += 0.5*(y[3] + y[7]);
   //y[3] -= (Omega0_new - omega_save);
   //y[7] -= (Omega0_new - omega_save);
   Real xp0 = y[0]*std::cos(y[1])*std::sin(y[2]);
   Real xp1 = y[6]*std::cos(y[7])*std::sin(y[8]);
   Real yp0 = y[0]*std::sin(y[1])*std::sin(y[2]);
   Real yp1 = y[6]*std::sin(y[7])*std::sin(y[8]);
   //Real zp0 = y[0]*std::cos(y[2]);
   //Real zp1 = y[6]*std::cos(y[8]);
   Real x_COM = (PS[0].getMass()*xp0+PS[1].getMass()*xp1)/(PS[0].getMass()+PS[1].getMass());
   Real y_COM = (PS[0].getMass()*yp0+PS[1].getMass()*yp1)/(PS[0].getMass()+PS[1].getMass());
   //Real z_COM = (PS[0].getMass()*zp0+PS[1].getMass()*zp1)/(PS[0].getMass()+PS[1].getMass());
   Real ypc_new = atan2(y_COM,x_COM);
   //cout << "ypc_new: before update: " << ypc_new << endl;
   y[1] -= (ypc_new - ypc);
   y[7] -= (ypc_new - ypc);
  }

  for (int n=0; n < nPlanet; n++){ // copy data to planet
     Real phi = y[6*n+1];
     while (phi > 2.0*PI) phi -= 2.0*PI;
     while (phi < 0.0)    phi += 2.0*PI;
     y[6*n+1] = phi;
  }

  delete [] yout;
  }
#ifdef MPI_PARALLEL
  MPI_Bcast(y, neqn+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


  for (int n=0; n < nPlanet; n++){ // copy data to planet
     PS[n].update(&y[6*n]);
     //cout << "update_planet after cycle: r, phi, vr, omega: "<< y[0]<<", " 
     //  << y[1] <<", " << y[2] << ", " 
     //  << y[3] <<endl;
  }
 
  delete [] y;
}

void derivs_facc(const Real& x, const Real y[], Real dydx[]) {
 Real *facc = new Real[nPlanet*3];

 for (int n=0; n < nPlanet; n++){ // copy data to planet, to calculate the force
   PS[n].update(&y[6*n]);
 }

 for (int n=0; n < nPlanet; n++){
       //if(!FIX_PHI){
       //PlanetUpdateFromDisk(pmb,n);
       PS[n].GravityFromConfig(Omega0,&facc[3*n],FIX_PHI);
       //PS[n].GravityFromConfig(Omega0,&facc[2*n]);
       //} else {
       // facc[2*n] = 0.0;
       // facc[2*n+1] = 0.0;
       //}
   if (nPlanet>1 && feelOthers) {
   for (int i=0; i < nPlanet; i++){
     if (i!=n){
       PS[n].GravityFromPlanet(PS[i],&facc[3*n]);
     }
    }// for i 
    //cout << "derive_facc:planet force for planet "<< n<<":"<< facc[2*n]<<","<< facc[2*n+1]<<endl;
    } // if nPlanet>1
    //cout << "derive_facc:planet force for planet "<< n<<":"<< facc[2*n]<<","<< facc[2*n+1]<<endl;
    int ii = 6*n;
    dydx[ii+0] = y[ii+3];
    dydx[ii+1] = y[ii+4];
    dydx[ii+2] = y[ii+5];
    dydx[ii+3] = facc[3*n];
    dydx[ii+4] = facc[3*n+1];
    dydx[ii+5] = facc[3*n+2];
 }   // for n
 delete [] facc;

}

string convertInt(int number) {
   stringstream ss; //create a stringstream
   ss << number;    //add number to the stream
   return ss.str(); //return a string with the contents of the stream
}

void WritePlanet(Real time) {
  static vector<ofstream*> outflist;
  //ofstream outflist1, outflist2;

 if (Globals::my_rank == 0) {
  static bool first = 1;
  //cout << "This is from WritePlanet() function" << endl;
  if (first) {
  //cout << "This is from WritePlanet() function: inside first" << endl;
    first = 0;
    for (int i =0; i<nPlanet;i++) {
       string filename = "planet"+convertInt(i)+".dat";
       std::setprecision(10);
       if (!res_flag) {
          outflist.push_back(new ofstream(filename.c_str(),ios::out));
       //cout << "filename" << filename << endl;
       } else {
          //cout << "filename" << filename << endl;
	  outflist.push_back(new ofstream(filename.c_str(),ios::app));
	  (*outflist[i]) <<endl<<endl;
       }
    }

  }
 } // my_rank == 0

  for (int i =0; i<nPlanet;i++) {
    if (Globals::my_rank == 0) {
     //cout << "This is from WritePlanet() function: just before write" << endl;
     PS[i].WriteFile((*outflist[i]),time);
    } // my_rank == 0
  }  
  //outflist1.open("planet0.dat");

//  for (int i =0; i<nPlanet;i++) {
//     static long ncount = 0;
//     *outflist[i]   << time << " "<< PS[i].getRad() << " "
//                 << PS[i].getPhi() << " " <<PS[i].getVr()<<" "
//                 << PS[i].getVp() << " " <<PS[i].getMass()<<" "
//                 <<  endl;
//     ncount++;
//     if (ncount%20) outf << flush; 
//  }

 }

Real MyTimeStep(MeshBlock *pmb)
{
  Real float_max = std::numeric_limits<float>::max();
  Real min_dt=float_max;
  //std::cout << "min_dt:" << min_dt << ", float_max:"<< float_max <<std::endl;
/*
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dt;
        Real dt_np = 1000.0;
        dt =  // calculate your own time step here
        min_dt = std::min(min_dt, dt);
      }
    }
  }
*/
 if(nPlanet>1) {
   for (int i=0; i<nPlanet;i++){
      for (int j=i+1;j<nPlanet;j++){
          Real dt;
          Real dij = PS[i].distance(PS[j]);
          //dt = 1.0/(400.0)*dij*
          dt = TWO_PI/(400.0)*dij*
    	    std::sqrt(dij/(PS[i].getMass()+PS[j].getMass()));
          min_dt = std::min(min_dt, dt);
      }
   }
 }
  //std::cout << "min_dt:" << min_dt <<std::endl;
  return min_dt;
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
//

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, 
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real rad_in(0.0), phi_in(0.0), z_in(0.0);
  Real rad_sphin(0.0), theta_sphin(0.0), phi_sphin(0.0);
  Real rad_sph(0.0), theta_sph(0.0), phi_sph(0.0);
  Real vel, vis_vel_r;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          GetCylCoord(pco,rad_in,phi_in,z_in,il,j,k);
          //GetSphCoord(pco,rad_sphin,theta_sphin,phi_sphin,il,j,k);
          //GetSphCoord(pco,rad_sph,theta_sph,phi_sph,il-i,j,k);
          //prim(IDN,k,j,il-i) = DenProfileCyl_gas(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real cs_square_in = PoverRho(rad_in, phi_in, z_in);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          Real omega_dyn_in = std::sqrt(gm0/(rad_in*rad_in*rad_in));
          Real nu     = nu_alpha*cs_square/omega_dyn;
          Real nu_in     = nu_alpha*cs_square_in/omega_dyn_in;
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il)*nu_in/nu;
          //prim(IDN,k,j,il-i) = dfloor;
          vel = VelProfileCyl_gas(rad,phi,z);
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          //vis_vel_r     = -1.5*(nu_iso/rad);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          //prim(IM1,k,j,il-i) = std::min(prim(IM1,k,j,il),0.0);
          prim(IM1,k,j,il-i) = std::min(vis_vel_r,0.0);
          prim(IM2,k,j,il-i) = vel;
          prim(IM3,k,j,il-i) = 0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverRho(rad, phi, z)*prim(IDN,k,j,il-i);
            //prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
            //prim(IEN,k,j,il-i) = PoverRho(rad, phi, z)*prim(IDN,k,j,il-i)/(gamma_gas - 1.0);;
            //prim(IEN,k,j,il-i) += 0.5*(SQR(prim(IM1,k,j,il-i))+SQR(prim(IM2,k,j,il-i))
            //                           + SQR(prim(IM3,k,j,il-i)))*prim(IDN,k,j,il-i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          GetCylCoord(pco,rad_in,phi_in,z_in,il,j,k);
          GetSphCoord(pco,rad_sph,theta_sph,phi_sph,il-i,j,k);
          GetSphCoord(pco,rad_sphin,theta_sphin,phi_sphin,il,j,k);
          //prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          //
          //Real cs_square = PoverRho(rad, phi, z);
          //Real cs_square_in = PoverRho(rad_in, phi_in, z_in);
          //Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          //Real omega_dyn_in = std::sqrt(gm0/(rad_in*rad_in*rad_in));
          //Real nu     = nu_alpha*cs_square/omega_dyn;
          //Real nu_in     = nu_alpha*cs_square_in/omega_dyn_in;
          //Real Hdisk = std::sqrt(cs_square)/omega_dyn;
          //Real Hdisk_in = std::sqrt(cs_square_in)/omega_dyn_in;
          //prim(IDN,k,j,il-i) = prim(IDN,k,j,il)*nu_in*Hdisk_in/nu/Hdisk;
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il)*std::pow(rad_sph/rad_sphin,-3.0-1.5*pslope);
          //prim(IDN,k,j,il-i) = dfloor;
          vel = VelProfileCyl_gas(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          //vis_vel_r     = -1.5*(nu_iso/rad);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = std::min(vis_vel_r,0.0);
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = vel;
          //prim(IM1,k,j,il-i) = std::min(prim(IM1,k,j,il),0.0);
          //prim(IM2,k,j,il-i) = prim(IM2,k,j,il);
          //prim(IM3,k,j,il-i) = prim(IM3,k,j,il);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverRho(rad, phi, z)*prim(IDN,k,j,il-i);
            //prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
            //prim(IEN,k,j,il-i) = PoverRho(rad, phi, z)*prim(IDN,k,j,il);
            //prim(IEN,k,j,il-i) = PoverRho(rad, phi, z)*prim(IDN,k,j,il-i)/(gamma_gas - 1.0);;
            //prim(IEN,k,j,il-i) += 0.5*(SQR(prim(IM1,k,j,il-i))+SQR(prim(IM2,k,j,il-i))
            //                           + SQR(prim(IM3,k,j,il-i)))*prim(IDN,k,j,il-i);
        }
      }
    }
  }
}


void DiskInnerX1_2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);

          Real vis_vel_r   = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, il-i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }


        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);


          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          //gas_vel1_ghost = gas_vel1_active;
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, il-i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  }
}
//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real rad_out(0.0), phi_out(0.0), z_out(0.0);
  Real vel, vis_vel_r;
  Real vel_rout, vis_vel_rout;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          GetCylCoord(pco,rad_out,phi_out,z_out,iu,j,k);
          //prim(IDN,k,j,iu+i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          vel_rout = VelProfileCyl_gas(rad_out,phi_out,z_out);
          Real cs_square = PoverRho(rad, phi, z);
          Real cs_square_rout = PoverRho(rad_out, phi_out, z_out);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          Real omega_dyn_rout = std::sqrt(gm0/(rad_out*rad_out*rad_out));
	  Real nu = nu_alpha*cs_square/omega_dyn;
	  Real nu_rout = nu_alpha*cs_square_rout/omega_dyn_rout;
          Real lmom = vel*rad;
          Real lmom_rout = vel_rout*rad_out;
          //Real den = prim(IDN,k,j,iu)*nu_rout*lmom_rout/nu/lmom+
          //    (lmom-lmom_rout)*DenProfileCyl_gas(rad,phi,z)/lmom;
          Real den = prim(IDN,k,j,iu)*nu_rout*lmom_rout/nu/lmom+
              (lmom-lmom_rout)*DenProfileCyl_gas(rad,phi,z)/lmom;
          prim(IDN,k,j,iu+i) = den;
          //prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu)*std::pow(rad/rad_out,-3.0+1.5*pslope)+
          //    (std::sqrt(1.0)-std::sqrt(rad_out/rad))*prim(IDN,k,j,iu);
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn)*
		DenProfileCyl_gas(rad,phi,z)/den;
          /*
	    if ((i==1) and (j==jl) and (k==kl)) {
            Real mdot = 2.0*PI*den*vis_vel_r*rad;
            std::cout << "outer bd rad: "<<rad<<", phi: "<< phi << std::endl;
            std::cout << "outer bd vis_vel_r: "<<vis_vel_r<<", den: "<<den<<", mdot: "<< mdot << std::endl;
            }
          */
           
          //vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          //vis_vel_rout     = -1.5*(nu_alpha*cs_square_rout/rad_out/omega_dyn_rout);
          //vis_vel_r     = -1.5*(nu_iso/rad);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = vis_vel_r;
          prim(IM2,k,j,iu+i) = vel;
          prim(IM3,k,j,iu+i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverRho(rad, phi, z)*prim(IDN,k,j,iu+i);
            //prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
            //prim(IEN,k,j,iu+i) = PoverRho(rad, phi, z)*prim(IDN,k,j,iu+i)/(gamma_gas - 1.0);;
            //prim(IEN,k,j,iu+i) += 0.5*(SQR(prim(IM1,k,j,iu+i))+SQR(prim(IM2,k,j,iu+i))
            //                           + SQR(prim(IM3,k,j,iu+i)))*prim(IDN,k,j,iu+i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          GetCylCoord(pco,rad_out,phi_out,z_out,iu+i,j,k);
          //prim(IDN,k,j,iu+i) = DenProfileCyl_gas(rad,phi,z);

          vel = VelProfileCyl_gas(rad,phi,z);
          vel_rout = VelProfileCyl_gas(rad_out,phi_out,z_out);
          Real cs_square = PoverRho(rad, phi, z);
          Real cs_square_rout = PoverRho(rad_out, phi_out, z_out);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          Real omega_dyn_rout = std::sqrt(gm0/(rad_out*rad_out*rad_out));
	  Real nu = nu_alpha*cs_square/omega_dyn;
	  Real nu_rout = nu_alpha*cs_square_rout/omega_dyn_rout;
          Real lmom = vel*rad;
          Real lmom_rout = vel_rout*rad_out;
          //Real Hdisk = std::sqrt(cs_square)/omega_dyn;
          //Real Hdisk_rout = std::sqrt(cs_square_rout)/omega_dyn_rout;
          //Real den = prim(IDN,k,j,iu)*nu_rout*lmom_rout/nu/lmom+
          //    (lmom-lmom_rout)*DenProfileCyl_gas(rad,phi,z)/lmom;
          Real den = prim(IDN,k,j,iu)*std::pow(rad/rad_out,-3.0+1.5*pslope)*lmom_rout/lmom+
              (lmom-lmom_rout)*DenProfileCyl_gas(rad,phi,z)/lmom;
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn)*
		DenProfileCyl_gas(rad,phi,z)/den;
          //vis_vel_r     = -1.5*(nu_iso/rad);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IDN,k,j,iu+i) = den;
          prim(IM1,k,j,iu+i) = vis_vel_r;
          prim(IM2,k,j,iu+i) = 0.0;
          prim(IM3,k,j,iu+i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverRho(rad, phi, z)*prim(IDN,k,j,iu+i);
            //prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
            //prim(IEN,k,j,iu+i) = PoverRho(rad, phi, z)*prim(IDN,k,j,iu+i)/(gamma_gas - 1.0);;
            //prim(IEN,k,j,iu+i) += 0.5*(SQR(prim(IM1,k,j,iu+i))+SQR(prim(IM2,k,j,iu+i))
            //                           + SQR(prim(IM3,k,j,iu+i)))*prim(IDN,k,j,iu+i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1_2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

          Real vis_vel_r   = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, vis_vel_r;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = vis_vel_r;
          prim(IM2,k,jl-j,i) = vel;
          prim(IM3,k,jl-j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverRho(rad, phi, z)*prim(IDN,k,jl-j,i);
            //prim(IEN,k,jl-j,i) = prim(IEN,k,jl,i);
            //prim(IEN,k,jl-j,i) = PoverRho(rad, phi, z)*prim(IDN,k,jl-j,i)/(gamma_gas - 1.0);;
            //prim(IEN,k,jl-j,i) += 0.5*(SQR(prim(IM1,k,jl-j,i))+SQR(prim(IM2,k,jl-j,i))
            //                           + SQR(prim(IM3,k,jl-j,i)))*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = vis_vel_r;
          prim(IM2,k,jl-j,i) = 0.0;
          prim(IM3,k,jl-j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverRho(rad, phi, z)*prim(IDN,k,jl-j,i);
            //prim(IEN,k,jl-j,i) = prim(IEN,k,jl,i);
            //prim(IEN,k,jl-j,i) = PoverRho(rad, phi, z)*prim(IDN,k,jl-j,i)/(gamma_gas - 1.0);;
            //prim(IEN,k,jl-j,i) += 0.5*(SQR(prim(IM1,k,jl-j,i))+SQR(prim(IM2,k,jl-j,i))
            //                           + SQR(prim(IM3,k,jl-j,i)))*prim(IDN,k,jl-j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, 
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, vis_vel_r;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = vis_vel_r;
          prim(IM2,k,ju+j,i) = vel;
          prim(IM3,k,ju+j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverRho(rad, phi, z)*prim(IDN,k,ju+j,i);
            //prim(IEN,k,ju+j,i) = prim(IEN,k,ju,i);
            //prim(IEN,k,ju+j,i) = PoverRho(rad, phi, z)*prim(IDN,k,ju+j,i)/(gamma_gas - 1.0);;
            //prim(IEN,k,ju+j,i) += 0.5*(SQR(prim(IM1,k,ju+j,i))+SQR(prim(IM2,k,ju+j,i))
            //                           + SQR(prim(IM3,k,ju+j,i)))*prim(IDN,k,ju+j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          prim(IM1,k,ju+j,i) = vis_vel_r;
          prim(IM2,k,ju+j,i) = 0.0;
          prim(IM3,k,ju+j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverRho(rad, phi, z)*prim(IDN,k,ju+j,i);
            //prim(IEN,k,ju+j,i) = prim(IEN,k,ju,i);
            //prim(IEN,k,ju+j,i) = PoverRho(rad, phi, z)*prim(IDN,k,ju+j,i)/(gamma_gas - 1.0);;
            //prim(IEN,k,ju+j,i) += 0.5*(SQR(prim(IM1,k,ju+j,i))+SQR(prim(IM2,k,ju+j,i))
            //                           + SQR(prim(IM3,k,ju+j,i)))*prim(IDN,k,ju+j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, vis_vel_r;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          prim(IM1,kl-k,j,i) = vis_vel_r;
          prim(IM2,kl-k,j,i) = vel;
          prim(IM3,kl-k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverRho(rad, phi, z)*prim(IDN,kl-k,j,i);
            //prim(IEN,kl-k,j,i) = prim(IEN,kl,j,i);
            //prim(IEN,kl-k,j,i) = PoverRho(rad, phi, z)*prim(IDN,kl-k,j,i)/(gamma_gas - 1.0);;
            //prim(IEN,kl-k,j,i) += 0.5*(SQR(prim(IM1,kl-k,j,i))+SQR(prim(IM2,kl-k,j,i))
            //                           + SQR(prim(IM3,kl-k,j,i)))*prim(IDN,kl-k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          prim(IM1,kl-k,j,i) = vis_vel_r;
          prim(IM2,kl-k,j,i) = 0.0;
          prim(IM3,kl-k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverRho(rad, phi, z)*prim(IDN,kl-k,j,i);
            //prim(IEN,kl-k,j,i) = prim(IEN,kl,j,i);
            //prim(IEN,kl-k,j,i) = PoverRho(rad, phi, z)*prim(IDN,kl-k,j,i)/(gamma_gas - 1.0);;
            //prim(IEN,kl-k,j,i) += 0.5*(SQR(prim(IM1,kl-k,j,i))+SQR(prim(IM2,kl-k,j,i))
            //                           + SQR(prim(IM3,kl-k,j,i)))*prim(IDN,kl-k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, vis_vel_r;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          prim(IM1,ku+k,j,i) = vis_vel_r;
          prim(IM2,ku+k,j,i) = vel;
          prim(IM3,ku+k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverRho(rad, phi, z)*prim(IDN,ku+k,j,i);
            //prim(IEN,ku+k,j,i) = prim(IEN,ku,j,i);
            //prim(IEN,ku+k,j,i) = PoverRho(rad, phi, z)*prim(IDN,ku+k,j,i)/(gamma_gas - 1.0);;
            //prim(IEN,ku+k,j,i) += 0.5*(SQR(prim(IM1,ku+k,j,i))+SQR(prim(IM2,ku+k,j,i))
            //                           + SQR(prim(IM3,ku+k,j,i)))*prim(IDN,ku+k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl_gas(rad,phi,z);
          vel = VelProfileCyl_gas(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          Real cs_square = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
          prim(IM1,ku+k,j,i) = vis_vel_r;
          prim(IM2,ku+k,j,i) = 0.0;
          prim(IM3,ku+k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverRho(rad, phi, z)*prim(IDN,ku+k,j,i);
            //prim(IEN,ku+k,j,i) = prim(IEN,ku,j,i);
            //prim(IEN,ku+k,j,i) = PoverRho(rad, phi, z)*prim(IDN,ku+k,j,i)/(gamma_gas - 1.0);;
            //prim(IEN,ku+k,j,i) += 0.5*(SQR(prim(IM1,ku+k,j,i))+SQR(prim(IM2,ku+k,j,i))
            //                           + SQR(prim(IM3,ku+k,j,i)))*prim(IDN,ku+k,j,i);
        }
      }
    }
  }
}


