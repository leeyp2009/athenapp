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
void GetPlanetAcc(const int order, Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gap(const Real rad, const Real phi, const Real z, const Real diff);
//Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
//                        const Real den_ratio, const Real H_ratio);
//Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
// problem parameters which are useful to make global to this file
Real tau_relax, rad_soft, gmstar, gmp, inv_sqrt2gmp, rad_planet, phi_planet_0, z_planet, ecc_planet,
inv_rad_planet, t0_planet, t_end_planet, vk_planet, omega_planet, inv_omega_planet, cs_planet,
gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, beta, gMth, nu_alpha,
dfloor, Omega0, user_dt, sigma0, amp, A_gap;
Real rad_planet1, phi_planet_1, z_planet1,gmp1, ecc_planet1;
Real hst_next_time, hst_dt;
int planet_output, res_flag;

//Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], weight_dust[NDUSTFLUIDS];
bool Damping_Flag, Isothermal_Flag, MassTransfer_Flag, RadiativeConduction_Flag,
		 Gap_Flag, TransferFeedback_Flag;

Real x1min, x1max, tau_damping, damping_rate, Hill_radius, accretion_radius, accretion_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
    inner_width_damping, outer_width_damping;
int PlanetaryGrvaityOrder;
int nPlanet;
vector<Planet> PS;
int NT;
bool BINARY;
bool FIX_PHI;
int IndirectTerm, feelOthers;
Real binary_orb;
Real ODE_TOL;
Real *mass0 = new Real[nPlanet];
Real *rp0 = new Real[nPlanet];
Real *phi0 = new Real[nPlanet];
Real *ecc0 = new Real[nPlanet];
Real *vp0 = new Real[nPlanet];
Real *vr0 = new Real[nPlanet];

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
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar, const int i);

void update_planet(Real dt, int NT);
void derivs_facc(const Real& x, const Real y[], Real dydx[]);



void writePlanet(Real time);

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
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
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
/*
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // allocateDataField
  if (gmp>0.0) { 
  //if ((gmp>0.0)&&(!res_flag)) { 
      AllocateRealUserMeshBlockDataField(1);
      ruser_meshblock_data[0].NewAthenaArray(4*nPlanet);
    for (int n=0;n<nPlanet;n++) {
      //if (!res_flag) {
        //std::cout << "inside res_flag:" << res_flag <<std::endl;
        //ruser_mesh_data[0](4*n+0) = PS[n].getRad();
        //ruser_mesh_data[0](4*n+1) = PS[n].getPhi();
        //ruser_mesh_data[0](4*n+2) = PS[n].getVr();
        //ruser_mesh_data[0](4*n+3) = PS[n].getVp();
      //}
        ruser_meshblock_data[0](4*n+0) = 1.0;
        ruser_meshblock_data[0](4*n+1) = 0.0;
        ruser_meshblock_data[0](4*n+2) = 0.0;
        ruser_meshblock_data[0](4*n+3) = 1.0;
        std::cout << "InitUserMeshBlock ruser_mesh_data 1:" << ruser_meshblock_data[0](0) <<std::endl;
    }

  }

 }
}
*/

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //BINARY = 0;
  FIX_PHI = 1;
  feelOthers = 1;
  NT     = 1;
  binary_orb = -1;
  ODE_TOL = 1.0e-8;
  // Get parameters for gravitatonal potential of central point mass
  gm0                      = pin->GetOrAddReal("problem", "GM", 0.0);
  r0                       = pin->GetOrAddReal("problem", "r0", 1.0);
  Damping_Flag             = pin->GetBoolean("problem", "Damping_Flag");
  Isothermal_Flag          = pin->GetBoolean("problem", "Isothermal_Flag");
  Gap_Flag                 = pin->GetOrAddBoolean("problem", "Gap_Flag", false);
  MassTransfer_Flag        = pin->GetOrAddBoolean("problem", "MassTransfer_Flag", false);
  RadiativeConduction_Flag = pin->GetOrAddBoolean("problem", "RadiativeConduction_Flag", false);
  TransferFeedback_Flag    = pin->GetOrAddBoolean("problem", "TransferFeedback_Flag", true);
  //Relaxation_Flag = pin->GetBoolean("problem",   "Relaxation_Flag");

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



  // The parameters of one planet
  tau_relax        = pin->GetOrAddReal("hydro",      "tau_relax",    0.01);
  nPlanet          = pin->GetOrAddInteger("problem",    "Nplanet",   1); // Number of the planet
  rad_planet       = pin->GetOrAddReal("problem",    "rad_planet",   1.0); // radial position of the planet
  phi_planet_0     = pin->GetOrAddReal("problem",    "phi_planet",   0.0); // azimuthal position of the planet
  z_planet         = pin->GetOrAddReal("problem",    "z_planet",     0.0); // vertical position of the planet
  ecc_planet       = pin->GetOrAddReal("problem",    "ecc_planet",   0.0); // eccentricity of the planet
  gmp              = pin->GetOrAddReal("problem",    "GMp",          0.0); // GM of the planet
  IndirectTerm     = pin->GetOrAddInteger("problem",    "IndTerm",  1); // indirect term of the planet potential
if (nPlanet==2){
  rad_planet1       = pin->GetOrAddReal("problem",    "rad_planet1",   1.0); // radial position of the planet
  phi_planet_1      = pin->GetOrAddReal("problem",    "phi_planet1",   0.0); // azimuthal position of the planet
  z_planet1         = pin->GetOrAddReal("problem",    "z_planet1",     0.0); // vertical position of the planet
  ecc_planet1       = pin->GetOrAddReal("problem",    "ecc_planet1",   0.0); // eccentricity of the planet
  gmp1              = pin->GetOrAddReal("problem",    "GMp1",          0.0); // GM of the planet
  feelOthers        = pin->GetOrAddInteger("problem",  "feelOthers", 1); // feel gravity from another planet
  BINARY            = pin->GetOrAddInteger("problem",  "Binary", 0); // binary flag
}
  t0_planet        = (pin->GetOrAddReal("problem",   "t0_planet",    0.0))*TWO_PI; // time to put in the planet
  t_end_planet     = (pin->GetOrAddReal("problem",   "t_end_planet", HUGE_NUMBER))*TWO_PI; // time to disapear the planet
  user_dt          = pin->GetOrAddReal("problem",    "user_dt",      0.0);
  planet_output    = pin->GetOrAddInteger("problem","planet_output",0);
  res_flag         = pin->GetOrAddInteger("problem","restart",0);

  if(BINARY) {
    Omega0 = Omega0*std::sqrt(1.0+gmp+gmp1);
  } else {
    Omega0 = Omega0*std::sqrt(1.0+gmp);
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


  PlanetaryGrvaityOrder = pin->GetOrAddInteger("problem", "PlanetaryGrvaityOrder", 2);
  if ((PlanetaryGrvaityOrder != 2) || (PlanetaryGrvaityOrder != 4) || (PlanetaryGrvaityOrder != 6))
    PlanetaryGrvaityOrder = 2;

  if (NON_BAROTROPIC_EOS)
    cs_planet = std::sqrt(p0_over_r0*std::pow(rad_planet/r0, pslope));
  else
    cs_planet = std::sqrt(p0_over_r0);

  if (gmp != 0.0) inv_sqrt2gmp = 1.0/std::sqrt(2.0*gmp);

  //gMth = gm0*SQR(cs_planet)*cs_planet/(SQR(vk_planet)*vk_planet);

  if (t_end_planet < t0_planet)
    t_end_planet = t0_planet;

  Hill_radius = (std::pow(gmp/gm0*ONE_3RD, ONE_3RD)*rad_planet);

  rad_soft  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  rad_soft *= Hill_radius; // 0.6 *r_Hill

  accretion_radius  = pin->GetOrAddReal("problem", "accretion_radius", 0.3); // Accretion radius of planets
  accretion_radius *= Hill_radius;

  if (accretion_radius < rad_soft)
    accretion_radius = 1.1*rad_soft;

  accretion_rate = pin->GetOrAddReal("problem", "accretion_rate", 0.1); // Accretion radius of planets

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




/*
  if (!res_flag){ 
     for (int n = 0; n<nPlanet; n++){
       PS.push_back(Planet(rp0[n],phi0[n],mass0[n]));
       PS[n].setIndex(n);
       PS[n].setVr(vr0[n]);
       PS[n].setVp(vp0[n]);
     }
   } else {
     for (int n = 0; n<nPlanet; n++){
       std::cout << "InitUserMesh restart ruser_mesh_data 1:" << ruser_mesh_data[0](0) <<std::endl;
       std::cout << "restart ruser_mesh_data 2:" << ruser_mesh_data[0](1) <<std::endl;
       std::cout << "restart mass0:" << mass0[n]<<", n:" << n <<std::endl;
       PS.push_back(Planet(ruser_mesh_data[0](4*n+0),ruser_mesh_data[0](4*n+1),1.0e-3));
       PS[n].setIndex(n);
       //PS[n].setRad(ruser_mesh_data[0](4*n+0)); 
       //PS[n].setPhi(ruser_mesh_data[0](4*n+1)); 
       PS[n].setVr(ruser_mesh_data[0](4*n+2)); 
       PS[n].setVp(ruser_mesh_data[0](4*n+3)); 
     }
   }
*/

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
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

/*
  Real *mass0 = new Real[nPlanet];
  Real *rp0 = new Real[nPlanet];
  Real *phi0 = new Real[nPlanet];
  Real *ecc0 = new Real[nPlanet];
  Real *vp0 = new Real[nPlanet];
  Real *vr0 = new Real[nPlanet];
*/
  if (nPlanet==1){
    mass0[0] = gmp;
    rp0[0] = rad_planet;
    phi0[0] = phi_planet_0*PI;
    ecc0[0] = ecc_planet;
  } else if (nPlanet==2) {
    mass0[0] = gmp;
    rp0[0] = rad_planet;
    phi0[0] = phi_planet_0*PI;
    ecc0[0] = ecc_planet;
    mass0[1] = gmp1;
    rp0[1] = rad_planet1;
    phi0[1] = phi_planet_1*PI;
    ecc0[1] = ecc_planet1;
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
  } else {
    //Omega0 = Omega0*std::sqrt((1.0+gmp));
    for (int n = 0; n<nPlanet; n++){
      rp0[n] *= (1.0-ecc0[n]);
      vp0[n] = std::sqrt((mass0[n]+1.0)/rp0[n])*std::sqrt(1.0+ecc0[n]) - Omega0*rp0[n];
      vr0[n] = 0.0;
    }
  }

     for (int n = 0; n<nPlanet; n++){
       PS.push_back(Planet(rp0[n],phi0[n],mass0[n]));
       PS[n].setIndex(n);
       PS[n].setVr(vr0[n]);
       PS[n].setVp(vp0[n]);
     }


  // allocateDataField
  if (gmp>0.0) { 
  //if ((gmp>0.0)&&(!res_flag)) { 
      AllocateRealUserMeshDataField(1);
      ruser_mesh_data[0].NewAthenaArray(4*nPlanet);
    for (int n=0;n<nPlanet;n++) {
      //if (!res_flag) {
        //std::cout << "inside res_flag:" << res_flag <<std::endl;
        ruser_mesh_data[0](4*n+0) = rp0[n];
        ruser_mesh_data[0](4*n+1) = phi0[n];
        ruser_mesh_data[0](4*n+2) = vr0[n];
        ruser_mesh_data[0](4*n+3) = vp0[n];
      //}
        //ruser_mesh_data[0](4*n+0) = 1.0;
        //ruser_mesh_data[0](4*n+1) = 0.0;
        //ruser_mesh_data[0](4*n+2) = 0.0;
        //ruser_mesh_data[0](4*n+3) = 1.0;
        std::cout << "InitUserMesh ruser_mesh_data 1:" << ruser_mesh_data[0](0) <<std::endl;
    }

  }

/*
  if (res_flag){ 
     for (int n = 0; n<nPlanet; n++){
       const Real &Rad_p = ruser_mesh_data[0](4*n+0); 
       const Real &Phi_p = ruser_mesh_data[0](4*n+1); 
       const Real &Vr_p = ruser_mesh_data[0](4*n+2); 
       const Real &Vp_p = ruser_mesh_data[0](4*n+3); 
       PS.push_back(Planet(Rad_p,Phi_p,1.0e-3));
       PS[n].setIndex(n);
       PS[n].setVr(Vr_p);
       PS[n].setVp(Vp_p);
     }
   }
*/  


//    if (res_flag) {
//     std::cout << "time:" << time <<std::endl;
//     std::cout << "update planet position when restart ..." <<std::endl;
//     std::cout << "ruser_mesh_data:" << Mesh::ruser_mesh_data[0](0) <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
     // for (int n=0;n<nPlanet;n++) {
     //   PS[n].setRad(ruser_mesh_data[0](4*n+0)); 
     //   PS[n].setPhi(ruser_mesh_data[0](4*n+1)); 
     //   PS[n].setVr(ruser_mesh_data[0](4*n+2)); 
     //   PS[n].setVp(ruser_mesh_data[0](4*n+3)); 
     // }
//    }


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

 //cout << "test: this is InitUserMeshData"<< endl;

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

  std::cout << "problemGen time:" << time <<std::endl;



  //Real qvalue = gmp/gm0;
  //bool vis_defined = phydro->hdif.hydro_diffusion_defined;

  if (Gap_Flag) {
    Real inv_2sigma2 = 1./(2.*SQR(3.0*Hill_radius));
    //Real norm_factor = 1./(std::sqrt(2*PI) * Hill_radius);
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
          Real den_gas_1     = DenProfileCyl_gas(rad, phi, z);
          //Real A_gap         = -(1.0 - std::pow(qvalue, -2.2) * std::pow(nu_alpha, 1.4) * std::pow(cs_square, 3.3));
          Real den_gas_2     = den_gas_1*A_gap*std::exp(-SQR(rad - rad_planet)*inv_2sigma2);
          Real den_gas_total = den_gas_1 + den_gas_2;
          Real pre_diff      = pslope*cs_square + cs_square/den_gas_total*(dslope*den_gas_1 +
                                        2.0*rad*(rad_planet - rad)*inv_2sigma2*den_gas_2);

          Real vis_vel_r     = -1.5*(nu_alpha*cs_square/omega_dyn/rad);
          Real vel_gas_phi   = VelProfileCyl_gap(rad, phi, z, pre_diff);
          //Real vis_vel_r = 0.0;

          if (porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          phydro->u(IDN, k, j, i) = den_gas_total;
          phydro->u(IM1, k, j, i) = den_gas_total*vis_vel_r;

          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            phydro->u(IM2, k, j, i) = den_gas_total*vel_gas_phi;
            phydro->u(IM3, k, j, i) = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            phydro->u(IM2, k, j, i) = 0.0;
            phydro->u(IM3, k, j, i) = den_gas_total*vel_gas_phi;
          }

          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
            phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                          + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
          }

        }
      }
    }
  } else {
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
  }

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  bool flag = false;
  static bool first = 1;
  Real present_time = time + dt;
  //Real hst_next_time = time;
  //int cncycle = ncycle + 1;
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


  //if ((gmp > 0.0) && (Globals::my_rank == 0)) {
  if (gmp > 0.0) {
    //std::cout << "UserWorkInLoop update planet position when restart ..." <<std::endl;
    //std::cout << "UserWorkInLoop time:" << time << ", dt:" << dt <<std::endl;
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[0].data(), 4*nPlanet, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &ruser_mesh_data[0](4*n+0), 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &ruser_mesh_data[0](4*n+1), 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &ruser_mesh_data[0](4*n+2), 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &ruser_mesh_data[0](4*n+3), 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
     std::cout << "time:" << time <<std::endl;
     std::cout << "UserWorkInLoop ruser_mesh_data:" << ruser_mesh_data[0](0) <<std::endl;
     //AthenaArray<Real> &pp = ruser_mesh_data[0];
    //if (res_flag) {
     //std::cout << "time:" << time <<std::endl;
     //std::cout << "update planet position when restart ..." <<std::endl;
     //std::cout << "ruser_mesh_data:" << Mesh::ruser_mesh_data[0](0) <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
      
     //if (first & res_flag) {
     //   first = 0;
        for (int n=0;n<nPlanet;n++) {
          PS[n].setRad(ruser_mesh_data[0](4*n+0)); 
          PS[n].setPhi(ruser_mesh_data[0](4*n+1)); 
          PS[n].setVr(ruser_mesh_data[0](4*n+2)); 
          PS[n].setVp(ruser_mesh_data[0](4*n+3)); 
        }
     //}
     
    //}

     //std::cout << "time:" << time << ", dt:" << dt <<std::endl;
     //std::cout << "update planet position when restart ..." <<std::endl;
     //std::cout << "phip:" << PS[0].getPhi() << "\n" <<std::endl;
     
     //AthenaArray<Real> &y = ruser_mesh_data[0];
     //Real *y = ruser_mesh_data[0];
     //for (int n=0;n<nPlanet;n++){
     //  PS[n].initializeRK(&y(n*4));
     //}

     //writePlanet(time);
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
          for (int j=i+1;j<nPlanet;j++){
              Real dij = PS[i].distance(PS[j]);
              dt_np = std::min(dt_np, TWO_PI/(100.0)*dij*
		    std::sqrt(dij/(PS[i].getMass()+PS[j].getMass())));
          }
       }
       NT = std::max(NT, int(dt/dt_np+0.6));
     }
   if (Globals::my_rank == 0) {
   //   cout << "NT: before update_planet: " << NT << std::endl;
     //std::cout << "my_rank==0,rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
     writePlanet(time);
     //update_planet(dt, NT);
   }
     //std::cout << "before update_planet,rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
     //update_planet(dt, NT);
     //std::cout << "after update_planet, rp, phip:" << PS[0].getRad() <<", "<< PS[0].getPhi() << "\n" <<std::endl;
     update_planet(dt, NT);

    for (int n=0;n<nPlanet;n++) {
     Real &Rad_p = ruser_mesh_data[0](4*n+0); 
     Real &Phi_p = ruser_mesh_data[0](4*n+1); 
     Real &Vr_p = ruser_mesh_data[0](4*n+2); 
     Real &Vp_p = ruser_mesh_data[0](4*n+3); 
     Rad_p = PS[n].getRad();
     Phi_p = PS[n].getPhi();
     Vr_p = PS[n].getVr();
     Vp_p = PS[n].getVp();
     //pp(4*n+0) = 1.0;
     //pp(4*n+1) = 0.0;
     //pp(4*n+2) = 0.0;
     //pp(4*n+3) = 1.0;
     //ruser_mesh_data[0](4*n+0) = PS[n].getRad();
     //ruser_mesh_data[0](4*n+1) = PS[n].getPhi();
     //ruser_mesh_data[0](4*n+2) = PS[n].getVr();
     //ruser_mesh_data[0](4*n+3) = PS[n].getVp();
     //std::cout << "Rad_p:" << pp(4*n+0) << "\n" <<std::endl;
     std::cout << "UserWorkInLoop updating ruser_mesh_data:" << ruser_mesh_data[0](4*n) <<std::endl;
     std::cout << "Rad_p:" << Rad_p << "\n" <<std::endl;
    }
  }
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
//        writePlanet(time);
//    }
    for (int i=0;i<nPlanet;i++){
    PlanetaryGravity(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar, i);
    //cout << "This is from main() function:\n";
    //cout << "main: rp, phi, vr, omega: "<< PS[0].getRad()<<", " 
    //     << PS[0].getPhi() <<", " << PS[0].getVr() << ", " 
    //     << PS[0].getVp() <<endl;
    }
}

  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
    OuterWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  }

  if ((gmp > 0.0) && MassTransfer_Flag)
    MassTransferWithinHill(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  if (Isothermal_Flag && NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  else if (beta > 0.0)
    ThermalRelaxation(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

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
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar, const int np) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //update_planet(dt, NT);
  //Real phi_planet_move    = omega_planet*time + phi_planet_0;
  
  Real phi_planet_new    = PS[np].getPhi();
  Real rad_planet_new    = PS[np].getRad();
  int nc1 = pmb->ncells1;

  //if (pmb->porb->orbital_advection_defined)
  //  phi_planet_new -= Omega0*time;
  if (phi_planet_new>TWO_PI)
    phi_planet_new += -TWO_PI;
  else if (phi_planet_new<0.0)
    phi_planet_new += TWO_PI;

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

  //Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
		if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //Real rad_arr, phi_arr, z_arr;
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

          planet_gm(i) = PS[np].getMass();

          x_dis(i) = rad_arr(i)*std::cos(phi_arr(i)) - rad_planet_new*std::cos(phi_planet_new);
          y_dis(i) = rad_arr(i)*std::sin(phi_arr(i)) - rad_planet_new*std::sin(phi_planet_new);
          z_dis(i) = z_arr(i) - z_planet;

          rad_dis(i) = rad_arr(i) - rad_planet_new*cos(phi_arr(i) - phi_planet_new);
          phi_dis(i) = rad_planet_new*sin(phi_arr(i) - phi_planet_new);

          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));
          //distance(i)        = std::sqrt(distance_square(i));

            //second order gravity
          if (PlanetaryGrvaityOrder == 2) {
            Real sec_g = planet_gm(i)/pow(distance_square(i)+SQR(rad_soft), 1.5);
            acc_r(i)   = -sec_g*rad_dis(i); // radial acceleration
            acc_phi(i) = -sec_g*phi_dis(i); // asimuthal acceleration
            acc_z(i)   = -sec_g*z_dis(i);   // vertical acceleartion
          }

          //fourth order gravity
          if (PlanetaryGrvaityOrder == 4) {
            Real forth_g = planet_gm(i)*(5.0*SQR(rad_soft)+2.0*distance_square(i))/
                                    (2.0*pow(SQR(rad_soft)+distance_square(i), 2.5));
            acc_r(i)     = -forth_g*rad_dis(i); // radial acceleration
            acc_phi(i)   = -forth_g*phi_dis(i); // asimuthal acceleration
            acc_z(i)     = -forth_g*z_dis(i);   // vertical acceleartion
          }

          //sixth order gravity
          if (PlanetaryGrvaityOrder == 6) {
            Real sixth_g = planet_gm(i)*(35.0*SQR(SQR(rad_soft))+28.0*SQR(rad_soft)*distance_square(i)+
                                    8.0*distance_square(i))/(8.0*pow(SQR(rad_soft)+distance_square(i), 3.5));
            acc_r(i)     = -sixth_g*rad_dis(i); // radial acceleration
            acc_phi(i)   = -sixth_g*phi_dis(i); // azimuthal acceleration
            acc_z(i)     = -sixth_g*z_dis(i);   // vertical acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = planet_gm(i)/SQR(rad_planet_new);
             acc_r_temp(i)   = -temp*std::cos(phi_arr(i) - phi_planet_new);
	     acc_phi_temp(i) = temp*std::sin(phi_arr(i) - phi_planet_new);
	     acc_z_temp(i)   = -temp*z_planet/rad_planet_new;
          }

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = +dt*gas_rho*(acc_r(i)+acc_r_temp(i));
          Real delta_mom2 = +dt*gas_rho*(acc_phi(i)+acc_phi_temp(i));
          Real delta_mom3 = +dt*gas_rho*(acc_z(i)+acc_z_temp(i));

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
        }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
			//Real rad_arr, theta_arr, phi_arr;
        GetCylCoord(pmb->pcoord, rad_arr(i), theta_arr(i), phi_arr(i), i, j, k);

          planet_gm(i) = PS[np].getMass();

          x_dis(i) = rad_arr(i)*std::cos(phi_arr(i))*std::sin(theta_arr(i)) - 
		             rad_planet_new*std::cos(phi_planet_new)*std::sin(theta_arr(i));
          y_dis(i) = rad_arr(i)*std::sin(phi_arr(i))*std::sin(theta_arr(i)) - 
		             rad_planet_new*std::sin(phi_planet_new)*std::sin(theta_arr(i));
          z_dis(i) = rad_arr(i)*std::cos(theta_arr(i))- rad_planet_new*std::cos(rad_planet_new);

          rad_dis(i) = rad_arr(i) - rad_planet_new*std::sin(theta_arr(i))*std::cos(phi_arr(i) - phi_planet_new);
		  theta_dis(i) = -rad_planet_new*std::cos(theta_arr(i))*std::cos(phi_arr(i) - phi_planet_new);
          phi_dis(i) = rad_planet_new*std::sin(phi_arr(i) - phi_planet_new);

          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));
          //distance(i)        = std::sqrt(distance_square(i));

            //second order gravity
          if (PlanetaryGrvaityOrder == 2) {
            Real sec_g = planet_gm(i)/pow(distance_square(i)+SQR(rad_soft), 1.5);
            acc_r(i)   = -sec_g*rad_dis(i); // radial acceleration
            acc_theta(i) = -sec_g*theta_dis(i); // polar acceleration
            acc_phi(i)   = -sec_g*phi_dis(i);   // azimuthal acceleartion
          }

          //fourth order gravity
          if (PlanetaryGrvaityOrder == 4) {
            Real forth_g = planet_gm(i)*(5.0*SQR(rad_soft)+2.0*distance_square(i))/
                                    (2.0*pow(SQR(rad_soft)+distance_square(i), 2.5));
            acc_r(i)     = -forth_g*rad_dis(i); // radial acceleration
            acc_theta(i)   = -forth_g*theta_dis(i); // polar acceleration
            acc_phi(i)     = -forth_g*phi_dis(i);   // azimuthal acceleartion
          }

          //sixth order gravity
          if (PlanetaryGrvaityOrder == 6) {
            Real sixth_g = planet_gm(i)*(35.0*SQR(SQR(rad_soft))+28.0*SQR(rad_soft)*distance_square(i)+
                                    8.0*distance_square(i))/(8.0*pow(SQR(rad_soft)+distance_square(i), 3.5));
            acc_r(i)     = -sixth_g*rad_dis(i); // radial acceleration
            acc_theta(i)   = -sixth_g*theta_dis(i); // polar acceleration
            acc_phi(i)     = -sixth_g*phi_dis(i);   // azimuthal acceleartion
          }
            
          if (IndirectTerm) {
             Real temp = planet_gm(i)/SQR(rad_planet_new);
             acc_r(i)   += -temp*std::cos(phi_arr(i) - phi_planet_new)*std::sin(theta_arr(i));
	     acc_theta(i) += -temp*std::cos(phi_arr(i) - phi_planet_new)*std::cos(theta_arr(i));
	     acc_phi(i)   += +temp*std::sin(phi_arr(i) - phi_planet_new);
          }

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = +dt*gas_rho*acc_r(i);
          Real delta_mom2 = +dt*gas_rho*acc_theta(i);
          Real delta_mom3 = +dt*gas_rho*acc_phi(i);

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
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  Real phi_planet_new    = PS[0].getPhi() ;
  Real rad_planet_new    = PS[0].getRad() ;
  if (pmb->porb->orbital_advection_defined)
    phi_planet_new -= Omega0*time;
  if (phi_planet_new>TWO_PI)
    phi_planet_new += -TWO_PI;
  else if (phi_planet_new<0.0)
    phi_planet_new += TWO_PI;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_rad_soft   = 1.0/rad_soft;
  Real inv_rad_soft_3 = 1.0/(rad_soft*rad_soft*rad_soft);

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    //Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      //Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //Real x1 = pmb->pcoord->x1v(i);
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        //Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= (t0_planet + t_growth)) {
        if (time >= t0_planet) {

          Real x_dis = rad*std::cos(phi) - rad_planet_new*std::cos(phi_planet_new);
          Real y_dis = rad*std::sin(phi) - rad_planet_new*std::sin(phi_planet_new);
          Real z_dis = z - z_planet;

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          Real distance        = std::sqrt(distance_square);

          if ((distance > rad_soft) && (distance <= accretion_radius)) {
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
            R_func(i)      = SQR((rad - radius_inner_damping)*inv_inner_damp);
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

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

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
            R_func(i)      = SQR((rad - radius_inner_damping)*inv_inner_damp);
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

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

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


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
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
            R_func(i)      = SQR((rad - radius_outer_damping)*inv_outer_damp);
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

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

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
            R_func(i)      = SQR((rad - radius_outer_damping)*inv_outer_damp);
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

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*damping_invtau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*damping_invtau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

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
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(i);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}




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

int RefinementCondition(MeshBlock *pmb) {
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
  int neqn = 4*nPlanet;
  Real *y = new Real[neqn+1];
  Real ypc;
  //Real *vp0 = new Real[nPlanet];
  //Real *facc = new Real[nPlanet*2];
 
  //cout << "This is from update_planet() function: outside loop";
 //if (true) {
 //cout << "ypc: before update: " << ypc << endl;

 if (Globals::my_rank == 0) {
  Real *yout = new Real[neqn];
  Real xt = 0.0;
  Real dtpl = dt/NT;
  int TEST_FIX_PHI =0;

 if (BINARY && FIX_PHI) {
   Real xp0 = PS[0].getRad()*std::cos(PS[0].getPhi());
   Real xp1 = PS[1].getRad()*std::cos(PS[1].getPhi());
   Real yp0 = PS[0].getRad()*std::sin(PS[0].getPhi());
   Real yp1 = PS[1].getRad()*std::sin(PS[1].getPhi());
   Real x_COM = (PS[0].getMass()*xp0+PS[1].getMass()*xp1)/(PS[0].getMass()+PS[1].getMass());
   Real y_COM = (PS[0].getMass()*yp0+PS[1].getMass()*yp1)/(PS[0].getMass()+PS[1].getMass());
   ypc = atan2(y_COM,x_COM);
   //cout << "ypc: before update: " << ypc << endl;
 //} else {
 //  Real ypc = PS[0].getPhi();
 }

  for (int n=0;n<nPlanet;n++){
    PS[n].initializeRK(&y[n*4]);
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
   for(int j=0; j<neqn;j++)
      y[j] = yout[j];
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
   Real xp0 = y[0]*std::cos(y[1]);
   Real xp1 = y[4]*std::cos(y[5]);
   Real yp0 = y[0]*std::sin(y[1]);
   Real yp1 = y[4]*std::sin(y[5]);
   Real x_COM = (PS[0].getMass()*xp0+PS[1].getMass()*xp1)/(PS[0].getMass()+PS[1].getMass());
   Real y_COM = (PS[0].getMass()*yp0+PS[1].getMass()*yp1)/(PS[0].getMass()+PS[1].getMass());
   Real ypc_new = atan2(y_COM,x_COM);
   //cout << "ypc_new: before update: " << ypc_new << endl;
   y[1] -= (ypc_new - ypc);
   y[5] -= (ypc_new - ypc);
  }

  for (int n=0; n < nPlanet; n++){ // copy data to planet
     Real phi = y[4*n+1];
     while (phi > 2.0*PI) phi -= 2.0*PI;
     while (phi < 0.0)    phi += 2.0*PI;
     y[4*n+1] = phi;
  }

  delete [] yout;
  }
#ifdef MPI_PARALLEL
  MPI_Bcast(y, neqn+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


  for (int n=0; n < nPlanet; n++){ // copy data to planet
     PS[n].update(&y[4*n]);
     //cout << "update_planet after cycle: r, phi, vr, omega: "<< y[0]<<", " 
     //  << y[1] <<", " << y[2] << ", " 
     //  << y[3] <<endl;
  }
 
  delete [] y;
}

void derivs_facc(const Real& x, const Real y[], Real dydx[]) {
 Real *facc = new Real[nPlanet*2];

 for (int n=0; n < nPlanet; n++){ // copy data to planet, to calculate the force
   PS[n].update(&y[4*n]);
 }

 for (int n=0; n < nPlanet; n++){
       //if(!FIX_PHI){
       PS[n].GravityFromConfig(Omega0,&facc[2*n],FIX_PHI);
       //PS[n].GravityFromConfig(Omega0,&facc[2*n]);
       //} else {
       // facc[2*n] = 0.0;
       // facc[2*n+1] = 0.0;
       //}
   if (nPlanet>1 && feelOthers) {
   for (int i=0; i < nPlanet; i++){
     if (i!=n){
       PS[n].GravityFromPlanet(PS[i],&facc[2*n]);
     }
    }// for i 
    //cout << "derive_facc:planet force for planet "<< n<<":"<< facc[2*n]<<","<< facc[2*n+1]<<endl;
    } // if nPlanet>1
    //cout << "derive_facc:planet force for planet "<< n<<":"<< facc[2*n]<<","<< facc[2*n+1]<<endl;
    int ii = 4*n;
    dydx[ii+0] = y[ii+2];
    dydx[ii+1] = y[ii+3];
    dydx[ii+2] = facc[2*n];
    dydx[ii+3] = facc[2*n+1];
 }   // for n
 delete [] facc;

}

string convertInt(int number) {
   stringstream ss; //create a stringstream
   ss << number;    //add number to the stream
   return ss.str(); //return a string with the contents of the stream
}

void writePlanet(Real time) {
  static vector<ofstream*> outflist;
  //ofstream outflist1, outflist2;

 if (Globals::my_rank == 0) {
  static bool first = 1;
  //cout << "This is from writePlanet() function" << endl;
  if (first) {
  //cout << "This is from writePlanet() function: inside first" << endl;
    first = 0;
    for (int i =0; i<nPlanet;i++) {
       string filename = "planet"+convertInt(i)+".dat";
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
     //cout << "This is from writePlanet() function: just before write" << endl;
     PS[i].writeFile((*outflist[i]),time);
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



} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
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

          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

          //Vr_interpolate_outer_nomatter(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              //gas_vel1_active, gas_vel1_ghost);
          //gas_vel1_ghost = gas_vel1_active;

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

          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

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

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
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

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  i, jl-j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, jl-j, i);
          Real &gas_vel1_ghost = prim(IM1, k, jl-j, i);
          Real &gas_vel2_ghost = prim(IM2, k, jl-j, i);
          Real &gas_vel3_ghost = prim(IM3, k, jl-j, i);

          Real vis_vel_r   = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, jl-j, i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, i, jl-j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_K     = vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, jl-j, i);
          Real &gas_vel1_ghost = prim(IM1, k, jl-j, i);
          Real &gas_vel2_ghost = prim(IM2, k, jl-j, i);
          Real &gas_vel3_ghost = prim(IM3, k, jl-j, i);
          Real &gas_pres_ghost = prim(IEN, k, jl-j, i);

          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, jl-j, i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, i, ju+j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, ju+j, i);
          Real &gas_vel1_ghost = prim(IM1, k, ju+j, i);
          Real &gas_vel2_ghost = prim(IM2, k, ju+j, i);
          Real &gas_vel3_ghost = prim(IM3, k, ju+j, i);

          Real vis_vel_r   = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, ju+j, i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, i, ju+j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad_ghost);
          Real vel_K     = vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, ju+j, i);
          Real &gas_vel1_ghost = prim(IM1, k, ju+j, i);
          Real &gas_vel2_ghost = prim(IM2, k, ju+j, i);
          Real &gas_vel3_ghost = prim(IM3, k, ju+j, i);
          Real &gas_pres_ghost = prim(IEN, k, ju+j, i);

          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, ju+j, i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, kl-k);
          prim(IDN, kl-k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1, kl-k, j, i) = 0.0;
          prim(IM2, kl-k, j, i) = vel_gas_phi;
          prim(IM3, kl-k, j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, kl-k, j, i) = PoverRho(rad, phi, z)*prim(IDN, kl-k, j, i);

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, kl-k);
          prim(IDN, kl-k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1, kl-k, j, i) = 0.0;
          prim(IM2, kl-k, j, i) = 0.0;
          prim(IM3, kl-k, j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, kl-k, j, i) = PoverRho(rad, phi, z)*prim(IDN, kl-k, j, i);

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, ku+k);
          prim(IDN, ku+k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1, ku+k, j, i) = 0.0;
          prim(IM2, ku+k, j, i) = vel_gas_phi;
          prim(IM3, ku+k, j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, ku+k, j, i) = PoverRho(rad, phi, z)*prim(IDN, ku+k, j, i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, ku+k);
          prim(IDN, ku+k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1, ku+k, j, i) = 0.0;
          prim(IM2, ku+k, j, i) = 0.0;
          prim(IM3, ku+k, j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, ku+k, j, i) = PoverRho(rad, phi, z)*prim(IDN, ku+k, j, i);

        }
      }
    }
  }
}


