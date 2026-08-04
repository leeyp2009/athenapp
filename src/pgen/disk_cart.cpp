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
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

//#include "../nbody/planet.hpp" //planet class

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
void GetSphCoord(Coordinates *pco, Real &r,Real &theta,Real &phi,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, beta;
//Real nu_iso;
Real nu_alpha;
Real dfloor;
Real Omega0;
Real Mp, ecc, Pp, epsilon, area, res, inv_sqrt2gmp;
Real racc, rate;
int nPlanet;
int gorder;
bool IndirectTerm;  
bool Accretion_Flag;  
bool Isothermal_Flag;
int Damping_Flag;
Real x1min, x1max;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
    inner_width_damping, outer_width_damping;
Real damping_rate;
Real vr_out;
//static Real deltaM;
//static Real deltaMp1;
//static Real deltaMp2;
//static Real deltaMp3;
//Real deltaM;
//Real deltaMp1;
//Real deltaMp2;
//Real deltaMp3;
//vector<Planet> PS;
Real HistoryAccretion(MeshBlock *pmb, int iout);
Real PlanetAccretionHistory(MeshBlock *pmb, int iout);
} // namespace

// User-defined boundary conditions for disk simulations


void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,  
                const AthenaArray<Real> &prim_scalar,
                const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
                AthenaArray<Real> &cons_scal);

void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar);
Real PlanetGravityR(Real x1, Real x2, Real x3, Real time);
Real PlanetGravityTheta(Real x1, Real x2, Real x3, Real time);
Real PlanetGravityPhi(Real x1, Real x2, Real x3, Real time);
void PlanetAccretion(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,  AthenaArray<Real> &cons_scalar);
int SphRefinementCondition(MeshBlock *pmb);
int CylRefinementCondition(MeshBlock *pmb);
int RefinementCondition(MeshBlock *pmb);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  //nu_iso = pin->GetOrAddReal("problem","nu_iso",0.0);
  nu_alpha       = pin->GetOrAddReal("problem", "nu_alpha",  0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
    beta       = pin->GetOrAddReal("problem", "beta", 0.0);
    //std::cout << "... p0_over_r0" << p0_over_r0 << std::endl;
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
    //std::cout << "... iso p0_over_r0" << p0_over_r0 << std::endl;
  }

  // Get parameters of planet
  nPlanet = 1;
  Mp = pin->GetOrAddReal("problem","Mp",0.0);
  ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Pp = pin->GetOrAddReal("problem","Pp",0.0);
  epsilon = pin->GetOrAddReal("problem","epsilon",0.0);
  racc = pin->GetOrAddReal("problem","racc",0.0);
  rate = pin->GetOrAddReal("problem","rate",1.0);
  IndirectTerm = pin->GetOrAddBoolean("problem","IndirectTerm", true);
  Accretion_Flag = pin->GetOrAddBoolean("problem","acc_flag", false);
  Damping_Flag    = pin->GetOrAddInteger("problem", "Damp_Flag", 0);
  Isothermal_Flag    = pin->GetOrAddBoolean("problem", "Iso_Flag", true);
  gorder = pin->GetOrAddInteger("problem","gorder",2);

  // Get AMR parameters
  res = pin->GetOrAddReal("problem","res",0.0002);
  area = pin->GetOrAddReal("problem","area",0.006);

  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  if (Mp != 0.0) inv_sqrt2gmp = 1.0/std::sqrt(2.0*gm0*Mp);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

  //std::cout << "radius_inner_damping: " << radius_inner_damping << std::endl;
  //std::cout << "radius_outer_damping: " << radius_outer_damping << std::endl;

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);


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
  // AMR
  if(adaptive==true){
    EnrollUserRefinementCondition(RefinementCondition);
  }

  // Source term
  EnrollUserExplicitSourceFunction(SourceTerm);
  if (nPlanet>0) {
    //AllocateUserHistoryOutput(4*nPlanet);
    AllocateUserHistoryOutput(nPlanet);
    for(int n=0;n<nPlanet;n++) { 
     //std::stringstream dmass_name, dmv1_name, dmv2_name, dmv3_name;
     std::stringstream dmass_name;
     dmass_name << "dmass_" << n + 1;
     //dmv1_name << "dmv1_" << n + 1;
     //dmv2_name << "dmv2_" << n + 1;
     //dmv3_name << "dmv3_" << n + 1;
     EnrollUserHistoryOutput(n, PlanetAccretionHistory, dmass_name.str().c_str(),UserHistoryOperation::sum); // default: sum
     //EnrollUserHistoryOutput(3*n+1, HistoryAccretion, dmv1_name.str().c_str(),UserHistoryOperation::sum);  // default: sum
     //EnrollUserHistoryOutput(3*n+2, HistoryAccretion, dmv2_name.str().c_str(),UserHistoryOperation::sum);  // default: sum
     //EnrollUserHistoryOutput(3*n+3, HistoryAccretion, dmv3_name.str().c_str(),UserHistoryOperation::sum);  // default: sum
    }
 }
    //std::cout << "... this is a test:" << std::endl;
      Real rad(0.0), phi(0.0), z(0.0);
      AllocateRealUserMeshDataField(3);
      //ruser_mesh_data[0].NewAthenaArray(2);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        ruser_mesh_data[0].NewAthenaArray(mesh_size.nx3,mesh_size.nx1);
        ruser_mesh_data[1].NewAthenaArray(mesh_size.nx3,mesh_size.nx1);
        ruser_mesh_data[2].NewAthenaArray(mesh_size.nx3,mesh_size.nx1);
      for (int i=0; i<mesh_size.nx1; i++) {
      	for (int j=0; j<mesh_size.nx3; j++) {
        //GetCylCoord(pcoord,rad,phi,z,i,0,0); // convert to cylindrical coordinates
        //Real cs_square = PoverRho(rad, phi, z);
        //Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        //Real vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vis_vel_r     = -1.5*(nu_alpha*p0_over_r0)*std::pow(x1max/r0,pslope+0.5);
        ruser_mesh_data[0](j,i) = vis_vel_r;
        ruser_mesh_data[1](j,i) = 1.0;
        ruser_mesh_data[2](j,i) = rho0*std::pow(x1max/r0,dslope);
        }
       }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        ruser_mesh_data[0].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
        ruser_mesh_data[1].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
        ruser_mesh_data[2].NewAthenaArray(mesh_size.nx2,mesh_size.nx1);
      for (int i=0; i<mesh_size.nx1; i++) {
      	for (int j=0; j<mesh_size.nx2; j++) {
        //GetCylCoord(pcoord,rad,phi,z,i,0,0); // convert to cylindrical coordinates
        //Real cs_square = PoverRho(rad, phi, z);
        //Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        //Real vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vis_vel_r     = -1.5*(nu_alpha*p0_over_r0)*std::pow(x1max/r0,pslope+0.5);
        ruser_mesh_data[0](j,i) = vis_vel_r;
        ruser_mesh_data[1](j,i) = 1.0;
        ruser_mesh_data[2](j,i) = rho0*std::pow(x1max/r0,dslope);
        }
       }
      }
      
    //for (int n=0;n<nPlanet;n++) {
        //std::cout << "inside res_flag:" << res_flag <<std::endl;
        //ruser_mesh_data[0](0) = 0.0;
        //ruser_mesh_data[0](1) = 1.0;
        //std::cout << "InitUserMesh ruser_mesh_data 1:" << ruser_mesh_data[0](0) <<std::endl;
    //}
     //std::cout << "mesh_size.nx1:" << mesh_size.nx1<<", nx2:"<< mesh_size.nx2<< ", nx3:"<< mesh_size.nx3<<  std::endl;

  return;
}

/*
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  // Allocate space for scratch arrays
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(4*nPlanet);
  for (int i=0;i<4*nPlanet;i++) {
     ruser_meshblock_data[0](i) = 0.0;
  }
     //ruser_meshblock_data[0](3) = -1.0;
  //cout << "... InitUserMeshBlockData:" << ruser_meshblock_data[0](3) << ", " <<  nPlanet << endl;
  return;
}
*/

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================


void Mesh::UserWorkInLoop() {

//Real vr_out;
//Real number;
//Real vr_sum;

//        std::cout << "before MPI: ruser_mesh_data: "<<ruser_mesh_data[0](0)<<", " << ruser_mesh_data[0](1)<< std::endl;

//#ifdef MPI_PARALLEL
//    MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[0].data(), 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[0].data(), 4*nPlanet, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
//#endif

 //Real &vr_sum = ruser_mesh_data[0](0);
 //Real &number = ruser_mesh_data[0](1);
    AthenaArray<Real> &vr_sum = ruser_mesh_data[0];
    AthenaArray<Real> &number = ruser_mesh_data[1];
    AthenaArray<Real> &dens_sum = ruser_mesh_data[2];
 //vr_sum = 0.0;
 //number = 0.0;
 //vr_out = vr_sum/number;
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
      //if (true) { // root level
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
      //if (Globals::my_rank == 0) {
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
      //} else {
      //MPI_Allreduce(&vr_sum, &vr_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      //MPI_Allreduce(&number, &number, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      //}
    //MPI_Allreduce(MPI_IN_PLACE, ruser_mesh_data[0].data(), 4*nPlanet, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
        //vr_out = vr_sum/number;
        //std::cout << "number: "<<number<<", vr_sum: "<< vr_sum << std::endl;
        //std::cout << "after MPI, vr_out: "<<vr_sum(ntot-1)/number(ntot-1)<<", number(-1): "<<number(ntot-1)<<
	//	", vr_sum(-1): "<< vr_sum(ntot-1) << std::endl;
        
        //std::cout << "After update: ruser_mesh_data: "<<ruser_mesh_data[0](0)<<", " << ruser_mesh_data[0](1)<< std::endl;

}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel, vis_vel_r;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        den = DenProfileCyl(rad,phi,z);
        vel = VelProfileCyl(rad,phi,z);
        Real cs_square = PoverRho(rad, phi, z);
        Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        Real vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        //vis_vel_r     = -1.5*(nu_iso/rad);
        if (porb->orbital_advection_defined)
          vel -= vK(porb, x1, x2, x3);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*vis_vel_r;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = den*vel;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = den*vel;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverRho(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

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

//----------------------------------------------------------------------------------------             
//! transform to spherical coordinate                                                               

void GetSphCoord(Coordinates *pco,Real &r,Real &theta,Real &phi,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    r=std::sqrt(pco->x1v(i)*pco->x1v(i)+pco->x3v(k)*pco->x3v(k));
    theta=atan2(pco->x1v(i),pco->x3v(k));
    phi=pco->x2v(j);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    r=pco->x1v(i);
    theta=pco->x2v(j);
    phi=pco->x3v(k);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel += 1.0 - rad*rad*rad/(rad*rad+z*z)/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

/* not used
Real HistoryAccretion(MeshBlock *pmb, int iout) {
  
  //int n = iout/4;
  int iout_loc = iout%4;
  //for (int n=0;n<nPlanet;n++ ) {
  return pmb->ruser_meshblock_data[0](iout_loc);
  //}
  //Real n = iout/3;
  //Real iout_loc = iout%3;
  //if (iout_loc==0){
  //  return PS[n].dmass;
  //} else if (iout_loc==1) { 
  //  return PS[n].dmvr;
  //} else if (iout_loc==2) { 
  //  return PS[n].dmvp;
  //}
  
}

not used */


Real PlanetAccretionHistory(MeshBlock *pmb,  int iout)
//const Real time, const Real dt, const AthenaArray<Real> &prim,
//    AthenaArray<Real> &cons 
 {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  //if (pmb->porb->orbital_advection_defined)
  //  phi_planet_move -= Omega0*time;
  Real rad(0.0), phi(0.0), z(0.0);
  Real GMp = gm0*Mp;
  Real pp_orbit_time = Pp*2.0*PI;
  Real time = pmb->pmy_mesh->time;
  Real dt = pmb->pmy_mesh->dt;
  Real e2 = SQR(epsilon);
  if (time < pp_orbit_time)
    GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  Real rp = 1.0-ecc*std::sin(time);
  AthenaArray<Real> &u = pmb->phydro->u;

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
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= t0_planet) {

          Real x_dis = rad*std::cos(phi) - rp*std::cos(ppos);
          Real y_dis = rad*std::sin(phi) - rp*std::sin(ppos);
          Real z_dis = z;
          //Real z_dis = z - z_planet;

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          distance        = std::sqrt(distance_square);
          //distance        = std::sqrt(d2);
          //distance = std::sqrt(SQR(x_dis) + SQR(y_dis));
          //cout << "d2: "<< d2 << endl;
          //cout << "distance_square: "<< distance_square << endl;

          //if ((distance > rad_soft) && (distance <= accretion_radius)) {
          //distance_temp = std::min(distance,distance_temp);
          if (distance <= racc) {
            //Real time_freefall  = distance_square*distance*inv_sqrt2gmp;
            //Real remove_percent = -rate*std::max(dt/time_freefall, 1.0);
            Real omega = std::sqrt(gm0/rp/rp/rp);
            Real facc   = std::min(rate*dt*omega, 0.8);
            Real vol = pmb->pcoord->GetCellVolume(k, j, i);

            const Real gas_rho = u(IDN,k,j,i);
            const Real gas_mom1 = u(IM1,k,j,i);
            const Real gas_mom2 = u(IM2,k,j,i);
            const Real gas_mom3 = u(IM2,k,j,i);
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

            Real delta_gas_dens = facc*gas_rho;
            Real delta_gas_mom1 = delta_gas_dens*gas_vel1;
            Real delta_gas_mom2 = delta_gas_dens*gas_vel2;
            Real delta_gas_mom3 = delta_gas_dens*gas_vel3;

            deltaM   += delta_gas_dens*vol;
            deltaMp1 += delta_gas_mom1*vol;
            deltaMp2 += delta_gas_mom2*vol;
            deltaMp3 += delta_gas_mom3*vol;

            //cout << "deltaM: "<< deltaM << endl;
            //cout << "facc: "<< facc << ", gas_rho: "<< gas_rho << ", time"<< time << endl;
            //if (deltaM <= 0.0)
               //cout << "deltaM: "<< deltaM << endl;
            //   cout << "in loop: deltaM: "<< deltaM << ", distance: " << distance << ", racc: " << racc << endl;


            //cout << "... accretion: delta gas_dens: " << delta_gas_dens << ", facc:" << facc << ", gas_rho:" << gas_rho << endl;
            //cout << "... after accretion: gas_dens: " << cons(IDN, k, j, i) << ", " << prim(IDN, k, j, i) << endl;


          } // if distance
        // } // if distance
        // } // if t0
      }
    }
  }
  // only one planet
  //pmb->ruser_meshblock_data[0](0) = deltaM;
  //pmb->ruser_meshblock_data[0](1) = deltaMp1;
  //pmb->ruser_meshblock_data[0](2) = deltaMp2;
  //pmb->ruser_meshblock_data[0](3) = deltaMp3;

  //if (deltaM >= 0.0)
  //   cout << "refine: "<< refine << endl;
  //   //cout << "deltaM: "<< deltaM << endl;
  //   cout << "... deltaM: "<< deltaM << ", time: " << time << ", ruser0: " << pmb->ruser_meshblock_data[0](0)  << endl;

  return deltaM;
 }

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, 
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
          //prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          Real cs_square = PoverRho(rad, phi, z);
          Real cs_square_in = PoverRho(rad_in, phi_in, z_in);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          Real omega_dyn_in = std::sqrt(gm0/(rad_in*rad_in*rad_in));
          Real nu     = nu_alpha*cs_square/omega_dyn;
          Real nu_in     = nu_alpha*cs_square_in/omega_dyn_in;
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il)*nu_in/nu;
          //prim(IDN,k,j,il-i) = dfloor;
          vel = VelProfileCyl(rad,phi,z);
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
          vel = VelProfileCyl(rad,phi,z);
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

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
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
          //prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          vel_rout = VelProfileCyl(rad_out,phi_out,z_out);
          Real cs_square = PoverRho(rad, phi, z);
          Real cs_square_rout = PoverRho(rad_out, phi_out, z_out);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          Real omega_dyn_rout = std::sqrt(gm0/(rad_out*rad_out*rad_out));
	  Real nu = nu_alpha*cs_square/omega_dyn;
	  Real nu_rout = nu_alpha*cs_square_rout/omega_dyn_rout;
          Real lmom = vel*rad;
          Real lmom_rout = vel_rout*rad_out;
          //Real den = prim(IDN,k,j,iu)*nu_rout*lmom_rout/nu/lmom+
          //    (lmom-lmom_rout)*DenProfileCyl(rad,phi,z)/lmom;
          Real den = prim(IDN,k,j,iu)*nu_rout*lmom_rout/nu/lmom+
              (lmom-lmom_rout)*DenProfileCyl(rad,phi,z)/lmom;
          prim(IDN,k,j,iu+i) = den;
          //prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu)*std::pow(rad/rad_out,-3.0+1.5*pslope)+
          //    (std::sqrt(1.0)-std::sqrt(rad_out/rad))*prim(IDN,k,j,iu);
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn)*
		DenProfileCyl(rad,phi,z)/den;
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
          //prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);

          vel = VelProfileCyl(rad,phi,z);
          vel_rout = VelProfileCyl(rad_out,phi_out,z_out);
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
          //    (lmom-lmom_rout)*DenProfileCyl(rad,phi,z)/lmom;
          Real den = prim(IDN,k,j,iu)*std::pow(rad/rad_out,-3.0+1.5*pslope)*lmom_rout/lmom+
              (lmom-lmom_rout)*DenProfileCyl(rad,phi,z)/lmom;
          vis_vel_r     = -1.5*(nu_alpha*cs_square/rad/omega_dyn)*
		DenProfileCyl(rad,phi,z)/den;
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

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
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
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, 
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
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
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
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
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
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
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



void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df, 
              const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, 
              AthenaArray<Real> &cons_scalar){
  Real dist;
  Real x1, x2, x3;
  Real g1(0.0), g2(0.0), g3(0.0);
  Real r(0.0), theta(0.0), phi(0.0);
  for(int k=pmb->ks; k<=pmb->ke; ++k) {
    x3 = pmb->pcoord->x3v(k);
    for(int j=pmb->js; j<=pmb->je; ++j) {
      x2 = pmb->pcoord->x2v(j);
      for(int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        x1 = pmb->pcoord->x1v(i);
        GetSphCoord(pmb->pcoord,r,theta,phi,i,j,k);  

        // planet gravity
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          g1 = PlanetGravityR(r,theta,phi,time)*std::sin(theta)
              +PlanetGravityTheta(r,theta,phi,time)*std::cos(theta);
          g2 = PlanetGravityPhi(r,theta,phi,time);
          g3 = PlanetGravityR(r,theta,phi,time)*std::cos(theta) 
              -PlanetGravityTheta(r,theta,phi,time)*std::sin(theta);
        }
        else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
          g1 = PlanetGravityR(r,theta,phi,time);
          g2 = PlanetGravityTheta(r,theta,phi,time);
          g3 = PlanetGravityPhi(r,theta,phi,time);
        }  
        cons(IM1,k,j,i) += dt*den*g1;
        cons(IM2,k,j,i) += dt*den*g2;
        if (pmb->pmy_mesh->mesh_size.nx3 > 1)
          cons(IM3,k,j,i) += dt*den*g3;
        if (NON_BAROTROPIC_EOS){
          cons(IEN,k,j,i) += prim(IVX,k,j,i)*dt*den*g1;
          cons(IEN,k,j,i) += prim(IVY,k,j,i)*dt*den*g2;
          if (pmb->pmy_mesh->mesh_size.nx3 > 1)
            cons(IEN,k,j,i) += prim(IVZ,k,j,i)*dt*den*g3;
        }
      }
    }
  }
  if ((Mp > 0.0) && Accretion_Flag) {
    //std::cout << "Accretion_Flag =1: " << Accretion_Flag << std::endl;
    //PlanetAccretion(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
    PlanetAccretion(pmb, time, dt, prim, prim_df, prim_scalar, bcc, cons, cons_scalar);
  }

  if (Damping_Flag==1) {
    //std::cout << "Damping_Flag =1: " << Damping_Flag << std::endl;
    InnerWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  } else if (Damping_Flag==2) {
    //std::cout << "Damping_Flag =2: " << Damping_Flag << std::endl;
    OuterWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  } else if (Damping_Flag==3) {
    //std::cout << "Damping_Flag =3: " << Damping_Flag << std::endl;
    InnerWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
    OuterWavedamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  }

  if (Isothermal_Flag && NON_BAROTROPIC_EOS) {
    //std::cout << "Isothermal_Flag =1: " << Isothermal_Flag << std::endl;
    LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_scalar, bcc, cons, cons_scalar);
  } else if (beta > 0.0) {
    //std::cout << "Isothermal_Flag =0: " << Isothermal_Flag << std::endl;
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_scalar, bcc, cons, cons_scalar);
  }

  return;
}


Real PlanetGravityR(Real x1, Real x2, Real x3, Real time)
{
  Real pgv1;
  Real d2;
  Real GMp = gm0*Mp;
  Real pp_orbit_time = Pp*2.0*PI;
  Real e2 = SQR(epsilon);
  if (time < pp_orbit_time)
    GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  Real rp = 1.0-ecc*std::sin(time);
  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //   d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::cos(x2-ppos)+SQR(x3);
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
     d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
  //}
  Real temp = GMp/((d2+e2)*std::sqrt(d2+e2));
  if (gorder == 4)
    temp *= (d2+2.5*e2)/(d2+e2);

  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //pgv1 = -temp*(x1-rp*std::cos(x2-ppos));
  //if (IndirectTerm) {
  //  temp = GMp/SQR(rp);
  //  pgv1 += -temp*std::cos(x2-ppos);
  // }
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
  pgv1 = -temp*(x1-rp*std::sin(x2)*std::cos(x3-ppos));
  if (IndirectTerm) {
    temp = GMp/SQR(rp);
    pgv1 += -temp*std::sin(x2)*std::cos(x3-ppos);
   }
  //}
  return pgv1;
}

Real PlanetGravityTheta(Real x1, Real x2, Real x3, Real time)
{
  Real pgv2;
  Real d2;
  Real GMp = gm0*Mp;
  Real pp_orbit_time = Pp*2.0*PI;
  Real e2 = SQR(epsilon);
  if (time < pp_orbit_time)
    GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  Real rp = 1.0-ecc*std::sin(time);
  Real zp = 0.0;
  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //   d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::cos(x2-ppos)+SQR(x3);
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
     d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
  //}
  Real temp = GMp/((d2+e2)*std::sqrt(d2+e2));
  if (gorder == 4)
    temp *= (d2+2.5*e2)/(d2+e2);

  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //pgv2 =  temp*(x3-zp);
  //if (IndirectTerm) {
  //  temp = GMp/SQR(rp);
  //  pgv2 += -temp*zp/rp;
  // }
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
  pgv2 =  temp*rp*std::cos(x2)*std::cos(x3-ppos);
  if (IndirectTerm) {
    temp = GMp/SQR(rp);
    pgv2 += -temp*std::cos(x2)*std::cos(x3-ppos);
   }
  //}
  return pgv2;
}

Real PlanetGravityPhi(Real x1, Real x2, Real x3, Real time)
{ 
  Real pgv3;
  Real d2;
  Real GMp = gm0*Mp;
  Real pp_orbit_time = Pp*2.0*PI;
  Real e2 = SQR(epsilon);
  if (time < pp_orbit_time)
    GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  Real rp = 1.0-ecc*std::sin(time);
  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //   d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::cos(x2-ppos)+SQR(x3);
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
     d2 = SQR(x1)+SQR(rp)-2.0*rp*x1*std::sin(x2)*std::cos(x3-ppos);
  //}
  Real temp = GMp/((d2+e2)*std::sqrt(d2+e2));
  if (gorder == 4)
    temp *= (d2+2.5*e2)/(d2+e2);

  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
  //pgv3 = -temp*rp*std::sin(x2-ppos);
  //if (IndirectTerm) {
  //  temp = GMp/SQR(rp);
  //  pgv3 += temp*std::sin(x2-ppos);
  // }
  //} else if (std::strcmp(COORDINATE_SYSTEM,"spherical_polar") == 0) {
  pgv3 = -temp*rp*std::sin(x3-ppos);
  if (IndirectTerm) {
    temp = GMp/SQR(rp);
    pgv3 += temp*std::sin(x3-ppos);
   }
  //}
  return pgv3;
}

int SphRefinementCondition(MeshBlock *pmb)
{
  int refine = 0;
  Real time = pmb->pmy_mesh->time;
  Real rp = 1.0-ecc*std::sin(time);
  Real ptheta = PI/2.0;
  Real ppos = fmod(time*(1.0-Omega0)-2.0*ecc*std::cos(time),2.0*PI);
  Real x1, x2, x3;
  Real dx1, dx2, dx3;
  for(int k=pmb->ks; k<=pmb->ke; ++k) {
    x3 = pmb->pcoord->x3v(k);
    dx3 = pmb->pcoord->dx3f(k);
    if ((ppos < 0.5*PI) && (x3 > 1.5*PI))
      x3 -= 2.0*PI;
    if ((ppos > 1.5*PI) && (x3 < 0.5*PI))
      x3 += 2.0*PI;
    for(int j=pmb->js; j<=pmb->je; ++j) {
      x2 = pmb->pcoord->x2v(j);
      dx2 = pmb->pcoord->dx2f(j);
      for(int i=pmb->is; i<=pmb->ie; ++i) {
        x1 = pmb->pcoord->x1v(i);
        dx1 = pmb->pcoord->dx1f(i);
        // if cell size > refine area, refine cells closest to planet
        if ((dx1 > area) || (rp*dx2 > area) || (rp*std::sin(x2)*dx3 > area)){
          if(std::fabs(x1-rp) < dx1){
            if(std::fabs(x2-ptheta) < dx2){
              if(std::fabs(x3-ppos) < dx3){
                refine = 1;
              }
            }
          }
        }
        // dr < area
        if(std::fabs(x1-rp) < area){
          // r*dtheta < area
          if(std::fabs(rp*(x2-ptheta)) < area){
            // r*sin(theta)*dphi < area 
            if(std::fabs(rp*std::sin(ptheta)*(x3-ppos)) < area){
              if(dx1 > res)
                refine = 1;
              if(x1*dx2 > res)
                refine = 1;
              if(x1*std::sin(x2)*dx3 > res)
       	        refine = 1;
            }
          }
        }
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


int CylRefinementCondition(MeshBlock *pmb){
  int refine = 0;
  Real time = pmb->pmy_mesh->time;
  Real rp = 1.0-ecc*std::sin(time);
  Real ptheta = PI/2.0;
  Real ppos = fmod(time*(1.0-Omega0)-2.0*ecc*std::cos(time),2.0*PI);
  Real zp = rp*std::cos(ptheta);
  Real sp = std::sqrt(rp*rp - zp*zp);
  Real x1, x2, x3;
  Real dx1, dx2, dx3;
  for(int k=pmb->ks; k<=pmb->ke; ++k) {
    x3 = pmb->pcoord->x3v(k);
    dx3 = pmb->pcoord->dx3f(k);
    for(int j=pmb->js; j<=pmb->je; ++j) {
      x2 = pmb->pcoord->x2v(j);
      if ((ppos < 0.5*PI) && (x2 > 1.5*PI))
        x2 -= 2.0*PI;
      if ((ppos > 1.5*PI) && (x2 < 0.5*PI))
        x2 += 2.0*PI;
      dx2 = pmb->pcoord->dx2f(j);
      for(int i=pmb->is; i<=pmb->ie; ++i) {
        x1 = pmb->pcoord->x1v(i);
        dx1 = pmb->pcoord->dx1f(i);
        // if cell size > refine area, refine cells closest to planet         
        if ((dx1 > area) || (sp*dx2 > area) || (dx3 > area)){
          if(std::fabs(x1-sp) < dx1){
            if(std::fabs(x2-ppos) < dx2){
              if(std::fabs(x3-zp) < dx3){
		std::cout << x1 << " " << x2 << " " << x3 << std::endl;
                refine = 1;
              }
            }
          }
        }
        // dr < area
        if(std::fabs(x1-sp) < area){
          // r*dtheta < area
          if(std::fabs(rp*(x2-ppos)) < area){
            // r*sin(theta)*dphi < area
            if(std::fabs(x3-zp) < area){
              if(dx1 > res)
                refine = 1;
              if(x1*dx2 > res)
                refine = 1;
              if(dx3 > res)
                refine = 1;
            }
          }
        }
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

int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;
  Real d2;
  Real time = pmb->pmy_mesh->time;
  // planet positions
  Real px1 = 1.0-ecc*std::sin(time);
  Real px2 = PI/2.0;
  Real px3 = fmod(time*(1.0-Omega0)-2.0*ecc*std::cos(time),2.0*PI);
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
            d2 = SQR(x1)+SQR(px1)-2.0*px1*x1*std::cos(x2-px2) + SQR(x3-px3);
        }
        else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            d2 = SQR(x1)+SQR(px1)-2.0*px1*x1*std::sin(x2)*std::cos(x3-px3);
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


// Mass Remove within Hill
void PlanetAccretion(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //Real phi_planet_move = omega_planet*time + phi_planet_0;
  //if (pmb->porb->orbital_advection_defined)
  //  phi_planet_move -= Omega0*time;
  Real rad(0.0), phi(0.0), z(0.0);
  Real GMp = gm0*Mp;
  Real pp_orbit_time = Pp*2.0*PI;
  Real e2 = SQR(epsilon);
  if (time < pp_orbit_time)
    GMp *= SQR(std::sin(0.5*PI*time/pp_orbit_time));
  Real ppos = time*(1.0-Omega0)-2.0*ecc*std::cos(time);
  Real rp = 1.0-ecc*std::sin(time);

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
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // Real t_growth  = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
        //if (time >= t0_planet) {

          Real x_dis = rad*std::cos(phi) - rp*std::cos(ppos);
          Real y_dis = rad*std::sin(phi) - rp*std::sin(ppos);
          Real z_dis = z;
          //Real z_dis = z - z_planet;

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //Real distance        = std::sqrt(distance_square);
          distance        = std::sqrt(distance_square);
          //distance        = std::sqrt(d2);
          //distance = std::sqrt(SQR(x_dis) + SQR(y_dis));
          //cout << "d2: "<< d2 << endl;
          //cout << "distance_square: "<< distance_square << endl;

          //if ((distance > rad_soft) && (distance <= accretion_radius)) {
          //distance_temp = std::min(distance,distance_temp);
          if (distance <= racc) {
            //Real time_freefall  = distance_square*distance*inv_sqrt2gmp;
            //Real remove_percent = -rate*std::max(dt/time_freefall, 1.0);
            Real omega = std::sqrt(gm0/rp/rp/rp);
            Real facc   = std::min(rate*dt*omega, 0.8);
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

            Real delta_gas_dens = facc*gas_rho;
            Real delta_gas_mom1 = delta_gas_dens*gas_vel1;
            Real delta_gas_mom2 = delta_gas_dens*gas_vel2;
            Real delta_gas_mom3 = delta_gas_dens*gas_vel3;

            deltaM   += delta_gas_dens*vol;
            deltaMp1 += delta_gas_mom1*vol;
            deltaMp2 += delta_gas_mom2*vol;
            deltaMp3 += delta_gas_mom3*vol;

            // only one planet
            //pmb->ruser_meshblock_data[0](0) = deltaM;
            //pmb->ruser_meshblock_data[0](1) = deltaMp1;
            //pmb->ruser_meshblock_data[0](2) = deltaMp2;
            //pmb->ruser_meshblock_data[0](3) = deltaMp3;
            //cout << "deltaM: "<< deltaM << endl;
            //cout << "facc: "<< facc << ", gas_rho: "<< gas_rho << ", time"<< time << endl;
            //if (deltaM <= 0.0)
               //cout << "deltaM: "<< deltaM << endl;
            //   cout << "in loop: deltaM: "<< deltaM << ", distance: " << distance << ", racc: " << racc << endl;

            gas_dens -= delta_gas_dens;
            gas_mom1 -= delta_gas_mom1;
            gas_mom2 -= delta_gas_mom2;
            gas_mom3 -= delta_gas_mom3;

            //cout << "... accretion: delta gas_dens: " << delta_gas_dens << ", facc:" << facc << ", gas_rho:" << gas_rho << endl;
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



//----------------------------------------------------------------------------------------
// Wavedamping function
//
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar) {

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
            R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad);

            Real gas_rho_0    = DenProfileCyl(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl(rad, phi, z);
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
              //Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                  + SQR(gas_mom3))*inv_dens_gas;
              //gas_pre           = internal_erg*(gamma_gas - 1.0);
              gas_pre     = cons(IPR, k, j, i);
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
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                                   + SQR(gas_mom3))*inv_dens_gas;
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
            R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/omega_dyn/rad);

            Real gas_rho_0    = DenProfileCyl(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl(rad, phi, z);
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
              //Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                  + SQR(gas_mom3))*inv_dens_gas;
              //gas_pre           = internal_erg*(gamma_gas - 1.0);
              gas_pre     = prim(IPR, k, j, i);
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
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                                   + SQR(gas_mom3))*inv_dens_gas;
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
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar) {

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
            R_func      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

            //Real gas_rho_0    = DenProfileCyl(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl(rad, phi, z);
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
              //Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              //gas_pre           = internal_erg*(gamma_gas - 1.0);
              gas_pre           = prim(IPR, k, j, i);
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
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                                        + SQR(gas_mom3))*inv_dens_gas;
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
            R_func      = SQR((rad - radius_outer_damping)*inv_outer_damp);
            damping_tau = 1.0/(damping_rate*omega_dyn);

            Real cs_square = PoverRho(rad, phi, z);
            //Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

            //Real gas_rho_0    = DenProfileCyl(rad, phi, z);

            Real vel_gas_phi  = VelProfileCyl(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            //Real gas_vel1_0 = vis_vel_r;
            //Real gas_vel1_0 = vr_out;
            Real vr0 = pmb->pmy_mesh->ruser_mesh_data[0](tj,ti);
            Real num0 = pmb->pmy_mesh->ruser_mesh_data[1](tj,ti);
            Real dens0 = pmb->pmy_mesh->ruser_mesh_data[2](tj,ti);
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
              //Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              //gas_pre           = internal_erg*(gamma_gas - 1.0);
              gas_pre           = prim(IPR, k, j, i);
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
              //gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
              //                                        + SQR(gas_mom3))*inv_dens_gas;
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



void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar) {

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
        //Real press        = PoverRho(rad, phi, z)*gas_dens;
        //gas_erg           = press*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
        Real press        = PoverRho(rad, phi, z)*gas_rho;
        gas_erg           = press*igm1 + 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3))*gas_rho;
      }
    }
  }
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_scalar) {

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
