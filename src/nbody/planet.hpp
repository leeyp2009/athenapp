//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers
//
// C++ headers
#include <iostream>

// Athena++ headers
#include "../athena.hpp"

#ifndef PI
#define PI 3.14159265358979323846264 
#endif

//#ifndef real
//#define real double
//#endif

using namespace std;

class Planet {
public: 
  Planet() : r(1.0), phi(PI), mass(1.0e-3) {} 
  Planet(Real r0,Real phi0,Real mass0) : r(r0), phi(phi0), mass(mass0), ecc(0.0), 
         vr(0.0), vp(1.0), index(0), theta(PI/2.0), vt(0.0),
         fr(0.0), fp(0.0), ft(0.0),
	 feelOthers(1), feelDisk(0) {}
  ~Planet() {}

  void setMass(Real mass0) { mass = mass0;}
  Real getMass() { return mass;}
  void setRad(Real r0)  { r= r0;}
  Real getRad() { return r;}
  void setPhi(Real phi0)   { phi = phi0;}
  Real getPhi() { return phi;}
  void setTheta(Real theta0)   { theta = theta0;}
  Real getTheta() { return theta;}
  void setVr(Real vr0) {vr = vr0;}
  Real getVr() { return vr;}
  void setVp(Real vp0) {vp = vp0;}
  Real getVp() { return vp;}
  void setVt(Real vt0) {vt = vt0;}
  Real getVt() { return vt;}

  Real getFr() { return fr;}
  void setFr(Real fr0) { fr = fr0; }
  Real getFp() { return fp;}
  void setFp(Real fp0) { fp = fp0;}
  Real getFt() { return ft;}
  void setFt(Real ft0) { ft = ft0;}

  void setEcc(Real ecc0) { ecc = ecc0;}
  Real getEcc() {return ecc;}
  void setInc(Real inc0)   { inc = inc0;}
  Real getInc() { return inc;}
  void setIndex(int index0) { index = index0;}
  int getIndex() {return index;}
  void setFeelDisk(bool feelDisk0) {feelDisk = feelDisk0;}

//  int Embedded_Verner_7_8(const Real y[],
//			const int n,
//			const Real& x,
//			const Real& h,
//			Real  yout[],
//			void (*derivs)(const double&, 
//				  const double [],
//				  Real [])); 
//  void Runge_Kutta(void (*f)(const double&,
//			const double [],
//			double []), 
//			const double *y, 
//			const double& x,
//			double yout[],
//			double [],
//			const double& h,
//			const int n,
//			double *tmpArray);
  
  void WriteFile(ofstream& outf, const Real& time) {
     static long ncount = 0;
     //std::setprecision(10);
     outf << std::setprecision(10)  << time << " "<< r << " "
                 << phi << " " << theta <<" "
                 << vr << " " << vp <<" "
                 << vt << " " << fr <<" "
                 << fp << " " << ft <<" "
                 << mass <<" "
                 <<  endl;
     ncount++;
     if (ncount%20) outf << flush; 
     //std::setprecision(6);
  }

  void initializeRK(Real *y) // initialize for the RK integration 
  {
   y[0] =r; y[1] = phi; y[2] = theta; y[3] = vr;  //angular velcoity: omega
   y[4] =vp/(r*sin(theta)); y[5] = vt/r;  //
   //cout << "r, phi, vr, omega:"<< y[0]<<", " 
   //    << y[1] <<", " << y[2] << ", " 
   //    << y[3] <<endl;
  }
  
  void update(const Real *y) 
  {
   r =y[0]; phi = y[1]; theta = y[2]; vr = y[3];  //velcoity
   vp =y[4]*r*sin(theta); vt = y[5]*r;  //velcoity
   //cout << "update r, phi, vr, omega:"<< r <<", " 
   //    << phi <<", " << vr << ", " 
   //    << vp <<endl;
  }

  void GravityFromConfig(const Real& OMEGA, Real* facc, const bool FIX_PHI){
      //fr = 0.0;
      //fp = 0.0;
      Real fr_p, fp_p;
      if (feelDisk==1) {
         fr_p = fr;
         fp_p = fp;
      } else {
         fr_p = 0.0;
         fp_p = 0.0;
      }
      if (FIX_PHI) {
      Real rcom = 1.0;
      facc[0] = (vp*vp+vt*vt)/rcom - (1.0+mass)/(rcom*rcom)  + fr_p + (OMEGA*OMEGA*rcom + 2.0*OMEGA*vp)*sin(theta); // r dir: ddot_r
      //facc[1] = (-2.0*vt*vr/rcom + vp*vp/rcom*cos(theta)/sin(theta)+
      //	OMEGA*OMEGA*rcom)/rcom;  //phi dir: ddot_phi; 
      facc[2]= (-2.0*vt*vr/rcom  + vp*vp/rcom*cos(theta)/sin(theta) + 
		 (OMEGA*OMEGA*rcom + 2.0*OMEGA*vp)*cos(theta))/rcom;  // theta-dir
      //facc[2] = (-2.0*vp*vr/rcom - 2.0*OMEGA*vr)/rcom;  //phi dir: ddot_phi; 
      facc[1] = (-2.0*vp*vr/rcom + fp_p/rcom - 2.0*vp*vt/rcom*cos(theta)/sin(theta)-
		2.0*OMEGA*vr)/(rcom*sin(theta));  //phi dir: ddot_phi; 
      //facc[0] = vp*vp/r - (1.0+mass)/(r*r)  + OMEGA*OMEGA + 2.0*OMEGA*vp; // r dir: ddot_r
      //facc[1] = (-2.0*vp*vr/r/r - 2.0*OMEGA*vr);  //phi dir: ddot_phi; 
      } else {
      facc[0] = (vp*vp+vt*vt)/r + fr_p - (1.0+mass)/(r*r)  + (OMEGA*OMEGA*r + 2.0*OMEGA*vp)*sin(theta); // r dir: ddot_r
      facc[2]= (-2.0*vt*vr/r   + vp*vp/r*cos(theta)/sin(theta) + 
		 (OMEGA*OMEGA*r + 2.0*OMEGA*vp)*cos(theta))/r;  // theta-dir
      facc[1] = (-2.0*vp*vr/r + fp_p/r - 2.0*vp*vt/r*cos(theta)/sin(theta)-
		2.0*OMEGA*vr)/(r*sin(theta));  //phi dir: ddot_phi; 
      }
  }

  void GravityFromPlanet(Planet& p, Real* facc){ // Gravity forces from another planet
    if (this->index !=p.index) {
       Real distance, dis3;
       Real tmp;
       Real tmp2 = sin(theta)*sin(p.theta)*cos(phi-p.phi)+cos(theta)*cos(p.theta);
       Real tmp1 = 2.0*r*p.r*tmp2;
       Real tmp3 = cos(theta)*sin(p.theta)*cos(phi-p.phi)-sin(theta)*cos(p.theta);
       
       //distance = sqrt(r*r+p.r*p.r - 2.0*r*p.r*cos(phi-p.phi));
       distance = sqrt(r*r+p.r*p.r - tmp1);
       dis3 = 1.0/pow(distance,3);

       tmp = p.r*dis3 - 1.0/p.r /p.r;
       //facc[0] += -p.mass*(r*dis3 - cos(phi-p.phi)*tmp);  
       //facc[1] += ; // theta 
       //facc[2] += -p.mass*(sin(phi-p.phi)*tmp)/r; // should divided by r since this is ddot_phi not ddot_vp?
       facc[0] += -p.mass*(r*dis3 - tmp2*tmp);  
       facc[2] += -p.mass*tmp*tmp3; // theta 
       facc[1] += -p.mass*(sin(phi-p.phi)*tmp)*sin(theta)/r; // should divided by r since this is ddot_phi not ddot_vp?
       }

  }

  int NperHydro (const Real& dt, const Real& dr, const Real& dphi) {
      int NT = int(20*dt*sqrt(vr*vr+vp*vp)/std::min(dr,r*dphi)+0.6);
      return NT;
  }

  Real distance(Planet& p) {
      //Real cosphi = cos(phi-p.phi);
      //Real cosphi = cos(phi)*cos(p.phi) + sin(phi)*sin(p.phi);
      //Real dis = sqrt(r*r + p.r*p.r - 2.0*p.r*r*cosphi);
      Real tmp2 = sin(theta)*sin(p.theta)*cos(phi-p.phi)+cos(theta)*cos(p.theta);
      Real tmp1 = 2.0*r*p.r*tmp2;
      Real dis = sqrt(r*r+p.r*p.r - tmp1);
      return dis;
  }

private:
  Real mass;
  Real r;
  Real phi;
  Real theta;
  Real ecc;
  Real inc;
  Real vp;
  Real vr;  
  Real vt;  
  Real fp;
  Real fr;  
  Real ft;  
  int index;
  bool feelDisk; // feel disk force
  bool feelOthers; // feel other planets in a system
};

