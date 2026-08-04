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
#include <algorithm>  // min
#include <cmath>   // sqrt
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
//
////////////////////////////////////////////////////////////////////////////////
// File: embedded_verner_7_8.c                                                //
// Routines:                                                                  //
//    Embedded_Verner_7_8                                                     //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Description:                                                              //
//     The Runge-Kutta-Verner method is an adaptive procedure for approxi-    //
//     mating the solution of the differential equation y'(x) = f(x,y) with   //
//     initial condition y(x0) = c.  This implementation evaluates f(x,y)     //
//     thirteen times per step using embedded seventh order and eight order   //
//     Runge-Kutta estimates to estimate the not only the solution but also   //
//     the error.                                                             //
//     The next step size is then calculated using the preassigned tolerance  //
//     and error estimate.                                                    //
//     For step i+1,                                                          //
//        y[i+1] = y[i] +  h * (13/288 * k1 + 32/125 * k6 + 31213/144000 * k7 //
//        + 2401/12375 * k8 + 1701/14080 * k9 + 2401/19200 k10 + 19/450 k11 ) //
//     where                                                                  //
//     k1 = f( x[i],y[i] ),                                                   //
//     k2 = f( x[i]+h/4, y[i] + h*k1/4),                                      //
//     k3 = f( x[i]+h/12, y[i]+h/72*(5 k1 + k2) ),                            //
//     k4 = f( x[i]+h/8, y[i]+h/32*( k1 + 3 k3) ),                            //
//     k5 = f( x[i]+2h/5, y[i]+h/125*(106 k1 - 408 k3 + 352 k4)),             //
//     k6 = f( x[i]+h/2, y[i]+h*( k1/48 + 8 k4 / 33 + 125 k5/528 ) ),         //
//     k7 = f( x[i]+6h/7, y[i]+h/26411*( -13893 k1 + 39936 k4 - 64125 k5      //
//                                                            + 60720 k6 ) ), //
//     k8 = f( x[i]+h/7, y[i]+h*( 37/392 k1 + 1625/9408 k5 - 2/15 k6          //
//                                                           + 61/6720 K7) )  //
//     k9 = f( x[i]+2h/3, y[i]+h*( 17176/25515 k1 - 47104/25515 k4            //
//        + 1325/504 k5 - 41792/25515 k6 + 20237/145800 k7 + 4312/6075 k8) ), //
//     k10 = f( x[i]+2h/7, y[i]+h*( -23834/180075 k1 - 77824/1980825 k4       //
//             - 636635/633864 k5 + 254048/300125 k6 - 183/7000 k7 + 8/11 K8  //
//                                                 - 324/3773 k9) )           //
//     k11 = f( x[i]+h, y[i]+h*( 12733/7600 k1 - 20032/5225 k4                //
//                + 456485/80256 k5 - 42599/7125 k6 + 339227/912000 k7        //
//                          - 1029/4108 K8 + 1701/1408 k9 + 5145/2432 k10) )  //
//     k12 = f( x[i]+h/3, y[i]+h*( -27061/204120 k1 + 40448/280665 k4         //
//                   - 1353775/1197504 k5 + 17662/25515 k6 - 71687/1166400 k7 //
//                        + 98/225 K8 + 1/16 k9 + 3773/11664 k10) )           //
//     k13 = f( x[i]+h, y[i]+h*( 11203/8680 k1 - 38144/11935 k4               //
//             + 2354425/458304 k5 - 84046/16275 k6 + 673309/1636800 k7       //
//             + 4704/8525 K8 + 9477/10912 k9 - 1029/992 k10 + 19/341 k12) )  //
//     x[i+1] = x[i] + h.                                                     //
//                                                                            //
//     The error is estimated to be                                           //
//        err = h*( 6600 k1 + 135168 k6 + 14406 k7 - 57624 k8 - 54675 k9      //
//         + 396165 k10 + 133760 k11 - 437400 k12 - 136400 k13) / 3168000     //
//     The step size h is then scaled by the scale factor                     //
//         scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     //
//     The scale factor is further constrained 0.125 < scale < 4.0.           //
//     The new step size is h := scale * h.                                   //
////////////////////////////////////////////////////////////////////////////////


using namespace std;

//#define real double
//#define real double

//#define max(x,y) ( (x) < (y) ? (y) : (x) )
//#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 10
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

static void Runge_Kutta(void (*f)(const Real&,
			const Real [],
			Real []), 
			const Real *y, 
			const Real& x,
			Real yout[],
			Real [],
			const Real& h,
			const int n,
			Real *tmpArray);

////////////////////////////////////////////////////////////////////////////////
// int Embedded_Verner_7_8( Real (*f)(Real, Real), Real y[],          //
//       Real x, Real h, Real xmax, Real *h_next, Real tolerance )  //
//                                                                            //
//  Description:                                                              //
//     This function solves the differential equation y'=f(x,y) with the      //
//     initial condition y(x) = y[0].  The value at xmax is returned in y[1]. //
//     The function returns 0 if successful or -1 if it fails.                //
//                                                                            //
//  Arguments:                                                                //
//     Real *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y0) corresponding to the //
//                initial condition y(x0) = y0.                               //
//     Real y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at xmax.                               //
//     Real x   The initial value of x.                                     //
//     Real h   Initial step size.                                          //
//     Real xmax The endpoint of x.                                         //
//     Real *h_next   A pointer to the estimated step size for successive   //
//                      calls to Embedded_Verner_7_8.                         //
//     Real tolerance The tolerance of y(xmax), i.e. a solution is sought   //
//                so that the relative error < tolerance.                     //
//                                                                            //
//  Return Values:                                                            //
//     0   The solution of y' = f(x,y) from x to xmax is stored y[1] and      //
//         h_next has the value to the next size to try.                      //
//    -1   The solution of y' = f(x,y) from x to xmax failed.                 //
//    -2   Failed because either xmax < x or the step size h <= 0.            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Embedded_Verner_7_8(const Real y[],
			const int n,
			const Real& x0,
			const Real& hmax,
			Real  yout[],
			void (*f)(const Real&, 
				  const Real [],
				  Real [])) 
{

  static const Real err_exponent = 1.0 / 7.0;

  Real scale;
  Real err;
  Real yy;
  Real *temp_y = new Real[n];
  Real *errArr = new Real[n];
  Real *tmpArray = new Real[n*14];
  int i;
  int last_interval = 0;

  Real h;  // initial time step
  Real x = x0;
  static Real h_next;
  Real xmax = x+hmax;
  static Real tol0 = 1e-8;
  Real tolerance;
  static bool first = 1;

  if (first) {
    h = 0.2*hmax;
    first = 0;
    tol0 = min(1e-8, yout[0]);
  } else if (y[n] > 0.5) {
    h = hmax;
  } else {
    h = h_next;
  }


  // Insure that the step size h is not larger than the length of the //
  // integration interval.                                            //
  
  if (h > (xmax - x) ) { h = xmax - x; last_interval = 1;}

  // Redefine the error tolerance to an error tolerance per unit    //
  // length of the integration interval.                            //

  tolerance = tol0/(xmax - x);

  // Integrate the diff eq y'=f(x,y) from x=x to x=xmax trying to  //
  // maintain an error less than tolerance * (xmax-x) using an     //
  // initial step size of h and initial value: y = y[0]            //

  for (int i =0; i<n; i++) temp_y[i] = y[i]; 
  
  int niter = 0;
  Real hmin = (xmax-x)/100.0;
  while ( x < xmax ) {
    scale = 1.0;
    niter++;
    //cout << "niter:" <<niter << endl;
    for (i = 0; i < ATTEMPTS; i++) {
      Runge_Kutta(f,temp_y, x, yout, errArr, h, n, tmpArray);
      //cout << "yout:" <<yout[0] << endl;
      err = 0.0;
      for (int j =0; j<n; j++) { // Yaping: should be a new index j rather than i?
	if (errArr[j] == 0.0) continue;
	//Real yy1 = (temp_y[j] == 0.0) ? tolerance : fabs(temp_y[j]);
	Real yy1 = (fabs(temp_y[j]) <= 0.01*tolerance) ? tolerance : fabs(temp_y[j]);
	Real err1 = fabs(errArr[j])/yy1;
	if (err1 < tolerance) continue;
	err = max(err, err1);
	Real scale1 = 0.8 * pow( tolerance * yy1 /  err1 , err_exponent );
	scale1 = min( max(scale1,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);
	scale = min(scale, scale1);
      }
      if (err == 0.0) { scale = MAX_SCALE_FACTOR; break; }
      
      if ( err < tolerance || h < hmin ) break;
      h *= scale;
      if ( x + h > xmax ) h = xmax - x;
      else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
    }
    if ( i >= ATTEMPTS || err > tolerance) { 
      h_next = h * scale; 
      cout << " ***failed, h_next="<< h_next<<" "<<x<<" "<<xmax<<" "<<err<<" "<<tolerance<<endl;
      // delete [] tmpArray; 
      // delete [] errArr;
      // delete [] temp_y; 
      // return -1; 
    };
    for (int i =0; i<n; i++)  temp_y[i] = yout[i];         
    x += h;
    h *= scale;
    if (h < hmin) h = hmin;  //Shengtai
    h_next = h;
    if ( last_interval ) break;
    if (  x + h > xmax ) { last_interval = 1; h = xmax - x; }
    else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
  }

  delete [] tmpArray; 
  delete [] errArr;
  delete [] temp_y;
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  static Real Runge_Kutta(Real (*f)(Real,Real), Real *y,          //
//                                                       Real x0, Real h) //
//                                                                            //
//  Description:                                                              //
//     This routine uses Verner's embedded 7th and 8th order methods to       //
//     approximate the solution of the differential equation y'=f(x,y) with   //
//     the initial condition y = y[0] at x = x0.  The value at x + h is       //
//     returned in y[1].  The function returns err / h ( the absolute error   //
//     per step size ).                                                       //
//                                                                            //
//  Arguments:                                                                //
//     Real *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y[0]).                   //
//     Real y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at x + h.                              //
//     Real x   Initial value of x.                                         //
//     Real h   Step size                                                   //
//                                                                            //
//  Return Values:                                                            //
//     This routine returns the err / h.  The solution of y(x) at x + h is    //
//     returned in y[1].                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

static void Runge_Kutta(void (*f)(const Real&, 
				  const Real [],
				  Real []), 
			const Real y[], 
			const Real& x0,
			Real yout [],
			Real errArr[],
			const Real& h,
			const int n,
			Real *tmpArray) {
   
   static const Real c1 = 13.0 / 288.0;
   static const Real c6 = 32.0 / 125.0;
   static const Real c7 = 31213.0 / 144000.0;
   static const Real c8 = 2401.0 / 12375.0;
   static const Real c9 = 1701.0 / 14080.0;
   static const Real c10 = 2401.0 / 19200.0;
   static const Real c11 = 19.0 / 450.0;

   static const Real a2 = 1.0 / 4.0;
   static const Real a3 = 1.0 / 12.0;
   static const Real a4 = 1.0 / 8.0;
   static const Real a5 = 2.0 / 5.0;
   static const Real a6 = 1.0 / 2.0;
   static const Real a7 = 6.0 / 7.0;
   static const Real a8 = 1.0 / 7.0;
   static const Real a9 = 2.0 / 3.0;
   static const Real a10 = 2.0 / 7.0;
   static const Real a12 = 1.0 / 3.0;

   static const Real b31 = 5.0 / 72.0;
   static const Real b32 = 1.0 / 72.0;
   static const Real b41 = 1.0 / 32.0;
   static const Real b43 = 3.0 / 32.0;
   static const Real b51 = 106.0 / 125.0;
   static const Real b53 = -408.0 / 125.0;
   static const Real b54 = 352.0 / 125.0;
   static const Real b61 = 1.0 / 48.0;
   static const Real b64 = 8.0 / 33.0;
   static const Real b65 = 125.0 / 528.0;
   static const Real b71 = -13893.0 / 26411.0;
   static const Real b74 =  39936.0 / 26411.0;
   static const Real b75 = -64125.0 / 26411.0;
   static const Real b76 =  60720.0 / 26411.0;
   static const Real b81 = 37.0/392.0;
   static const Real b85 = 1625.0/9408.0;
   static const Real b86 = -2.0/15.0;
   static const Real b87 = 61.0/6720.0;
   static const Real b91 = 17176.0 / 25515.0;
   static const Real b94 = -47104.0 / 25515.0;
   static const Real b95 = 1325.0 / 504.0;
   static const Real b96 = -41792.0 / 25515.0;
   static const Real b97 = 20237.0 / 145800.0;
   static const Real b98 = 4312.0 / 6075.0;
   static const Real b10_1 = -23834.0 / 180075.0;
   static const Real b10_4 = -77824.0 / 1980825.0;
   static const Real b10_5 = -636635.0 / 633864.0;
   static const Real b10_6 = 254048.0 / 300125.0;
   static const Real b10_7 = -183.0 / 7000.0;
   static const Real b10_8 = 8.0 / 11.0;
   static const Real b10_9 = -324.0 / 3773.0;
   static const Real b11_1 = 12733.0 / 7600.0;
   static const Real b11_4 = -20032.0 / 5225.0;
   static const Real b11_5 = 456485.0 / 80256.0;
   static const Real b11_6 = -42599.0 / 7125.0;
   static const Real b11_7 = 339227.0 / 912000.0;
   static const Real b11_8 = -1029.0 / 4180.0;
   static const Real b11_9 = 1701.0 / 1408.0;
   static const Real b11_10 = 5145.0 / 2432.0;
   static const Real b12_1 = -27061.0 / 204120.0;
   static const Real b12_4 = 40448.0 / 280665.0;
   static const Real b12_5 = -1353775.0 / 1197504.0;
   static const Real b12_6 = 17662.0 / 25515.0;
   static const Real b12_7 = -71687.0 / 1166400.0;
   static const Real b12_8 = 98.0 / 225.0;
   static const Real b12_9 = 1.0 / 16.0;
   static const Real b12_10 = 3773.0 / 11664.0;
   static const Real b13_1 = 11203.0 / 8680.0;
   static const Real b13_4 = -38144.0 / 11935.0;
   static const Real b13_5 = 2354425.0 / 458304.0;
   static const Real b13_6 = -84046.0 / 16275.0;
   static const Real b13_7 = 673309.0 / 1636800.0;
   static const Real b13_8 = 4704.0 / 8525.0;
   static const Real b13_9 = 9477.0 / 10912.0;
   static const Real b13_10 = -1029.0 / 992.0;
   static const Real b13_12 = 729.0 / 341.0;
   
   static const Real e1 = -6600.0 / 3168000.0;
   static const Real e6 = -135168.0 / 3168000.0;
   static const Real e7 = -14406.0 / 3168000.0;
   static const Real e8 = 57624.0 / 3168000.0;
   static const Real e9 = 54675.0 / 3168000.0;
   static const Real e10 = -396165.0 / 3168000.0;
   static const Real e11 = -133760.0 / 3168000.0;
   static const Real e12 = 437400.0 / 3168000.0;
   static const Real e13 = 136400.0 / 3168000.0;

   int i;
   Real *k1, *k2, *k3, *k4, *k5, *k6, *k7, *k8, 
     *k9, *k10, *k11, *k12, *k13, *yt;
   Real h4 = a2 * h;
   Real h6 = a3 * h;
   Real xh;
   
   k1 = tmpArray;
   k2 = tmpArray + n;
   k3 = k2 + n;
   k4 = k3 + n;
   k5 = k4 + n;
   k6 = k5 + n;
   k7 = k6 + n;
   k8 = k7 + n;
   k9 = k8 + n;
   k10 = k9 + n;
   k11 = k10 + n;
   k12 = k11 + n;
   k13 = k12 + n;
   yt = k13 + n;

   (*f)(x0, y, k1);
   xh = x0+h4;
   for (i=0;i<n; i++) yt[i] = y[i] + h4 * k1[i];
   (*f)(xh, yt, k2);
   xh = x0+a3*h;
   for (i=0;i<n; i++) yt[i] = y[i] + h * (b31*k1[i] + b32*k2[i]);
   (*f)(xh, yt, k3);
   xh = x0+a4*h;
   for (i=0;i<n; i++) yt[i] = y[i] + h * (b41*k1[i] + b43*k3[i]);   
   (*f)(xh, yt,k4 );
   xh = x0+a5*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b51*k1[i] + b53*k3[i] + b54*k4[i]);   
   (*f)(xh, yt,k5 );
   //k5 = (*f)(x0+a5*h, *y + h * ( b51*k1 + b53*k3 + b54*k4) );
   xh = x0+a6*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b61*k1[i] + b64*k4[i] + b65*k5[i]);   
   (*f)(xh, yt,k6 );   
   //k6 = (*f)(x0+a6*h, *y + h * ( b61*k1 + b64*k4 + b65*k5) );
   xh = x0+a7*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b71*k1[i] + b74*k4[i] + 
			 b75*k5[i] + b76*k6[i]);   
   (*f)(xh, yt,k7 );     
   //k7 = (*f)(x0+a7*h, *y + h * ( b71*k1 + b74*k4 + b75*k5 + b76*k6) );
   xh = x0+a8*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b81*k1[i] + b85*k5[i] + 
			 b86*k6[i] + b87*k7[i]);   
   (*f)(xh, yt,k8 );  
   //k8 = (*f)(x0+a8*h, *y + h * ( b81*k1 + b85*k5 + b86*k6 + b87*k7) );
   xh = x0+a9*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b91*k1[i] + b94*k4[i] +
			 b95*k5[i] + b96*k6[i] +
			 b97*k7[i] + b98*k8[i]);   
   (*f)(xh, yt,k9 );  
   // k9 = (*f)(x0+a9*h, *y + h * ( b91*k1 + b94*k4 + b95*k5 + b96*k6
   //                                                        + b97*k7 + b98*k8) );
   xh = x0+a10*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b10_1*k1[i] + b10_4*k4[i] + b10_5*k5[i] + 
			 b10_6*k6[i] + b10_7*k7[i] + b10_8*k8[i] + 
			 b10_9*k9[i]);   
   (*f)(xh, yt,k10 );     
   // k10 = (*f)(x0+a10*h, *y + h * ( b10_1*k1 + b10_4*k4 + b10_5*k5 + b10_6*k6
   //                                        + b10_7*k7 + b10_8*k8 + b10_9*k9 ) );
   xh = x0+h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b11_1*k1[i] + b11_4*k4[i] + b11_5*k5[i] + 
			 b11_6*k6[i] + b11_7*k7[i] + b11_8*k8[i] + 
			 b11_9*k9[i] + b11_10*k10[i]);   
   (*f)(xh, yt,k11 );       
   // k11 = (*f)(x0+h, *y + h * ( b11_1*k1 + b11_4*k4 + b11_5*k5 + b11_6*k6
   //                         + b11_7*k7 + b11_8*k8 + b11_9*k9 + b11_10 * k10 ) );
   xh = x0+a12*h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b12_1*k1[i] + b12_4*k4[i] + b12_5*k5[i] + 
			 b12_6*k6[i] + b12_7*k7[i] + b12_8*k8[i] + 
			 b12_9*k9[i] + b12_10 * k10[i]);   
   (*f)(xh, yt,k12 );          
   // k12 = (*f)(x0+a12*h, *y + h * ( b12_1*k1 + b12_4*k4 + b12_5*k5 + b12_6*k6
   //                         + b12_7*k7 + b12_8*k8 + b12_9*k9 + b12_10 * k10 ) );
   xh = x0+h;
//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) 
     yt[i] = y[i] + h * (b13_1*k1[i] + b13_4*k4[i] + b13_5*k5[i] + 
			 b13_6*k6[i] + b13_7*k7[i] + b13_8*k8[i] + 
			 b13_9*k9[i] + b13_10*k10[i] + 
			 b13_12*k12[i] );
   (*f)(xh, yt,k13 );          
   // k13 = (*f)(x0+h, *y + h * ( b13_1*k1 + b13_4*k4 + b13_5*k5 + b13_6*k6
   //              + b13_7*k7 + b13_8*k8 + b13_9*k9 + b13_10*k10 + b13_12*k12 ) );

//#ifdef OPENMP
#ifdef OPENMP_PARALLEL
#pragma omp parallel for
#endif
   for (i=0;i<n; i++) {
     yout[i] = y[i] + h*(c1 * k1[i] + c6 * k6[i] + c7 * k7[i] + c8 * k8[i] + 
			 c9 * k9[i] + c10 * k10[i] + c11 * k11[i]);
     errArr[i] = (e1*k1[i] + e6*k6[i] + e7*k7[i] + 
		  e8*k8[i] + e9*k9[i] + e10*k10[i] + 
		  e11*k11[i] + e12*k12[i] + e13*k13[i]);
   }
   // *(y+1) = *y +  h * (c1 * k1 + c6 * k6 + c7 * k7 + c8 * k8 + c9 * k9
   //                                                    + c10 * k10 + c11 * k11);
   // return e1*k1 + e6*k6 + e7*k7 + e8*k8 + e9*k9 + e10*k10 + e11*k11
   //                                                         + e12*k12 + e13*k13;
}

//extern "C" {
//    int My_Embedded_Verner_7_8(const Real y[],
//			const int n,
//			const Real& x0,
//			const Real& hmax,
//			Real  yout[],
//			void (*f)(const Real&, const Real [], Real [])) {  
//	return Embedded_Verner_7_8(const Real y[], const int n, const Real& x0, const Real& hmax,Real  yout[],
//			void (*f)(const Real&, const Real [], Real [])); 
//				  }
//}

