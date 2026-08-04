//#ifndef UTILS_UTILS_HPP_
//#define UTILS_UTILS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nbutil.hpp
//! \brief prototypes of functions and class definitions for nboby/*.cpp files

// C headers

// C++ headers

// Athena++ headers
//double ran2(std::int64_t *idum);
int Embedded_Verner_7_8(const Real y[],
			const int n,
			const Real& x,
			const Real& h,
			Real  yout[],
			void (*derivs)(const double&, 
				  const double [],
				  Real [])); 

