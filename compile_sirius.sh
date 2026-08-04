#!/bin/bash

#module load mpich
#module load intel
#module unload hdf5
#module load phdf5/1.10.8

#hdf5path=/opt/ohpc/pub/libs/intel/mvapich2/hdf5/1.10.8
hdf5path=/share/apps/hdf5-1.8.19/hdf5_parallel
#hdf5path=/share/apps/hdf5-1.8.19/hdf5
source /share/apps/intel/ipsxe2015u5/parallel_studio_xe_2015/psxevars.sh >/dev/null 2>&1


#python configure.py --prob=dusty_soundwave --ndustfluids=4 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=disk_dustdrift --coord=cylindrical --ndustfluids=1 --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_collision --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_collision_different_Ts --ndustfluids=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_inelastic_collision --ndustfluids=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=disk_planet_dust --coord=cylindrical --ndustfluids=1 --nscalars=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd
#python configure.py --prob=disk_planet_dust --coord=cylindrical --eos=isothermal --ndustfluids=0 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++
#python configure.py --prob=disk_binary --coord=cylindrical --eos=isothermal --ndustfluids=0 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++

##python configure.py --prob=disk_binary --coord=cylindrical --ndustfluids=0 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ -d 

##python configure.py --prob=disk_binary --eos=isothermal --coord=cylindrical --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 
##python configure.py --prob=disk_planet --eos=isothermal --coord=spherical_polar --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 
python configure.py --prob=disk_planet --coord=cylindrical --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 
##python configure.py --prob=disk_planet_dz --coord=cylindrical --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 

##python configure.py --prob=disk_planet --eos=isothermal --coord=cylindrical --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 

##python configure.py --prob=disk_planet --coord=cylindrical  --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++ 


#python configure.py --prob=disk_amr --coord=spherical_polar --ndustfluids=0 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --cxx=g++ 
##python configure.py --prob=disk --eos=isothermal --coord=spherical_polar --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --cxx=g++ 
#python configure.py --prob=ssheet_planet --eos=isothermal --nghost=2 --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++

#python configure.py --prob=disk_planet_dust --coord=spherical_polar --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

##python configure.py --prob=disk_RWI_2D --coord=cylindrical --ndustfluids=3 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=icpx

#python configure.py --prob=disk_planet_dust_sph --coord=spherical_polar --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double #-omp # --cxx=g++-simd

#python configure.py --prob=dust_squaredrag --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dusty_shock --ndustfluids=3 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=disk --coord=cylindrical --eos=adiabatic --flux=hllc --ndustfluids=3 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dusty_kh --ndustfluids=1 --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -debug

#python configure.py --prob=kh_dust --ndustfluids=2 --nscalars=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=kh_dust --ndustfluids=0 --nscalars=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=kh_dust --ndustfluids=0 --nscalars=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=ssheet --flux=hlle --eos=isothermal --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=ssheet_planet_dust_3D --eos=isothermal --nghost=2 --ndustfluids=1 --nscalars=1 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd
#python configure.py --prob=ssheet_planet_dust_3D --eos=isothermal --nghost=2 --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++
#python configure.py --prob=ssheet_planet --eos=isothermal --nghost=2 --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++

#python configure.py --prob=ssheet_planet_dust --eos=isothermal --nghost=2 --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++

#python configure.py --prob=ssheet_planet_dust_3D --nghost=4 --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_diffusion -mpi --eos=isothermal --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_diffusion -mpi --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double --coord=cylindrical -omp --cxx=g++-simd

#python configure.py --prob=hb3 -b --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --ndustfluids=2 -omp --cxx=g++-simd

#python configure.py --prob=nsh_dust --ndustfluids=1 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=streaming_eigen --ndustfluids=1 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python configure.py --prob=streaming_eigen_2dust --ndustfluids=2 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python configure.py --prob=disk_streaming_cylindrical --ndustfluids=1 --coord=cylindrical --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=disk_streaming_spherical --coord=spherical_polar --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=streaming_nonlinear --ndustfluids=1 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python configure.py --prob=dust_NSH --ndustfluids=2 --eos=isothermal --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python configure.py --prob=dust_NSH --ndustfluids=9 --eos=isothermal --nghost=2 -mpi --flux=roe -omp --cxx=g++-simd

#python configure.py --prob=dmr_dust --ndustfluids=3 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd --nscalars=1

#python configure.py --prob=streaming_nonlinear_stratified --ndustfluids=3 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python configure.py --prob=streaming_stratified --ndustfluids=3 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python configure.py --prob=KHI_PPD_dust --ndustfluids=1 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python configure.py --prob=test_period_dust --ndustfluids=1 --flux=roe --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py --prob=dust_streaming --ndustfluids=1 --nghost=2 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python configure.py -g -b --prob gr_torus --coord=kerr-schild --flux hlle --nghost 4 -hdf5 --hdf5_path=${hdf5path} -mpi


make clean
make -j 16
