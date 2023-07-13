#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include "AMReX_Gpu.H"
#include <complex>
#include <fftw3.h>
#include <vector>

namespace modes_hosgrid {

void copy_complex(int n0, int n1,
                  std::vector<std::complex<double>> complex_vector,
                  fftw_complex *ptr);
fftw_complex *allocate_complex(int n0, int n1);

fftw_plan plan_ifftw(int n0, int n1, fftw_complex *in);

fftw_complex *
allocate_plan_copy(int n0, int n1, fftw_plan &p,
                   std::vector<std::complex<double>> complex_vector);

fftw_complex *allocate_copy(int n0, int n1,
                            std::vector<std::complex<double>> complex_vector);

void populate_hos_eta(int n0, int n1, fftw_plan p, fftw_complex *eta_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_vel(int n0, int n1, double xlen, double ylen, double depth,
                      double z, std::vector<std::complex<double>> mX_vector,
                      std::vector<std::complex<double>> mY_vector,
                      std::vector<std::complex<double>> mZ_vector, fftw_plan p,
                      fftw_complex *x_modes, fftw_complex *y_modes,
                      fftw_complex *z_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                      int indv_start = 0);

void do_ifftw(int n0, int n1, fftw_plan p, fftw_complex *f_in, double *sp_out);

} // namespace modes_hosgrid
#endif