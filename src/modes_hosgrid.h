#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include "AMReX_Gpu.H"
#include <fftw3.h>
#include "read_modes.h"

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

void populate_hos_eta(ReadModes rm_obj, fftw_plan p, fftw_complex *eta_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_eta(int n0, int n1, const double dimL, fftw_plan p,
                      fftw_complex *eta_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_eta_nondim(int n0, int n1, fftw_plan p,
                             fftw_complex *eta_modes,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void dimensionalize_eta(const double dimL,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_vel(ReadModes rm_obj, double z,
                      std::vector<std::complex<double>> mX_vector,
                      std::vector<std::complex<double>> mY_vector,
                      std::vector<std::complex<double>> mZ_vector, fftw_plan p,
                      fftw_complex *x_modes, fftw_complex *y_modes,
                      fftw_complex *z_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                      int indv_start = 0);

void populate_hos_vel(int n0, int n1, double xlen, double ylen, double depth,
                      double z, const double dimL, const double dimT,
                      std::vector<std::complex<double>> mX_vector,
                      std::vector<std::complex<double>> mY_vector,
                      std::vector<std::complex<double>> mZ_vector, fftw_plan p,
                      fftw_complex *x_modes, fftw_complex *y_modes,
                      fftw_complex *z_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                      int indv_start = 0);

void populate_hos_vel_nondim(int n0, int n1, double nd_xlen, double nd_ylen,
                             double nd_depth, double nd_z,
                             std::vector<std::complex<double>> mX_vector,
                             std::vector<std::complex<double>> mY_vector,
                             std::vector<std::complex<double>> mZ_vector,
                             fftw_plan p, fftw_complex *x_modes,
                             fftw_complex *y_modes, fftw_complex *z_modes,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                             int indv_start = 0);

void dimensionalize_vel(int n0, int n1, const double dimL, const double dimT,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                        int indv_start = 0);

void do_ifftw(int n0, int n1, fftw_plan p, fftw_complex *f_in, double *sp_out);

} // namespace modes_hosgrid
#endif