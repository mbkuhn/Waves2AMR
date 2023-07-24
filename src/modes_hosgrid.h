#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include "AMReX_Gpu.H"
#include "read_modes.h"
#include <fftw3.h>

namespace modes_hosgrid {

void copy_complex(const int n0, const int n1,
                  std::vector<std::complex<double>> complex_vector,
                  fftw_complex *ptr);
fftw_complex *allocate_complex(const int n0, const int n1);

fftw_plan plan_ifftw(const int n0, const int n1, fftw_complex *in);

fftw_complex *
allocate_plan_copy(const int n0, const int n1, fftw_plan &p,
                   std::vector<std::complex<double>> complex_vector);

fftw_complex *allocate_copy(const int n0, const int n1,
                            std::vector<std::complex<double>> complex_vector);

void populate_hos_eta(ReadModes rm_obj, fftw_plan p, fftw_complex *eta_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_eta(const int n0, const int n1, const double dimL,
                      fftw_plan p, fftw_complex *eta_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_eta_nondim(const int n0, const int n1, fftw_plan p,
                             fftw_complex *eta_modes,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void dimensionalize_eta(const double dimL,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta);

void populate_hos_vel(ReadModes rm_obj, const double z,
                      std::vector<std::complex<double>> mX_vector,
                      std::vector<std::complex<double>> mY_vector,
                      std::vector<std::complex<double>> mZ_vector, fftw_plan p,
                      fftw_complex *x_modes, fftw_complex *y_modes,
                      fftw_complex *z_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                      int indv_start = 0);

void populate_hos_vel(const int n0, const int n1, const double xlen,
                      const double ylen, const double depth, const double z,
                      const double dimL, const double dimT,
                      std::vector<std::complex<double>> mX_vector,
                      std::vector<std::complex<double>> mY_vector,
                      std::vector<std::complex<double>> mZ_vector, fftw_plan p,
                      fftw_complex *x_modes, fftw_complex *y_modes,
                      fftw_complex *z_modes,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                      amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                      const int indv_start = 0);

void populate_hos_vel_nondim(const int n0, const int n1, const double nd_xlen,
                             const double nd_ylen, const double nd_depth,
                             const double nd_z,
                             std::vector<std::complex<double>> mX_vector,
                             std::vector<std::complex<double>> mY_vector,
                             std::vector<std::complex<double>> mZ_vector,
                             fftw_plan p, fftw_complex *x_modes,
                             fftw_complex *y_modes, fftw_complex *z_modes,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                             amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                             int indv_start = 0);

void dimensionalize_vel(const int n0, const int n1, const double dimL,
                        const double dimT,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
                        amrex::Gpu::DeviceVector<amrex::Real> &HOS_w,
                        int indv_start = 0);

void do_ifftw(const int n0, const int n1, fftw_plan p, fftw_complex *f_in,
              double *sp_out);

} // namespace modes_hosgrid
#endif