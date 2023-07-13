#include "modes_hosgrid.h"
#include "cmath"
#include <iostream>

void modes_hosgrid::copy_complex(
    int n0, int n1, std::vector<std::complex<double>> complex_vector,
    fftw_complex *ptr) {
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1 / 2 + 1; ++j) {
      int idx = i * (n1 / 2 + 1) + j;
      ptr[idx][0] = complex_vector[idx].real();
      ptr[idx][1] = complex_vector[idx].imag();
    }
  }
}

fftw_complex *modes_hosgrid::allocate_complex(int n0, int n1) {
  // Allocate data needed for modes and create pointer
  fftw_complex *a_ptr = new fftw_complex[n0 * (n1 / 2 + 1)];
  // Return pointer to fftw_complex data
  return a_ptr;
}

fftw_plan modes_hosgrid::plan_ifftw(int n0, int n1, fftw_complex *in) {
  unsigned int flag = FFTW_PATIENT;
  // Output array is used for planning (except for FFTW_ESTIMATE)
  double out[n0][n1];
  // Make and return plan
  return fftw_plan_dft_c2r_2d(n0, n1, in, &out[0][0], flag);
}

fftw_complex *modes_hosgrid::allocate_plan_copy(
    int n0, int n1, fftw_plan &p,
    std::vector<std::complex<double>> complex_vector) {
  // Allocate and get pointer
  auto a_ptr = allocate_complex(n0, n1);
  // Create plan before data is initialized
  p = plan_ifftw(n0, n1, a_ptr);
  // Copy mode data from input vector
  copy_complex(n0, n1, complex_vector, a_ptr);
  // Return pointer to fftw_complex data
  return a_ptr;
}

fftw_complex *
modes_hosgrid::allocate_copy(int n0, int n1,
                             std::vector<std::complex<double>> complex_vector) {
  // Allocate and get pointer
  auto a_ptr = allocate_complex(n0, n1);
  // Copy mode data from input vector
  copy_complex(n0, n1, complex_vector, a_ptr);
  // Return pointer to fftw_complex data
  return a_ptr;
}

void modes_hosgrid::populate_hos_eta(
    int n0, int n1, fftw_plan p, fftw_complex *eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {
  // Local array for output data
  double out[n0 * n1];
  // Perform complex-to-real (inverse) FFT
  do_ifftw(n0, n1, p, eta_modes, &out[0]);

  // Copy data to output vector
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + HOS_eta.size(),
                   HOS_eta.begin());

  // !! -- This function MODIFIES the modes -- !! //
  //   .. they are not intended to be reused ..   //
}

void modes_hosgrid::populate_hos_vel(
    int n0, int n1, double xlen, double ylen, double depth, double z,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector, fftw_plan p,
    fftw_complex *x_modes, fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {
  // Reused constants (lengths are nondim)
  const double twoPi_xlen = 2.0 * M_PI / xlen;
  const double twoPi_ylen = 2.0 * M_PI / ylen;
  // Loop modes to modify them
  for (int ix = 0; ix < n0; ++ix) {
    for (int iy = 0; iy < n1 / 2 + 1; ++iy) {

      // Get wavenumbers
      const double kxN2 = (double)(ix < n0 / 2 + 1 ? ix : ix - n0) * twoPi_xlen;
      const double ky = (double)iy * twoPi_ylen;
      const double k = sqrt(kxN2 * kxN2 + ky * ky);
      // Get depth-related quantities
      const double kZ = k * (z + depth);
      const double kD = k * depth;
      // Get coefficients
      double coeff = 1.0;
      double coeff2 = 1.0;
      if (iy == 0) {
        // Do nothing for ix = 0, iy = 0
        if (ix != 0) {
          // Modified coeffs for iy = 0, ix > 0
          if ((kZ < 50.0) && (kD <= 50.0)) {
            coeff =
                exp(k * z) * (1.0 + exp(-2.0 * kZ)) / (1.0 + exp(-2.0 * kD));
            coeff2 =
                exp(k * z) * (1.0 - exp(-2.0 * kZ)) / (1.0 - exp(-2.0 * kD));
          } else {
            coeff = exp(k * z);
            coeff2 = coeff;
          }
          if (coeff >= 3.0) {
            coeff = 3.0;
          }
          if (coeff2 >= 3.0) {
            coeff2 = 3.0;
          }
        }
      } else {
        // Ordinary coefficients for other cases
        if ((kZ < 50.0) && (kD <= 50.0)) {
          coeff = cosh(kZ) / cosh(kD);
          coeff2 = sinh(kZ) / sinh(kD);
        } else {
          coeff = exp(k * z);
          coeff2 = coeff;
        }
        if (coeff >= 1000.0) {
          coeff = 1000.0;
        }
        if (coeff2 >= 1000.0) {
          coeff2 = 1000.0;
        }
      }
      // Multiply modes by coefficients
      // hosProcedure is velocity, I think
      int idx = ix * (n1 / 2 + 1) + iy;
      (x_modes[idx])[0] = coeff * mX_vector[idx].real();
      (x_modes[idx])[1] = coeff * mX_vector[idx].imag();
      (y_modes[idx])[0] = coeff * mY_vector[idx].real();
      (y_modes[idx])[1] = coeff * mY_vector[idx].imag();
      (z_modes[idx])[0] = coeff2 * mZ_vector[idx].real();
      (z_modes[idx])[1] = coeff2 * mZ_vector[idx].imag();
    }
  }
  // Output pointer
  int xy_size = n0 * n1;
  auto out = new double[xy_size];
  // Perform inverse fft
  do_ifftw(n0, n1, p, x_modes, &out[0]);
  // Copy to output vectors
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + xy_size,
                   &HOS_u[indv_start]);
  // Repeat in other directions
  do_ifftw(n0, n1, p, y_modes, &out[0]);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + xy_size,
                   &HOS_v[indv_start]);
  do_ifftw(n0, n1, p, z_modes, &out[0]);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + xy_size,
                   &HOS_w[indv_start]);
}

void modes_hosgrid::do_ifftw(int n0, int n1, fftw_plan p, fftw_complex *f_in,
                             double *sp_out) {
  // Modify modes with conversion coefficients
  for (int ix = 0; ix < n0; ++ix) {
    for (int iy = 0; iy < n1 / 2 + 1; ++iy) {
      int idx = ix * (n1 / 2 + 1) + iy;
      double f2s = (iy == 0 ? 1.0 : 0.5);
      (f_in[idx])[0] *= f2s;
      (f_in[idx])[1] *= f2s;
    }
  }
  // Perform fft
  fftw_execute_dft_c2r(p, f_in, sp_out);
}
