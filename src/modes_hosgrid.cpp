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

void modes_hosgrid::populate_hos_eta(int n0, int n1, fftw_plan p,
                                     fftw_complex *eta_modes,
                                     std::vector<double> &HOS_eta) {
  // Local array for output data
  double out[n0][n1];
  // Perform complex-to-real (inverse) FFT
  fftw_execute_dft_c2r(p, eta_modes, &out[0][0]);
  // Copy data to output vector
  std::copy(&out[0][0], &out[0][0] + HOS_eta.size(), HOS_eta.begin());
}

void modes_hosgrid::populate_hos_vel(
    int n0, int n1, double xlen, double ylen, double depth, double z,
    fftw_plan p, fftw_complex *x_modes, fftw_complex *y_modes,
    fftw_complex *z_modes, std::vector<double> &HOS_u,
    std::vector<double> &HOS_v, std::vector<double> &HOS_w) {
  // Reused constants (lengths are nondim)
  const double twoPi_xlen = 2.0 * M_PI / xlen;
  const double twoPi_ylen = 2.0 * M_PI / ylen;
  // Loop modes to modify them
  for (int ix = 0; ix < n0 / 2 + 1; ++ix) { // index limit?
    for (int iy = 0; iy < n1; ++iy) {

      // Get wavenumbers
      const double kx = (double)ix * twoPi_xlen;
      const double kyN2 = (double)(iy < n1 / 2 + 1 ? iy : n1 - iy) * twoPi_ylen;
      const double k = sqrt(kx * kx + kyN2 * kyN2);
      // Get depth-related quantities
      const double kZ = k * (z + depth);
      const double kD = k * depth;
      // Get coefficients
      double coeff = 1.0;
      double coeff2 = 1.0;
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
      // Multiply modes by coefficients
      // hosProcedure is velocity, I think
      int idx = ix * n1 + iy;
      (x_modes[idx])[0] *= coeff;
      (x_modes[idx])[1] *= coeff;
      (y_modes[idx])[0] *= coeff;
      (y_modes[idx])[1] *= coeff;
      (z_modes[idx])[0] *= coeff2;
      (z_modes[idx])[1] *= coeff2;
    }
  }
  // Output pointer
  auto out = new double[HOS_u.size()];
  // Perform inverse fft
  fftw_execute_dft_c2r(p, x_modes, out);
  // Copy to output vectors
  std::copy(&out[0], &out[0] + HOS_u.size(), HOS_u.begin());
  // Repeat in other directions
  fftw_execute_dft_c2r(p, y_modes, out);
  std::copy(&out[0], &out[0] + HOS_v.size(), HOS_v.begin());
  fftw_execute_dft_c2r(p, z_modes, out);
  std::copy(&out[0], &out[0] + HOS_w.size(), HOS_w.begin());
}
