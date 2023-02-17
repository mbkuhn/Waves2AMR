#include "modes_hosgrid.h"
#include "cmath"

void modes_hosgrid::copy_complex(std::vector<std::complex<double> > complex_vector, fftw_complex* ptr) {
  for (int i = 0; i < complex_vector.size(); ++i) {
    ptr[i][0] = complex_vector[i].real();
    ptr[i][1] = complex_vector[i].imag();
  }
}

void modes_hosgrid::allocate_copy_complex(std::vector<std::complex<double> > complex_vector, fftw_complex* ptr) {
  ptr = new fftw_complex [complex_vector.size()];
  copy_complex(complex_vector, ptr);
}

fftw_plan modes_hosgrid::plan_ifftw(int n0, int n1, fftw_complex* in) {
  unsigned int flag = FFTW_PATIENT;
  double *out;
  return fftw_plan_dft_c2r_2d(n0, n1, in, out, flag);
}

void modes_hosgrid::populate_hos_eta(fftw_plan p,
                                     fftw_complex* eta_modes,
                                     std::vector<double> HOS_eta) {
  double *out;
  fftw_execute_dft_c2r(p, eta_modes, out);
  std::copy(&out[0], &out[0] + HOS_eta.size(), HOS_eta.begin());
}
void modes_hosgrid::populate_hos_vel(
    fftw_plan p, fftw_complex* x_modes,
    fftw_complex* y_modes, fftw_complex* z_modes,
    int n0, int n1, double xlen, double ylen, double depth, double z,
    std::vector<double> HOS_u, std::vector<double> HOS_v,
    std::vector<double> HOS_w) {
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
  double *out;
  // Perform inverse fft
  fftw_execute_dft_c2r(p, x_modes, out);
  std::copy(&out[0], &out[0] + HOS_u.size(), HOS_u.begin());
  // Repeat in other directions
  fftw_execute_dft_c2r(p, y_modes, out);
  std::copy(&out[0], &out[0] + HOS_v.size(), HOS_v.begin());
  fftw_execute_dft_c2r(p, z_modes, out);
  std::copy(&out[0], &out[0] + HOS_w.size(), HOS_w.begin());
}
