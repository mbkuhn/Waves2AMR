#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include <complex>
#include <fftw3.h>
#include <vector>

namespace modes_hosgrid {

fftw_plan plan_ifftw(int n0, int n1, fftw_complex *in, double *out) {
  unsigned int flag;
  //fftw_plan p;
  return fftw_plan_dft_c2r_2d(n0, n1, in, out, flag);
}

void populate_hos_eta(fftw_plan p, std::vector<double> modes,
                      std::vector<double> HOS_eta) {}
void populate_hos_u(fftw_plan p, std::vector<double> modes, double z,
                    std::vector<double> HOS_u) {}

} // namespace modes_hosgrid
#endif