#include "modes_hosgrid.h"

fftw_plan modes_hosgrid::plan_ifftw(int n0, int n1, fftw_complex *in, double *out) {
  unsigned int flag;
  return fftw_plan_dft_c2r_2d(n0, n1, in, out, flag);
}

void modes_hosgrid::populate_hos_eta(fftw_plan p, std::vector<double> modes,
                      std::vector<double> HOS_eta) {}
void modes_hosgrid::populate_hos_u(fftw_plan p, std::vector<double> modes, double z,
                    std::vector<double> HOS_u) {}
