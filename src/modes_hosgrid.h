#ifndef MODES_HOSGRID_H
#define MODES_HOSGRID_H
#include <fftw3.h>
#include <vector>
#include <complex>

namespace modes_hosgrid {

void copy_complex(std::vector<std::complex<double> > complex_vector, fftw_complex* ptr);
void allocate_copy_complex(std::vector<std::complex<double> > complex_vector, fftw_complex* ptr);

fftw_plan plan_ifftw(int n0, int n1, fftw_complex* in);

void populate_hos_eta(fftw_plan p, fftw_complex* eta_modes,
                      std::vector<double> HOS_eta);
void populate_hos_vel(fftw_plan p, fftw_complex* x_modes,
                      fftw_complex* y_modes,
                      fftw_complex* z_modes, int n0, int n1,
                      double xlen, double ylen, double depth, double z,
                      std::vector<double> HOS_u, std::vector<double> HOS_v,
                      std::vector<double> HOS_w);

} // namespace modes_hosgrid
#endif