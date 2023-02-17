#include "../src/modes_hosgrid.h"
#include "gtest/gtest.h"
#include <array>

namespace {

class HOSGridTest : public testing::Test {};

TEST_F(HOSGridTest, Sine) {
  // Create modes in vector form
  int nx = 4;
  int ny = 20;
  double omega0 = 0.5;
  std::vector<std::complex<double>> modes2D;
  modes2D.resize((nx / 2 + 1) * ny);
  for (int ix = 0; ix < nx / 2 + 1; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      double omega = 2.0 * M_PI / ((double)ix + 1.0);
      (modes2D[ix * ny + iy]).real(0.0);
      (modes2D[ix * ny + iy]).imag(-M_PI * (abs(omega) == omega0 ? 1.0 : 0.0));
    }
  }

  // Set up fftw_complex ptr
  fftw_complex* ptr_modes;
  modes_hosgrid::allocate_copy_complex(modes2D, ptr_modes);
  
  // Setup output vector
  std::vector<double> spatial2D;
  spatial2D.resize((nx * ny));

  // Get plan
  auto plan = modes_hosgrid::plan_ifftw(nx, ny, ptr_modes);
  // Perform fftw
  modes_hosgrid::populate_hos_eta(plan, ptr_modes, spatial2D);

  // Solution should be sine curve
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      EXPECT_EQ(sin(omega0 * (double)iy / (double)ny), spatial2D[ix * ny + iy]);
    }
  }
}

} // namespace