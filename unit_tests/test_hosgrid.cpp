#include "../src/modes_hosgrid.h"
#include "gtest/gtest.h"
#include <array>

namespace w2a_tests {

class HOSGridTest : public testing::Test {};

TEST_F(HOSGridTest, AllocatePlanCopy) {
  // Create modes in vector form
  int nx = 4;
  int ny = 6;
  std::vector<std::complex<double>> modes2D;
  modes2D.resize(nx * (ny / 2 + 1));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny / 2 + 1; ++iy) {
      (modes2D[ix * (ny / 2 + 1) + iy]).real(ix + iy);
      (modes2D[ix * (ny / 2 + 1) + iy]).imag(ix - iy);
    }
  }

  // Set up fftw_complex ptr
  fftw_plan plan;
  fftw_complex *ptr_modes =
      modes_hosgrid::allocate_plan_copy(nx, ny, plan, modes2D);

  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny / 2 + 1; ++iy) {
      int idx = ix * (ny / 2 + 1) + iy;
      EXPECT_EQ(ix + iy, (ptr_modes[idx])[0]);
      EXPECT_EQ(ix - iy, (ptr_modes[idx])[1]);
    }
  }

  // Delete complex pointer to allocated data
  delete[] ptr_modes;
  // Delete plan
  fftw_destroy_plan(plan);
}

TEST_F(HOSGridTest, Sine) {
  // Create modes in vector form
  int nx = 10;
  int ny = 16;
  int nf = 2;
  double omega0 = 2.0 * M_PI * nf;
  std::vector<std::complex<double>> modes2D;
  modes2D.resize(nx * (ny / 2 + 1));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny / 2 + 1; ++iy) {
      (modes2D[ix * (ny / 2 + 1) + iy]).real(0.0);
      (modes2D[ix * (ny / 2 + 1) + iy])
          .imag((ix == nf && iy == 0 ? -1.0 : 0.0));
      // Anticipate conversion coeffs
      if (iy != 0) {
        (modes2D[ix * (ny / 2 + 1) + iy])
            .imag(2.0 * (modes2D[ix * (ny / 2 + 1) + iy]).imag());
      }
    }
  }

  // Set up fftw_complex ptr and get plan
  fftw_plan plan;
  fftw_complex *ptr_modes =
      modes_hosgrid::allocate_plan_copy(nx, ny, plan, modes2D);

  // Set up output vector
  amrex::Gpu::DeviceVector<amrex::Real> spatial2D(nx * ny, 0.0);
  // Perform fftw
  modes_hosgrid::populate_hos_eta(nx, ny, plan, ptr_modes, spatial2D);

  // Copy from device to host
  std::vector<amrex::Real> vlocal;
  vlocal.resize(nx * ny);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, spatial2D.begin(), spatial2D.end(),
                   &vlocal[0]);

  // Solution should be sine curve
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      EXPECT_NEAR(sin(omega0 * (double)ix / (double)nx), vlocal[ix * ny + iy],
                  1e-15);
    }
  }

  // Delete complex pointer to allocated data
  delete[] ptr_modes;
  // Delete plan
  fftw_destroy_plan(plan);
}

TEST_F(HOSGridTest, Cosine) {
  // Create modes in vector form
  int nx = 10;
  int ny = 16;
  int nf = 2;
  double omega0 = 2.0 * M_PI * nf;
  std::vector<std::complex<double>> modes2D;
  modes2D.resize(nx * (ny / 2 + 1));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny / 2 + 1; ++iy) {
      (modes2D[ix * (ny / 2 + 1) + iy])
          .real((iy == nf && ix == 0 ? 2.0 * ny : 0.0));
      (modes2D[ix * (ny / 2 + 1) + iy]).imag(0.0);
      // Anticipate conversion coeffs
      if (iy != 0) {
        (modes2D[ix * (ny / 2 + 1) + iy])
            .real(2.0 * (modes2D[ix * (ny / 2 + 1) + iy]).real());
      }
    }
  }

  // Set up fftw_complex ptr and get plan
  fftw_plan plan;
  fftw_complex *ptr_modes =
      modes_hosgrid::allocate_plan_copy(nx, ny, plan, modes2D);

  // Setup output vector
  amrex::Gpu::DeviceVector<amrex::Real> spatial2D(nx * ny, 0.0);
  // Perform fftw
  modes_hosgrid::populate_hos_eta(nx, ny, plan, ptr_modes, spatial2D);

  // Copy from device to host
  std::vector<amrex::Real> vlocal;
  vlocal.resize(nx * ny);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, spatial2D.begin(), spatial2D.end(),
                   &vlocal[0]);

  // Solution should be cosine curve
  double factor = 0.25 / ny;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      EXPECT_NEAR(cos(omega0 * (double)iy / (double)ny),
                  factor * vlocal[ix * ny + iy], 1e-15);
    }
  }

  // Delete complex pointer to allocated data
  delete[] ptr_modes;
  // Delete plan
  fftw_destroy_plan(plan);
}

} // namespace w2a_tests