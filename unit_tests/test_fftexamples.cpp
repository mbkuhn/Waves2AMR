#include <complex>
#include <fftw3.h>
#include <vector>
#include "gtest/gtest.h"
#include <array>

namespace aux_tests {

class FFTExampleTest : public testing::Test {};

// Not actually a test of the code, but a working example of fftw steps
TEST_F(FFTExampleTest, HowToBasic) {
  int nx = 4;
  int ny = 20;
  fftw_complex in[nx / 2 + 1][ny];
  double out[nx][ny];

  for (int i = 0; i < nx / 2 + 1; ++i) {
    for (int j = 0; j < ny; ++j) {
      (in[i][j])[0] = 0.0;
      (in[i][j])[1] = 0.0;
    }
  }

  auto plan = fftw_plan_dft_c2r_2d(nx, ny, &in[0][0], &out[0][0], FFTW_MEASURE);
  fftw_execute_dft_c2r(plan, &in[0][0], &out[0][0]);
  fftw_destroy_plan(plan);
}

TEST_F(FFTExampleTest, HowTo) {
  int nx = 4;
  int ny = 20;
  fftw_complex in[nx / 2 + 1][ny];
  double out[nx][ny];

  for (int i = 0; i < nx / 2 + 1; ++i) {
    for (int j = 0; j < ny; ++j) {
      (in[i][j])[0] = 0.0;
      (in[i][j])[1] = 0.0;
    }
  }

  fftw_complex *inptr = &in[0][0];
  double *outptr = &out[0][0];

  auto plan = fftw_plan_dft_c2r_2d(nx, ny, &in[0][0], outptr, FFTW_MEASURE);
  fftw_execute_dft_c2r(plan, &in[0][0], outptr);
  fftw_destroy_plan(plan);
}

TEST_F(FFTExampleTest, SineSolution) {
  // Data size
  int nx = 4;
  int ny = 12;
  // Declare arrays
  fftw_complex modes2D[nx][ny / 2 + 1];
  double spatial2D[nx][ny];
  // Create plans
  auto plan_r2c = fftw_plan_dft_r2c_2d(nx, ny, &spatial2D[0][0], &modes2D[0][0],
                                       FFTW_PATIENT);
  auto plan_c2r = fftw_plan_dft_c2r_2d(nx, ny, &modes2D[0][0], &spatial2D[0][0],
                                       FFTW_PATIENT);

  // Initialize real data (after plan)
  int nf = 3;
  double omega0 = 2.0 * M_PI * nf;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      spatial2D[ix][iy] = sin(omega0 * (double)iy / (double)ny);
    }
  }

  // Execute r2c and destroy plan
  fftw_execute_dft_r2c(plan_r2c, &spatial2D[0][0], &modes2D[0][0]);
  fftw_destroy_plan(plan_r2c);

  // Check solution against solution used in next test
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny / 2 + 1; ++iy) {
      EXPECT_NEAR((modes2D[ix][iy])[0], 0.0, 1e-13);
      EXPECT_NEAR((modes2D[ix][iy])[1],
                  -2.0 * ((iy == nf) && (ix == 0) ? (double)ny : 0.0), 1e-13);
    }
  }

  // Execute c2r and destroy plan
  fftw_execute_dft_c2r(plan_c2r, &modes2D[0][0], &spatial2D[0][0]);
  fftw_destroy_plan(plan_c2r);

  // Check against initial result
  double factor = 0.25 / ny;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      EXPECT_NEAR(sin(omega0 * (double)iy / (double)ny),
                  factor * spatial2D[ix][iy], 1e-13);
    }
  }
}

} // namespace