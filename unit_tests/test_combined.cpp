#include "../Waves2AMR.h"
#include "gtest/gtest.h"
#include <array>

namespace {

class CombinedTest : public testing::Test {};

TEST_F(CombinedTest, ReadFFTNonDim) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read and convert nondim quantities
  ReadModes rmodes(fname, true, true);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  std::vector<std::complex<double>> mX(vsize, 0.0);
  std::vector<std::complex<double>> mY(vsize, 0.0);
  std::vector<std::complex<double>> mZ(vsize, 0.0);
  std::vector<std::complex<double>> mFS(vsize, 0.0);

  // Populate mode data
  rmodes.get_data(0.0, mX, mY, mZ, mFS);
  // Get dimensions
  int n0 = rmodes.get_first_dimension();
  int n1 = rmodes.get_second_dimension();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();
  double depth = rmodes.get_depth();

  // Allocate complex pointers and get plan
  fftw_plan plan;
  auto eta_modes = modes_hosgrid::allocate_plan_copy(n0, n1, plan, mFS);
  auto u_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto v_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto w_modes = modes_hosgrid::allocate_complex(n0, n1);

  // Set up output vectors
  std::vector<double> eta, u, v, w;
  eta.resize((n0 * n1));
  u.resize((n0 * n1));
  v.resize((n0 * n1));
  w.resize((n0 * n1));

  // Get spatial data for eta
  modes_hosgrid::populate_hos_eta(n0, n1, plan, eta_modes, eta);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, eta[idx]);
      min_eta = std::min(min_eta, eta[idx]);
    }
  }

  // !! -- Reference values are from Grid2Grid library -- !! //

  // Check max and min
  EXPECT_NEAR(max_eta, 0.18810489039622619, 1e-10);
  EXPECT_NEAR(min_eta, -0.16057046363299562, 1e-10);

  // Store values to check velocity
  double ht[2]{-0.75325763322349282, -1.3822500981745969};
  double umaxref[2]{0.10194064101886856, 7.5818709771411807E-002};
  double uminref[2]{-9.0532780211604741E-002, -7.0284594145064244E-002};
  double vmaxref[2]{2.9974269816685099E-002, 2.3336316239396335E-002};
  double vminref[2]{-2.7966748429457091E-002, -2.1250200622124214E-002};
  double wmaxref[2]{6.6195664852053832E-002, 1.2065346007419406E-002};
  double wminref[2]{-6.6361625157720372E-002, -1.2117387641263945E-002};
  // Get spatial data for velocity at different heights
  for (int n = 0; n < 2; ++n) {

    modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht[n], mX, mY,
                                    mZ, plan, u_modes, v_modes, w_modes, u, v,
                                    w);

    double max_u = -100.0;
    double min_u = 100.0;
    double max_v = -100.0;
    double min_v = 100.0;
    double max_w = -100.0;
    double min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, u[idx]);
        min_u = std::min(min_u, u[idx]);
        max_v = std::max(max_v, v[idx]);
        min_v = std::min(min_v, v[idx]);
        max_w = std::max(max_w, w[idx]);
        min_w = std::min(min_w, w[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref[n], 1e-10);
    EXPECT_NEAR(min_u, uminref[n], 1e-10);
    EXPECT_NEAR(max_v, vmaxref[n], 1e-10);
    EXPECT_NEAR(min_v, vminref[n], 1e-10);
    EXPECT_NEAR(max_w, wmaxref[n], 1e-10);
    EXPECT_NEAR(min_w, wminref[n], 1e-10);
  }

  // Delete complex pointers to allocated data
  delete[] eta_modes;
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  // Delete plan
  fftw_destroy_plan(plan);
}

} // namespace