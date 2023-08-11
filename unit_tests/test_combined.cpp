#include "Waves2AMR.h"
#include "gtest/gtest.h"
#include <array>

namespace w2a_tests {

class CombinedTest : public testing::Test {};

TEST_F(CombinedTest, ReadFFTNonDim) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read
  ReadModes rmodes(fname);
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
  double nd_xlen = rmodes.get_nondim_xlen();
  double nd_ylen = rmodes.get_nondim_ylen();
  double nd_depth = rmodes.get_nondim_depth();

  // Allocate complex pointers and get plan
  fftw_plan plan;
  auto eta_modes = modes_hosgrid::allocate_plan_copy(n0, n1, plan, mFS);
  auto u_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto v_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto w_modes = modes_hosgrid::allocate_complex(n0, n1);

  // Set up output vectors
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w(n0 * n1, 0.0);

  // Get spatial data for eta
  modes_hosgrid::populate_hos_eta_nondim(n0, n1, plan, eta_modes, eta);
  // Transfer to host
  std::vector<amrex::Real> etalocal;
  etalocal.resize(eta.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(),
                   &etalocal[0]);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etalocal[idx]);
      min_eta = std::min(min_eta, etalocal[idx]);
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

    modes_hosgrid::populate_hos_vel_nondim(n0, n1, nd_xlen, nd_ylen, nd_depth,
                                           ht[n], mX, mY, mZ, plan, u_modes,
                                           v_modes, w_modes, u, v, w);

    // Transfer to host
    std::vector<amrex::Real> ulocal, vlocal, wlocal;
    ulocal.resize(n0 * n1);
    vlocal.resize(n0 * n1);
    wlocal.resize(n0 * n1);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, u.begin(), u.end(), &ulocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), &vlocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, w.begin(), w.end(), &wlocal[0]);
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

TEST_F(CombinedTest, ReadFFTDim) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read
  ReadModes rmodes(fname);
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
  // Get dimensional constants
  double dimL = rmodes.get_L();
  double dimT = rmodes.get_T();
  // Get other values, which have been dimensionalized by readmodes
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
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w(n0 * n1, 0.0);

  // Get spatial data for eta
  modes_hosgrid::populate_hos_eta(n0, n1, dimL, plan, eta_modes, eta);
  // Transfer to host
  std::vector<amrex::Real> etalocal;
  etalocal.resize(eta.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(),
                   &etalocal[0]);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etalocal[idx]);
      min_eta = std::min(min_eta, etalocal[idx]);
    }
  }

  // !! -- Reference values are from Grid2Grid library -- !! //

  // Check max and min
  EXPECT_NEAR(max_eta, 4.2660225672921364, 1e-10);
  EXPECT_NEAR(min_eta, -3.6415705091772734, 1e-10);

  // Store values to check velocity
  double ht[2]{-17.083096859139136, -31.347989411831382};
  double umaxref[2]{1.4526142174097856, 1.0803869257525434};
  double uminref[2]{-1.2900566679060890, -1.0015279450822712};
  double vmaxref[2]{0.42712160780051028, 0.33253336856145094};
  double vminref[2]{-0.39851521412184132, -0.30280703787995972};
  double wmaxref[2]{0.94326230376743103, 0.17192645615306479};
  double wminref[2]{-0.94562717313775668, -0.17266802905729850};
  // Get spatial data for velocity at different heights
  for (int n = 0; n < 2; ++n) {

    modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht[n], dimL,
                                    dimT, mX, mY, mZ, plan, u_modes, v_modes,
                                    w_modes, u, v, w);

    // Transfer to host
    std::vector<amrex::Real> ulocal, vlocal, wlocal;
    ulocal.resize(n0 * n1);
    vlocal.resize(n0 * n1);
    wlocal.resize(n0 * n1);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, u.begin(), u.end(), &ulocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), &vlocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, w.begin(), w.end(), &wlocal[0]);
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

TEST_F(CombinedTest, ReadFFTDimRMObj) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read
  ReadModes rmodes(fname);
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

  // Allocate complex pointers and get plan
  fftw_plan plan;
  auto eta_modes = modes_hosgrid::allocate_plan_copy(n0, n1, plan, mFS);
  auto u_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto v_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto w_modes = modes_hosgrid::allocate_complex(n0, n1);

  // Set up output vectors
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w(n0 * n1, 0.0);

  // Get spatial data for eta
  modes_hosgrid::populate_hos_eta(rmodes, plan, eta_modes, eta);
  // Transfer to host
  std::vector<amrex::Real> etalocal;
  etalocal.resize(eta.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(),
                   &etalocal[0]);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etalocal[idx]);
      min_eta = std::min(min_eta, etalocal[idx]);
    }
  }

  // !! -- Reference values are from Grid2Grid library -- !! //

  // Check max and min
  EXPECT_NEAR(max_eta, 4.2660225672921364, 1e-10);
  EXPECT_NEAR(min_eta, -3.6415705091772734, 1e-10);

  // Store values to check velocity
  double ht[2]{-17.083096859139136, -31.347989411831382};
  double umaxref[2]{1.4526142174097856, 1.0803869257525434};
  double uminref[2]{-1.2900566679060890, -1.0015279450822712};
  double vmaxref[2]{0.42712160780051028, 0.33253336856145094};
  double vminref[2]{-0.39851521412184132, -0.30280703787995972};
  double wmaxref[2]{0.94326230376743103, 0.17192645615306479};
  double wminref[2]{-0.94562717313775668, -0.17266802905729850};
  // Get spatial data for velocity at different heights
  for (int n = 0; n < 2; ++n) {

    modes_hosgrid::populate_hos_vel(rmodes, ht[n], mX, mY, mZ, plan, u_modes,
                                    v_modes, w_modes, u, v, w);

    // Transfer to host
    std::vector<amrex::Real> ulocal, vlocal, wlocal;
    ulocal.resize(n0 * n1);
    vlocal.resize(n0 * n1);
    wlocal.resize(n0 * n1);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, u.begin(), u.end(), &ulocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), &vlocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, w.begin(), w.end(), &wlocal[0]);
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

} // namespace w2a_tests