#include "../src/interp_to_mfab.h"
#include "gtest/gtest.h"

namespace w2a_tests {

namespace {
amrex::Real sum_multifab(amrex::MultiFab &mf, int ncomp) {
  amrex::Real f_sum = 0.0;
  f_sum += amrex::ReduceSum(
      mf, 0,
      [=] AMREX_GPU_HOST_DEVICE(
          amrex::Box const &bx,
          amrex::Array4<amrex::Real const> const &fab_arr) -> amrex::Real {
        amrex::Real f_sum_fab = 0;
        amrex::Loop(bx, ncomp,
                    [=, &f_sum_fab](int i, int j, int k, int n) noexcept {
                      f_sum_fab += fab_arr(i, j, k, n);
                    });
        return f_sum_fab;
      });
  return f_sum;
}
} // namespace

class InterpToMFabTest : public testing::Test {};

TEST_F(InterpToMFabTest, create_height_vector) {
  const int n = 17;
  const amrex::Real r = 1.005;
  const amrex::Real dz0 = 0.1;
  const amrex::Real dzn = dz0 * std::pow(r, n - 2);
  const amrex::Real l = dz0 * (1.0 - std::pow(r, n - 1)) / (1.0 - r);

  // Check proper behavior
  amrex::Vector<amrex::Real> hvec;
  int flag = interp_to_mfab::create_height_vector(hvec, n, dz0, 0.0, -l);
  // Should not fail with these inputs
  EXPECT_EQ(flag, (int)0);
  // Check beginning and ending size
  EXPECT_NEAR(2.0 * hvec[0], dz0, 1e-10);
  EXPECT_NEAR(-2.0 * hvec[1], dz0, 1e-10);
  EXPECT_NEAR(hvec[n - 1], -l + 0.5 * dzn, 1e-2);

  // Check failure flags
  flag = interp_to_mfab::create_height_vector(hvec, 4, 0.5, 1.0, 0.0, -1.0);
  EXPECT_EQ(flag, (int)1);
  flag = interp_to_mfab::create_height_vector(hvec, 1, 0.25, 0.0, -1.0);
  EXPECT_EQ(flag, (int)2);
  flag = interp_to_mfab::create_height_vector(hvec, 5, -0.25, 0.0, -1.0);
  EXPECT_EQ(flag, (int)4);
  flag = interp_to_mfab::create_height_vector(hvec, 5, 0.0, 0.0, -1.0);
  EXPECT_EQ(flag, (int)4);
  flag = interp_to_mfab::create_height_vector(hvec, 5, 0.0, 0.0, 0.0);
  EXPECT_EQ(flag, (int)4);
  flag = interp_to_mfab::create_height_vector(hvec, 5, -0.25, 0.0, 1.0);
  EXPECT_EQ(flag, (int)5);
  // flag = 3 is hard to trigger without triggering flag = 1
}

} // namespace w2a_tests