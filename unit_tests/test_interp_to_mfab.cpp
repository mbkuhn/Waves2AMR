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

TEST_F(InterpToMFabTest, get_local_height_indices) {
  // Create height vector for test
  int nheights = 5;
  amrex::Vector<amrex::Real> hvec;
  hvec.resize(nheights);
  for (int n = 0; n < nheights; ++n) {
    hvec[n] = 0.125 - 0.25 * (amrex::Real)n;
  }

  // Make index vector for in/out
  amrex::Vector<int> indvec;

  // Make vector of const multifabs (like an AMR-Wind field)
  const int nz = 8;
  amrex::BoxArray ba(amrex::Box(amrex::IntVect{0, 0, 0},
                                amrex::IntVect{nz - 1, nz - 1, nz - 1}));
  amrex::DistributionMapping dm{ba};
  const int ncomp = 3;
  const int nghost = 3;
  const amrex::MultiFab mf0(ba, dm, ncomp, nghost);
  const amrex::MultiFab mf1(ba, dm, ncomp, nghost);
  amrex::Vector<const amrex::MultiFab *> field_fabs{&mf0, &mf1};

  // Make vectors of GpuArrays for geometry information
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev0{0.1, 0.1, 0.1};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev1{0.05, 0.05, 0.05};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx{dx_lev0,
                                                                 dx_lev1};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_all{0., 0., -1.};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo{
      problo_all, problo_all};

  // Call function being tested
  int flag = interp_to_mfab::get_local_height_indices(indvec, hvec, field_fabs,
                                                      problo, dx);

  // Check results
  int indsize = indvec.size();
  EXPECT_EQ(indsize, 4);
  int ind = 0;
  for (int n = 1; n < 5; ++n) {
    EXPECT_EQ(n, indvec[ind]);
    ++ind;
  }

  // Test scenario with partial overlap
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_po{0., 0., -0.3};
  problo[0] = problo_po;
  problo[1] = problo_po;
  amrex::Vector<int> indvec_po;
  flag = interp_to_mfab::get_local_height_indices(indvec_po, hvec, field_fabs,
                                                      problo, dx);
  indsize = indvec_po.size();
  EXPECT_EQ(indsize, 3);
  for (int n = 0; n < 3; ++n) {
    EXPECT_EQ(n, indvec_po[n]);
  }

  // Test scenario with no overlap
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_no{0., 0., 0.5};
  problo[0] = problo_no;
  problo[1] = problo_no;
  amrex::Vector<int> indvec_no;
  flag = interp_to_mfab::get_local_height_indices(indvec_no, hvec, field_fabs,
                                                      problo, dx);
  indsize = indvec_no.size();
  EXPECT_EQ(indsize, 0);
  EXPECT_EQ(flag, 1);
}

} // namespace w2a_tests