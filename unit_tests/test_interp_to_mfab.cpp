#include "../src/interp_to_mfab.h"
#include "gtest/gtest.h"

namespace w2a_tests {

namespace {
amrex::Real sum_multifab(amrex::MultiFab &mf, int icomp) {
  amrex::Real f_sum = 0.0;
  f_sum += amrex::ReduceSum(
      mf, 0,
      [=] AMREX_GPU_HOST_DEVICE(
          amrex::Box const &bx,
          amrex::Array4<amrex::Real const> const &fab_arr) -> amrex::Real {
        amrex::Real f_sum_fab = 0;
        amrex::Loop(bx, [=, &f_sum_fab](int i, int j, int k) noexcept {
          f_sum_fab += fab_arr(i, j, k, icomp);
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
  amrex::MultiFab mf0(ba, dm, ncomp, nghost);
  amrex::MultiFab mf1(ba, dm, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs{&mf0, &mf1};

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
  EXPECT_EQ(indsize, 5);
  for (int n = 0; n < 5; ++n) {
    EXPECT_EQ(n, indvec[n]);
  }

  // Test scenario with partial overlap
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_po{0., 0., -0.3};
  problo[0] = problo_po;
  problo[1] = problo_po;
  amrex::Vector<int> indvec_po;
  flag = interp_to_mfab::get_local_height_indices(indvec_po, hvec, field_fabs,
                                                  problo, dx);
  indsize = indvec_po.size();
  EXPECT_EQ(indsize, 4);
  for (int n = 0; n < 4; ++n) {
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

TEST_F(InterpToMFabTest, interp_velocity_to_multifab) {
  // Set up 2D dimensions
  const int spd_nx = 10, spd_ny = 20;
  const amrex::Real spd_dx = 0.1, spd_dy = 0.05;
  // Set up heights
  int nheights = 3;
  amrex::Vector<amrex::Real> hvec;
  hvec.resize(nheights);
  hvec[0] = 0.0;
  hvec[1] = -3. / 4.;
  hvec[2] = -1.0;
  // Set up velocity data
  amrex::Gpu::DeviceVector<amrex::Real> dv2(spd_nx * spd_ny, 2.0);
  amrex::Gpu::DeviceVector<amrex::Real> dv3(spd_nx * spd_ny, 3.0);
  amrex::Gpu::DeviceVector<amrex::Real> dv4(spd_nx * spd_ny, 4.0);
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> uvec{dv2, dv2, dv2};
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> vvec{dv2, dv3, dv4};
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> wvec{dv4, dv4, dv3};
  // Set up target mfabs and mesh
  const int nz = 8;
  amrex::BoxArray ba(amrex::Box(amrex::IntVect{0, 0, 0},
                                amrex::IntVect{nz - 1, nz - 1, nz - 1}));
  amrex::DistributionMapping dm{ba};
  const int ncomp = 3;
  const int nghost = 3;
  amrex::MultiFab mf(ba, dm, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs{&mf};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev{0.125, 0.125, 0.125};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx{dx_lev};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_lev{0., 0., -1.};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo{
      problo_lev};
  // Get indices
  amrex::Vector<int> indvec;
  int flag = interp_to_mfab::get_local_height_indices(indvec, hvec, field_fabs,
                                                      problo, dx);
  EXPECT_EQ(flag, 0);
  // Perform interpolation
  interp_to_mfab::interp_velocity_to_multifab(spd_nx, spd_ny, spd_dx, spd_dy,
                                              indvec, hvec, uvec, vvec, wvec,
                                              field_fabs, problo, dx);
  // Check sum
  const amrex::Real mf_sum_u = sum_multifab(*field_fabs[0], 0);
  const amrex::Real mf_sum_v = sum_multifab(*field_fabs[0], 1);
  const amrex::Real mf_sum_w = sum_multifab(*field_fabs[0], 2);
  const amrex::Real u_sum = 2.0 * nz * nz * nz;
  const amrex::Real v_sum =
      ((2.0 + (4. / 3.) * (1. / 16.)) + (2.0 + (4. / 3.) * (3. / 16.)) +
       (2.0 + (4. / 3.) * (5. / 16.)) + (2.0 + (4. / 3.) * (7. / 16.)) +
       (2.0 + (4. / 3.) * (9. / 16.)) + (2.0 + (4. / 3.) * (11. / 16.)) +
       (3.0 + (4. / 1.) * (1. / 16.)) + (3.0 + (4. / 1.) * (3. / 16.))) *
      nz * nz;
  const amrex::Real w_sum =
      (6.0 * 4.0 + (4.0 - 1. / 4.) + (3.0 + 1. / 4.)) * nz * nz;
  EXPECT_NEAR(mf_sum_u, u_sum, 1e-8);
  EXPECT_NEAR(mf_sum_v, v_sum, 1e-8);
  EXPECT_NEAR(mf_sum_w, w_sum, 1e-8);
  //  Note: this only checks variation in z
}

TEST_F(InterpToMFabTest, linear_interp) {
  // Test points directly
  amrex::Real x1 = 1.0, y1 = 1.0, z1 = 1.0;
  amrex::Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  amrex::Real a = interp_to_mfab::linear_interp(
      1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  // Test line in each direction, interp and extrap
  a = interp_to_mfab::linear_interp(1., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 0.5, 1e-15);
  a = interp_to_mfab::linear_interp(1., 0., 0., 0., 0., 0., 0., 0., -0.5, 0.,
                                    0., x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.5, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 1., 0., 0., 0., 0., 0., 0., 0.5, 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 0.5, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 1., 0., 0., 0., 0., 0., 0., 1.5, 0.,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 1.5, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.5,
                                    x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, 0.5, 1e-15);
  a = interp_to_mfab::linear_interp(0., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
                                    -0.5, x0, y0, z0, x1, y1, z1);
  EXPECT_NEAR(a, -0.5, 1e-15);
}

} // namespace w2a_tests