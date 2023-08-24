#include "interp_to_mfab.h"
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

void initialize_eta(amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta, int spd_nx,
                    int spd_ny) {
  // Get pointers to eta because it is on device
  auto *eta_ptr = HOS_eta.data();
  // Get size for loop
  const int n2D = HOS_eta.size();
  // Multiply each component of velocity vectors in given range of indices to
  // dimensionalize the velocity
  int j_half = spd_ny / 2;
  amrex::ParallelFor(n2D, [=] AMREX_GPU_DEVICE(int n) {
    // Row-major - get i and j
    int j = n / spd_nx;
    int i = n - j * spd_nx;
    eta_ptr[n] = (j < j_half ? 1.0 : -1.0);
  });
}

amrex::Real error_eta(amrex::MultiFab &mf, int ny, amrex::Real zsl,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {
  amrex::Real f_sum = 0.0;
  int j_half = ny / 2;
  f_sum += amrex::ReduceSum(
      mf, 0,
      [=] AMREX_GPU_HOST_DEVICE(
          amrex::Box const &bx,
          amrex::Array4<amrex::Real const> const &fab_arr) -> amrex::Real {
        amrex::Real f_sum_fab = 0;
        amrex::Loop(bx, [=, &f_sum_fab](int i, int j, int k) noexcept {
          amrex::Real sol = j < j_half ? 1.0 : -1.0;
          const amrex::Real z = plo[2] + (k + 0.5) * dx[2];
          // Avoid interpolation points to keep things simple
          if (j != j_half - 1 && j != ny - 1) {
            f_sum_fab += std::abs(fab_arr(i, j, k) - (sol + zsl - z));
          }
        });
        return f_sum_fab;
      });
  return f_sum;
}

void initialize_velocity_component(amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
                                   int spd_nx, int spd_ny) {
  // Get pointers to eta because it is on device
  auto *u_ptr = HOS_u.data();
  // Get size for loop
  const int n2D = HOS_u.size();
  // Multiply each component of velocity vectors in given range of indices to
  // dimensionalize the velocity
  int i_half = spd_nx / 2;
  amrex::ParallelFor(n2D, [=] AMREX_GPU_DEVICE(int n) {
    // Row-major - get i and j
    int j = n / spd_nx;
    int i = n - j * spd_nx;
    u_ptr[n] = (i < i_half ? 1.0 : -1.0);
  });
}

amrex::Real error_velocity(amrex::MultiFab &mf, int nx) {
  amrex::Real f_sum = 0.0;
  int i_half = nx / 2;
  f_sum += amrex::ReduceSum(
      mf, 0,
      [=] AMREX_GPU_HOST_DEVICE(
          amrex::Box const &bx,
          amrex::Array4<amrex::Real const> const &fab_arr) -> amrex::Real {
        amrex::Real f_sum_fab = 0;
        amrex::Loop(bx, 3,
                    [=, &f_sum_fab](int i, int j, int k, int n) noexcept {
                      amrex::Real sol = i < i_half ? 1.0 : -1.0;
                      // Avoid interpolation points to keep things simple
                      if (i != i_half - 1 && i != nx - 1) {
                        f_sum_fab += std::abs(fab_arr(i, j, k, n) - sol);
                      }
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
  flag = interp_to_mfab::create_height_vector(hvec, 2, 0.5, 0.0, -1.0);
  EXPECT_EQ(flag, (int)2);
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

  // Test scenario where mfab is between points, but no points are in mfab
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_wi{0., 0., -0.8};
  problo[0] = problo_wi;
  problo[1] = problo_wi;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wi0{0.01, 0.01, 0.01};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wi1{0.005, 0.005, 0.005};
  dx[0] = dx_wi0;
  dx[1] = dx_wi1;
  amrex::Vector<int> indvec_wi;
  flag = interp_to_mfab::get_local_height_indices(indvec_wi, hvec, field_fabs,
                                                  problo, dx);
  indsize = indvec_wi.size();
  EXPECT_EQ(indsize, 2);
  EXPECT_EQ(flag, 0);
  for (int n = 0; n < indsize; ++n) {
    EXPECT_EQ(n + 3, indvec_wi[n]);
  }
}

TEST_F(InterpToMFabTest, check_lateral_overlap) {
  // Make vectors of GpuArrays for geometry information
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev0{1. / 32., 1. / 32.,
                                                       1. / 32.};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev1{1. / 64., 1. / 64.,
                                                       1. / 64.};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx{dx_lev0,
                                                                 dx_lev1};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_all{0., 0., -0.5};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo{
      problo_all, problo_all};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probhi_all{1., 1., 0.5};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> probhi{
      probhi_all, probhi_all};

  // Make vector of const multifabs (like an AMR-Wind field)
  const int nz = 8;
  amrex::BoxArray ba_lo(amrex::Box(amrex::IntVect{0, 0, 0},
                                   amrex::IntVect{nz - 1, nz - 1, nz - 1}));
  amrex::BoxArray ba_hi0(amrex::Box(amrex::IntVect{32 - nz, 32 - nz, 32 - nz},
                                    amrex::IntVect{32 - 1, 32 - 1, 32 - 1}));
  amrex::BoxArray ba_hi1(amrex::Box(amrex::IntVect{64 - nz, 64 - nz, 64 - nz},
                                    amrex::IntVect{64 - 1, 64 - 1, 64 - 1}));
  amrex::BoxArray ba_mid0(amrex::Box(amrex::IntVect{16 - nz, 16 - nz, 16 - nz},
                                     amrex::IntVect{16 - 1, 16 - 1, 16 - 1}));
  amrex::BoxArray ba_mid1(amrex::Box(amrex::IntVect{48 - nz, 48 - nz, 48 - nz},
                                     amrex::IntVect{48 - 1, 48 - 1, 48 - 1}));
  // ranges: lo [0., 0.25], mid [0.25, 0.5], and hi [0.75, 1.0]. cc w/ ghosts:
  // [-0.078125, 0.328125], [0.171875, 0.578125], and [0.671875, 1.078125]

  amrex::DistributionMapping dm_lo{ba_lo};
  amrex::DistributionMapping dm_hi0{ba_hi0};
  amrex::DistributionMapping dm_hi1{ba_hi1};
  amrex::DistributionMapping dm_mid0{ba_mid0};
  amrex::DistributionMapping dm_mid1{ba_mid1};

  const int ncomp = 3;
  const int nghost = 3;
  amrex::MultiFab mf_lo0(ba_lo, dm_lo, ncomp, nghost);
  amrex::MultiFab mf_lo1(ba_lo, dm_lo, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs_lo{&mf_lo0, &mf_lo1};
  amrex::MultiFab mf_hi0(ba_hi0, dm_hi0, ncomp, nghost);
  amrex::MultiFab mf_hi1(ba_hi1, dm_hi1, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs_hi{&mf_hi0, &mf_hi1};
  amrex::MultiFab mf_mid0(ba_mid0, dm_mid0, ncomp, nghost);
  amrex::MultiFab mf_mid1(ba_mid1, dm_mid1, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs_mid{&mf_mid0, &mf_mid1};

  // More precise checks are to ensure use of cell-centered locations

  // Checks with lo mfabs
  int flag = interp_to_mfab::check_lateral_overlap_hi(
      21. / 32. + 1e-8, 0, field_fabs_lo, problo, probhi, dx);
  EXPECT_EQ(flag, 0);
  flag = interp_to_mfab::check_lateral_overlap_hi(
      21.5 / 32. + 1e-8, 0, field_fabs_lo, problo, probhi, dx);
  EXPECT_EQ(flag, 1);
  flag = interp_to_mfab::check_lateral_overlap_lo(0.1, 0, field_fabs_lo, problo,
                                                  probhi, dx);
  EXPECT_EQ(flag, 1);

  // Checks with hi mfabs
  flag = interp_to_mfab::check_lateral_overlap_hi(0.1, 0, field_fabs_hi, problo,
                                                  probhi, dx);
  EXPECT_EQ(flag, 1);
  flag = interp_to_mfab::check_lateral_overlap_lo(
      21. / 32. + 1e-8, 0, field_fabs_hi, problo, probhi, dx);
  EXPECT_EQ(flag, 0);

  // Checks with mid mfabs
  flag = interp_to_mfab::check_lateral_overlap_hi(0.1, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 0);
  flag = interp_to_mfab::check_lateral_overlap_lo(0.1, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 0);
  flag = interp_to_mfab::check_lateral_overlap_hi(0.3, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 1);
  flag = interp_to_mfab::check_lateral_overlap_lo(0.5, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 1);
  flag = interp_to_mfab::check_lateral_overlap_hi(0.9, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 1);
  flag = interp_to_mfab::check_lateral_overlap_lo(0.9, 0, field_fabs_mid,
                                                  problo, probhi, dx);
  EXPECT_EQ(flag, 1);
}

TEST_F(InterpToMFabTest, interp_eta_to_multifab_lateral) {
  // Set up 2D dimensions
  const int spd_nx = 10, spd_ny = 20;
  const amrex::Real spd_dx = 0.1, spd_dy = 0.05;

  // Set up eta data
  amrex::Gpu::DeviceVector<amrex::Real> deta(spd_nx * spd_ny, 0.0);
  initialize_eta(deta, spd_nx, spd_ny);
  // Call function to populate
  amrex::Gpu::DeviceVector<amrex::Real> etavec;
  amrex::Gpu::DeviceVector<amrex::Real> vvec;
  amrex::Gpu::DeviceVector<amrex::Real> wvec;
  // U, V, W are all initialized, checked the same way
  etavec.insert(etavec.end(), deta.begin(), deta.end());

  // Set up target mfabs and mesh
  const int nz = 8;
  amrex::BoxArray ba(amrex::Box(amrex::IntVect{0, 0, 0},
                                amrex::IntVect{nz - 1, nz - 1, nz - 1}));
  amrex::DistributionMapping dm{ba};
  const int ncomp = 1;
  const int nghost = 3;
  amrex::MultiFab mf(ba, dm, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> field_fabs{&mf};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev{0.125, 0.125, 0.125};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx{dx_lev};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_lev{0., 0., -1.};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo{
      problo_lev};

  // Perform interpolation
  const amrex::Real zsl = 0.0;
  interp_to_mfab::interp_eta_to_levelset_field(
      spd_nx, spd_ny, spd_dx, spd_dy, zsl, etavec, field_fabs, problo, dx);
  // Check error directly
  const amrex::Real error = error_eta(mf, nz, zsl, problo[0], dx[0]);
  EXPECT_NEAR(error, 0.0, 1e-8);
  //  Note: this checks variation in y, ensures i, j indices aren't mixed up
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
  amrex::Gpu::DeviceVector<amrex::Real> uvec; //{dv2, dv2, dv2};
  amrex::Gpu::DeviceVector<amrex::Real> vvec; //{dv2, dv3, dv4};
  amrex::Gpu::DeviceVector<amrex::Real> wvec; //{dv4, dv4, dv3};
  // U velocity is dv2 at all three heights
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  // V velocity is {dv2, dv3, dv4}
  vvec.insert(vvec.end(), dv2.begin(), dv2.end());
  vvec.insert(vvec.end(), dv3.begin(), dv3.end());
  vvec.insert(vvec.end(), dv4.begin(), dv4.end());
  // W velocity is {dv4, dv4, dv3}
  wvec.insert(wvec.end(), dv4.begin(), dv4.end());
  wvec.insert(wvec.end(), dv4.begin(), dv4.end());
  wvec.insert(wvec.end(), dv3.begin(), dv3.end());

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
  interp_to_mfab::interp_velocity_to_field(spd_nx, spd_ny, spd_dx, spd_dy,
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

TEST_F(InterpToMFabTest, interp_velocity_to_multifab_modindices) {
  // Set up 2D dimensions
  const int spd_nx = 10, spd_ny = 20;
  const amrex::Real spd_dx = 0.1, spd_dy = 0.05;
  // Set up heights
  int nheights = 5;
  amrex::Vector<amrex::Real> hvec;
  hvec.resize(nheights);
  hvec[0] = 1.0;
  hvec[1] = 0.0;
  hvec[2] = -3. / 4.;
  hvec[3] = -1.0;
  hvec[4] = -2.0;
  // Set up velocity data
  amrex::Gpu::DeviceVector<amrex::Real> dv2(spd_nx * spd_ny, 2.0);
  amrex::Gpu::DeviceVector<amrex::Real> dv3(spd_nx * spd_ny, 3.0);
  amrex::Gpu::DeviceVector<amrex::Real> dv4(spd_nx * spd_ny, 4.0);
  amrex::Gpu::DeviceVector<amrex::Real> uvec; //{dv2, dv2, dv2};
  amrex::Gpu::DeviceVector<amrex::Real> vvec; //{dv2, dv3, dv4};
  amrex::Gpu::DeviceVector<amrex::Real> wvec; //{dv4, dv4, dv3};
  // There are 5 heights, but only 3 overlap with amr domain
  // U velocity is dv2 at all three heights
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  uvec.insert(uvec.end(), dv2.begin(), dv2.end());
  // V velocity is {dv2, dv3, dv4}
  vvec.insert(vvec.end(), dv2.begin(), dv2.end());
  vvec.insert(vvec.end(), dv3.begin(), dv3.end());
  vvec.insert(vvec.end(), dv4.begin(), dv4.end());
  // W velocity is {dv4, dv4, dv3}
  wvec.insert(wvec.end(), dv4.begin(), dv4.end());
  wvec.insert(wvec.end(), dv4.begin(), dv4.end());
  wvec.insert(wvec.end(), dv3.begin(), dv3.end());

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
  // Set available indices to the 3 middle heights
  amrex::Vector<int> indvec{1, 2, 3};

  // Perform interpolation
  interp_to_mfab::interp_velocity_to_field(spd_nx, spd_ny, spd_dx, spd_dy,
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

TEST_F(InterpToMFabTest, interp_velocity_to_multifab_lateral) {
  // Set up 2D dimensions
  const int spd_nx = 10, spd_ny = 20;
  const amrex::Real spd_dx = 0.1, spd_dy = 0.05;
  // Set up heights - not concerned with vertical interp
  int nheights = 2;
  amrex::Vector<amrex::Real> hvec;
  hvec.resize(nheights);
  hvec[0] = 1.5;
  hvec[1] = -2.0;
  // Set up velocity data
  amrex::Gpu::DeviceVector<amrex::Real> dv(spd_nx * spd_ny, 0.0);
  initialize_velocity_component(dv, spd_nx, spd_ny);
  // Call function to populate
  amrex::Gpu::DeviceVector<amrex::Real> uvec;
  amrex::Gpu::DeviceVector<amrex::Real> vvec;
  amrex::Gpu::DeviceVector<amrex::Real> wvec;
  // U, V, W are all initialized, checked the same way
  uvec.insert(uvec.end(), dv.begin(), dv.end());
  uvec.insert(uvec.end(), dv.begin(), dv.end());
  vvec.insert(vvec.end(), dv.begin(), dv.end());
  vvec.insert(vvec.end(), dv.begin(), dv.end());
  wvec.insert(wvec.end(), dv.begin(), dv.end());
  wvec.insert(wvec.end(), dv.begin(), dv.end());

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
  interp_to_mfab::interp_velocity_to_field(spd_nx, spd_ny, spd_dx, spd_dy,
                                           indvec, hvec, uvec, vvec, wvec,
                                           field_fabs, problo, dx);
  // Check error directly
  const amrex::Real error = error_velocity(mf, nz);
  EXPECT_NEAR(error, 0.0, 1e-8);
  //  Note: this checks variation in x, ensures i, j indices aren't mixed up
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

TEST_F(InterpToMFabTest, linear_interp2D) {
  // Test points directly
  amrex::Real x1 = 1.0, y1 = 1.0;
  amrex::Real x0 = 0.0, y0 = 0.0;
  amrex::Real a =
      interp_to_mfab::linear_interp2D(1., 0., 0., 0., 0., 0., x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp2D(0., 1., 0., 0., 1., 0., x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp2D(0., 0., 1., 0., 0., 1., x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  a = interp_to_mfab::linear_interp2D(0., 0., 0., 1., 1., 1., x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.0, 1e-15);
  // Test line in each direction, interp and extrap
  a = interp_to_mfab::linear_interp2D(1., 0., 0., 0., 0.5, 0., x0, y0, x1, y1);
  EXPECT_NEAR(a, 0.5, 1e-15);
  a = interp_to_mfab::linear_interp2D(1., 0., 0., 0., -0.5, 0., x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.5, 1e-15);
  a = interp_to_mfab::linear_interp2D(0., 0., 1., 0., 0., 0.5, x0, y0, x1, y1);
  EXPECT_NEAR(a, 0.5, 1e-15);
  a = interp_to_mfab::linear_interp2D(0., 0., 1., 0., 0., 1.5, x0, y0, x1, y1);
  EXPECT_NEAR(a, 1.5, 1e-15);
}

} // namespace w2a_tests