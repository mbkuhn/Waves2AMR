#include "../src/amrex_interface_ops.h"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "gtest/gtest.h"

namespace {

class AMReXInterfaceTest : public testing::Test {};

TEST_F(AMReXInterfaceTest, 1D) {
  // Create init data using vector on host
  int nx = 10;
  int ny = 16;
  std::vector<double> init_vec;
  init_vec.resize(nx * ny);
  double correct_sum = 0.;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      init_vec[ix * ny + iy] = (double)(ix + iy);
      correct_sum += (double)(ix + iy);
    }
  }

  // Copy from host to device
  amrex::Gpu::DeviceVector<amrex::Real> vhost(nx * ny, 0.0);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &init_vec[0],
                   &init_vec[0] + nx * ny + 1, vhost.begin());

  // Copy vector to fab
  amrex::FArrayBox fab;
  data_amrex::copy_to_fab(nx, ny, vhost, fab);

  // Create multifab, copy to multifab
  amrex::BoxArray ba(
      amrex::Box(amrex::IntVect{0, 0, 0}, amrex::IntVect{nx - 1, ny - 1, 0}));
  amrex::DistributionMapping dm{ba};
  amrex::MultiFab mf(ba, dm, 1, 0);
  for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const amrex::Box &bx = mfi.validbox();
    amrex::FArrayBox &fab_mf = mf[mfi];
    fab_mf.copy(fab, bx, 0, bx, 0, 1);
  }

  // Sum over multifab
  amrex::Real f_sum = 0.0;
  f_sum += amrex::ReduceSum(
      mf, 0,
      [=] AMREX_GPU_HOST_DEVICE(
          amrex::Box const &bx,
          amrex::Array4<amrex::Real const> const &fab_arr) -> amrex::Real {
        amrex::Real f_sum_fab = 0;
        amrex::Loop(bx, [=, &f_sum_fab](int i, int j, int k) noexcept {
          f_sum_fab += fab_arr(i, j, k);
        });
        return f_sum_fab;
      });

  // Check result
  EXPECT_NEAR(f_sum, correct_sum, 1e-12);
}

} // namespace