#include "AMReX.H"
#include "AMReX_FArrayBox.H"
#include "gtest/gtest.h"

namespace aux_tests {

class AMReXExampleTest : public testing::Test {};

// Not actually a test of the code, but a working example of AMReX steps
TEST_F(AMReXExampleTest, FillFAB) {
  amrex::Box bx(amrex::IntVect{0, 0, 0}, amrex::IntVect{2, 2, 1});
  amrex::FArrayBox fab(bx, 1);
  amrex::Array4<amrex::Real> const &data_slab = fab.array();
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    data_slab(i, j, k) = (amrex::Real)(i + j + k);
  });
}

} // namespace aux_tests