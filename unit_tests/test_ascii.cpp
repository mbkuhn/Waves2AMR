#include "../src/read_methods.h"
#include "gtest/gtest.h"
#include <array>

namespace {

std::array<double, 6> ModeSum(double time, double initval) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read and convert nondim quantities
  ReadModes rmodes(fname);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  std::vector<double> mX(vsize, initval);
  std::vector<double> mY(vsize, initval);
  std::vector<double> mZ(vsize, initval);
  std::vector<double> mT(vsize, initval);
  std::vector<double> mFS(vsize, initval);
  std::vector<double> mFST(vsize, initval);

  // Mode quantities are set to 0 at t = 0
  rmodes.get_data(time, mX, mY, mZ, mT, mFS, mFST);

  // Get sum of vectors
  double mX_sum = 0;
  double mY_sum = 0;
  double mZ_sum = 0;
  double mT_sum = 0;
  double mFS_sum = 0;
  double mFST_sum = 0;
  for (int i = 0; i < vsize; ++i) {
    mX_sum += std::abs(mX[i]);
    mY_sum += std::abs(mY[i]);
    mZ_sum += std::abs(mZ[i]);
    mT_sum += std::abs(mT[i]);
    mFS_sum += std::abs(mFS[i]);
    mFST_sum += std::abs(mFST[i]);
  }

  return std::array<double, 6>{mX_sum, mY_sum,  mZ_sum,
                               mT_sum, mFS_sum, mFST_sum};
}

class AsciiReadTest : public testing::Test {};

TEST_F(AsciiReadTest, Init) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";

  // Read and keep nondim quantities
  ReadModes rmodes(fname, true);

  EXPECT_EQ(rmodes.get_n1(), 64);
  EXPECT_EQ(rmodes.get_n2(), 64);
  EXPECT_EQ(rmodes.get_f(), 1.0 / 6.2831853072E+01);
  EXPECT_EQ(rmodes.get_Tstop(), 6.2831853072E+01);
  EXPECT_EQ(rmodes.get_xlen(), 1.2566370614E+02);
  EXPECT_EQ(rmodes.get_ylen(), 1.2566370614E+02);
  EXPECT_EQ(rmodes.get_depth(), 1.5432809039);
  EXPECT_EQ(rmodes.get_g(), 1.0956862426);
  EXPECT_EQ(rmodes.get_L(), 2.2678956184E+01);
  EXPECT_EQ(rmodes.get_T(), 1.5915494309);
}

TEST_F(AsciiReadTest, InitDim) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read and convert nondim quantities
  ReadModes rmodes(fname);

  constexpr double tol = 1e-11;
  EXPECT_NEAR(rmodes.get_f(), 0.01, tol);
  EXPECT_NEAR(rmodes.get_Tstop(), 100.0, tol * 1e2);
  EXPECT_NEAR(rmodes.get_xlen(), 2.8499216855720260E+03, tol * 5e4);
  EXPECT_NEAR(rmodes.get_ylen(), 2.8499216855720260E+03, tol * 5e4);
  EXPECT_NEAR(rmodes.get_depth(), 35.0, tol * 1e2);
  EXPECT_NEAR(rmodes.get_g(), 9.81, tol * 1e2);
}

TEST_F(AsciiReadTest, Modes0) {

  // Get mode sums at t = 0
  auto sums = ModeSum(0.0, 1.0);
  // Test for expected values
  EXPECT_EQ(sums[0], 0.0);
  EXPECT_EQ(sums[1], 0.0);
  EXPECT_EQ(sums[2], 0.0);
  EXPECT_EQ(sums[3], 0.0);
  EXPECT_EQ(sums[4], 0.0);
  EXPECT_EQ(sums[5], 0.0);
}

TEST_F(AsciiReadTest, Modes1) {

  // Get mode sums at first output time
  auto sums = ModeSum(100.0, -1.0);
  // Test for expected values
  EXPECT_GT(sums[0], 0.0);
  EXPECT_GT(sums[1], 0.0);
  EXPECT_GT(sums[2], 0.0);
  EXPECT_GT(sums[3], 0.0);
  EXPECT_GT(sums[4], 0.0);
  EXPECT_GT(sums[5], 0.0);
}
} // namespace