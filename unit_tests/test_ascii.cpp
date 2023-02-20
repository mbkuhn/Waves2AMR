#include "../src/read_modes.h"
#include "gtest/gtest.h"
#include <array>

namespace {

std::array<double, 8> ModeSum(double time, double initval) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read and convert nondim quantities
  ReadModes rmodes(fname, false, true);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  std::vector<std::complex<double>> mX(vsize, initval);
  std::vector<std::complex<double>> mY(vsize, initval);
  std::vector<std::complex<double>> mZ(vsize, initval);
  std::vector<std::complex<double>> mT(vsize, initval);
  std::vector<std::complex<double>> mFS(vsize, initval);
  std::vector<std::complex<double>> mFST(vsize, initval);

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
    mX_sum += std::abs(mX[i].real()) + std::abs(mX[i].imag());
    mY_sum += std::abs(mY[i].real()) + std::abs(mY[i].imag());
    mZ_sum += std::abs(mZ[i].real()) + std::abs(mZ[i].imag());
    mT_sum += std::abs(mT[i].real()) + std::abs(mT[i].imag());
    mFS_sum += std::abs(mFS[i].real()) + std::abs(mFS[i].imag());
    mFST_sum += std::abs(mFST[i].real()) + std::abs(mFST[i].imag());
  }

  double mFST_lastr = mFST[vsize - 1].real();
  double mFST_lasti = mFST[vsize - 1].imag();

  return std::array<double, 8>{mX_sum,  mY_sum,   mZ_sum,     mT_sum,
                               mFS_sum, mFST_sum, mFST_lastr, mFST_lasti};
}

std::array<double, 6> ModeSumBrief(double time, double initval) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Read and convert nondim quantities
  ReadModes rmodes(fname, false, false);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  std::vector<std::complex<double>> mX(vsize, initval);
  std::vector<std::complex<double>> mY(vsize, initval);
  std::vector<std::complex<double>> mZ(vsize, initval);
  std::vector<std::complex<double>> mFS(vsize, initval);

  // Mode quantities are set to 0 at t = 0
  rmodes.get_data(time, mX, mY, mZ, mFS);

  // Get sum of vectors
  double mX_sum = 0;
  double mY_sum = 0;
  double mZ_sum = 0;
  double mT_sum = 0;
  double mFS_sum = 0;
  double mFST_sum = 0;
  for (int i = 0; i < vsize; ++i) {
    mX_sum += std::abs(mX[i].real()) + std::abs(mX[i].imag());
    mY_sum += std::abs(mY[i].real()) + std::abs(mY[i].imag());
    mZ_sum += std::abs(mZ[i].real()) + std::abs(mZ[i].imag());
    mFS_sum += std::abs(mFS[i].real()) + std::abs(mFS[i].imag());
  }

  double mX_scndr = mX[1].real();
  double mX_scndi = mX[1].imag();

  return std::array<double, 6>{mX_sum,  mY_sum,   mZ_sum,
                               mFS_sum, mX_scndr, mX_scndi};
}

class AsciiReadTest : public testing::Test {};

TEST_F(AsciiReadTest, Init) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";

  // Read and keep nondim quantities
  ReadModes rmodes(fname);

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
  ReadModes rmodes(fname, false);

  constexpr double tol = 1e-11;
  EXPECT_NEAR(rmodes.get_f(), 0.01, tol);
  EXPECT_NEAR(rmodes.get_Tstop(), 100.0, tol * 1e2);
  EXPECT_NEAR(rmodes.get_xlen(), 2.8499216855720260E+03, tol * 5e4);
  EXPECT_NEAR(rmodes.get_ylen(), 2.8499216855720260E+03, tol * 5e4);
  EXPECT_NEAR(rmodes.get_depth(), 35.0, tol * 1e2);
  EXPECT_NEAR(rmodes.get_g(), 9.81, tol * 1e2);
}

TEST_F(AsciiReadTest, ModesInit) {

  // Get mode sums written at initialization, which are placeholders
  auto sums = ModeSum(-1, 1.0);
  // Test for expected values
  EXPECT_EQ(sums[0], 0.0);
  EXPECT_EQ(sums[1], 0.0);
  EXPECT_EQ(sums[2], 0.0);
  EXPECT_EQ(sums[3], 0.0);
  EXPECT_EQ(sums[4], 0.0);
  EXPECT_EQ(sums[5], 0.0);
}

TEST_F(AsciiReadTest, Modes0Brief) {

  // Get mode sums at t = 0
  auto sums = ModeSumBrief(0.0, -1.0);
  // Test for expected values
  EXPECT_GT(sums[0], 0.0);
  EXPECT_GT(sums[1], 0.0);
  EXPECT_GT(sums[2], 0.0);
  EXPECT_GT(sums[3], 0.0);
  EXPECT_NEAR(sums[4], 4.38291e-06, 1e-11);
  EXPECT_NEAR(sums[5], 1.12252e-06, 1e-11);
}

TEST_F(AsciiReadTest, Modes1) {

  // Get mode sums at next output time
  auto sums = ModeSum(100.0, -1.0);
  // Test for expected values
  EXPECT_GT(sums[0], 0.0);
  EXPECT_GT(sums[1], 0.0);
  EXPECT_GT(sums[2], 0.0);
  EXPECT_GT(sums[3], 0.0);
  EXPECT_GT(sums[4], 0.0);
  EXPECT_GT(sums[5], 0.0);
  EXPECT_EQ(sums[6], 3.1760843980E-20);
  EXPECT_EQ(sums[7], 6.6965350771E-20);
}

} // namespace