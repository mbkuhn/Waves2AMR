#include "../src/read_methods.h"
#include "gtest/gtest.h"

namespace {
class AsciiReadTest : public testing::Test {};

TEST_F(AsciiReadTest, Init) {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";

  // Read and keep nondim quantities
  ReadModes rmodes(fname, true);

  EXPECT_EQ(rmodes.get_n1(), 64);
  EXPECT_EQ(rmodes.get_n2(), 64);
  EXPECT_EQ(rmodes.get_f(), 1.0/6.2831853072E+01);
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
} // namespace