#include "../src/read_modes.h"
#include "gtest/gtest.h"
#include <array>

namespace {

class ReadOpsTest : public testing::Test {
protected:
  const double dt_out_ = 0.1;
  const double T_stop_ = 1000;
  const double xlen_ = 225;
  const double ylen_ = 164;
  const double depth_ = 99;
  const double g_ = 1.3;
  const double L_ = 8;
  const double T_ = 2.5;
};

TEST_F(ReadOpsTest, Dimensionalize) {
  ReadModes rmodes(dt_out_, T_stop_, xlen_, ylen_, depth_, g_, L_, T_);

  EXPECT_EQ(rmodes.get_f(), 1.0 / dt_out_ / T_);
  EXPECT_EQ(rmodes.get_Tstop(), T_stop_ * T_);
  EXPECT_EQ(rmodes.get_xlen(), xlen_ * L_);
  EXPECT_EQ(rmodes.get_ylen(), ylen_ * L_);
  EXPECT_EQ(rmodes.get_depth(), depth_ * L_);
  EXPECT_EQ(rmodes.get_g(), g_ * L_ / T_ / T_);
  EXPECT_EQ(rmodes.get_L(), L_);
  EXPECT_EQ(rmodes.get_T(), T_);
}

TEST_F(ReadOpsTest, Timestep) {
  ReadModes rmodes(dt_out_, T_stop_, xlen_, ylen_, depth_, g_, L_, T_);
  double dt = dt_out_ * T_;

  int itime = rmodes.time2step(0.0);
  EXPECT_EQ(itime, 0);

  itime = rmodes.time2step(dt);
  EXPECT_EQ(itime, 1);

  itime = rmodes.time2step(10 * dt);
  EXPECT_EQ(itime, 10);

  itime = rmodes.time2step(9.8 * dt);
  EXPECT_EQ(itime, 10);

  itime = rmodes.time2step((10.0 + 1e-8) * dt);
  EXPECT_EQ(itime, 10);
}

} // namespace