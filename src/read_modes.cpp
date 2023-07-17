#include "read_modes.h"

ReadModes::ReadModes(std::string filename, bool allmodes)
    : m_filename(filename) {
  // Set time index value
  itime_now = 0;
  // TODO: Determine filetype

  // Initialize (TODO: according to file type)
  ascii_initialize();

  // Get working dimensions
  n1o2p1 = n1 / 2 + 1;

  // Calculate size of mode vectors
  vec_size = n2 * n1o2p1;

  // Set size of mode vectors
  modeX.resize(vec_size);
  modeY.resize(vec_size);
  modeZ.resize(vec_size);
  modeFS.resize(vec_size);
  // These modes are optional
  if (allmodes) {
    modeT.resize(vec_size);
    modeFST.resize(vec_size);
  }

  // Dimensionalize all nondim scalar quantities
  dimensionalize();
}

ReadModes::ReadModes(double dt_out_, double T_stop_, double xlen_, double ylen_,
                     double depth_, double g_, double L_, double T_)
    : dt_out(dt_out_), T_stop(T_stop_), xlen(xlen_), ylen(ylen_), depth(depth_),
      g(g_), L(L_), T(T_) {
  // ^Manually set metadata for the sake of testing, do other expected steps
  // No treatment of integer dimensions needed at the moment

  // Initialize time index
  itime_now = 0;
  // Initialize output frequency
  f_out = 1.0 / dt_out;

  dimensionalize();
}

int ReadModes::time2step(double time) {
  // Return -1 if time is negative
  if (time < 0.0) return -1;
  // Look for same time or after
  bool done = false;
  while (!done) {
    // Use a tolerance to avoid skipping close matches
    if (itime_now * dt_out < (time - dt_out * 1e-8)) {
      ++itime_now;
    } else if ((itime_now - 1) * dt_out > time) {
      --itime_now;
    } else {
      done = true;
    }
  }
  return itime_now;
}

void ReadModes::dimensionalize() {
  // Dimensionalize read-in nondim quantities
  dt_out *= T;
  f_out /= T;
  T_stop *= T;
  xlen *= L;
  ylen *= L;
  depth *= L;
  g *= L / T / T;
}

void ReadModes::read_data(double time) {
  int itime = time2step(time);
  // Read (TODO: according to file type)
  ascii_read(itime);
}

void ReadModes::output_data(std::vector<std::complex<double>> &v1,
                            std::vector<std::complex<double>> &v2,
                            std::vector<std::complex<double>> &v3,
                            std::vector<std::complex<double>> &v4,
                            std::vector<std::complex<double>> &v5,
                            std::vector<std::complex<double>> &v6) {
  // Copy class variables to input/output variables
  std::copy(modeX.begin(), modeX.end(), v1.begin());
  std::copy(modeY.begin(), modeY.end(), v2.begin());
  std::copy(modeZ.begin(), modeZ.end(), v3.begin());
  std::copy(modeT.begin(), modeT.end(), v4.begin());
  std::copy(modeFS.begin(), modeFS.end(), v5.begin());
  std::copy(modeFST.begin(), modeFST.end(), v6.begin());
}

void ReadModes::output_data(std::vector<std::complex<double>> &v1,
                            std::vector<std::complex<double>> &v2,
                            std::vector<std::complex<double>> &v3,
                            std::vector<std::complex<double>> &v4) {
  // Copy class variables to input/output variables
  std::copy(modeX.begin(), modeX.end(), v1.begin());
  std::copy(modeY.begin(), modeY.end(), v2.begin());
  std::copy(modeZ.begin(), modeZ.end(), v3.begin());
  std::copy(modeFS.begin(), modeFS.end(), v4.begin());
}

void ReadModes::get_data(double time, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mT,
                         std::vector<std::complex<double>> &mFS,
                         std::vector<std::complex<double>> &mFST) {
  // Read data
  read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mT, mFS, mFST);
}

void ReadModes::get_data(double time, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mFS) {
  // Read data
  read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mFS);
}

void ReadModes::print_file_constants() {
  std::cout << "f_out " << f_out << " T " << T << " T_stop " << T_stop
            << std::endl;
  std::cout << "n1 " << n1 << " n2 " << n2 << std::endl;
  std::cout << "xlen " << xlen << " ylen " << ylen << std::endl;
  std::cout << "depth " << depth << " g " << g << " L " << L << std::endl;
}