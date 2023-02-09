#ifndef READ_MODES_H
#define READ_MODES_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
class ReadModes {
public:
  ReadModes(std::string);

  void print_file_constants();

  void read_data(double time);

  void output_data(std::vector<double> &v1, std::vector<double> &v2,
                   std::vector<double> &v3, std::vector<double> &v4,
                   std::vector<double> &v5, std::vector<double> &v6);

  void get_data(double time, std::vector<double> &mX, std::vector<double> &mY,
                 std::vector<double> &mZ, std::vector<double> &mT,
                 std::vector<double> &mFS, std::vector<double> &mFST);

private:
  // Convert time to timestep
  int time2step(double time);

  // ASCII functions
  void ascii_initialize();
  void ascii_read(int itime);

  // HOS data filename
  std::string m_filename;

  // HOS data dimensions
  int n1, n2;
  double f_out, T_stop, xlen, ylen, depth, g, L, T;

  // HOS data vectors
  std::vector<double> modeX, modeY, modeZ, modeT, modeFS, modeFST;

  // HOS working dimensions
  int n1o2p1;
  int nYmode;

  // Current time index
  int itime_now;
};

inline ReadModes::ReadModes(std::string filename) : m_filename(filename) {
  // Set time index value
  itime_now = 0;
  // TODO: Determine filetype

  // Initialize (TODO: according to file type)
  ascii_initialize();
}

inline int ReadModes::time2step(double time) {
  // Look for same time or after
  if (itime_now / f_out < time) {
    ++itime_now;
  } else if ((itime_now - 1) / f_out > time) {
    --itime_now;
  }
  return itime_now;
}

inline void ReadModes::read_data(double time) {
  int itime = time2step(time);
  // Read (TODO: according to file type)
  ascii_read(itime);
}

inline void ReadModes::get_data(double time, std::vector<double> &mX,
                         std::vector<double> &mY, std::vector<double> &mZ,
                         std::vector<double> &mT, std::vector<double> &mFS,
                         std::vector<double> &mFST) {
  // Read data
  read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mT, mFS, mFST);
}

inline void ReadModes::print_file_constants() {
  std::cout << "f_out " << f_out << " T " << T << " T_stop " << T_stop
            << std::endl;
  std::cout << "n1 " << n1 << " n2 " << n2 << std::endl;
  std::cout << "xlen " << xlen << " ylen " << ylen << std::endl;
  std::cout << "depth " << depth << " g " << g << " L " << L << std::endl;
}

inline void ReadModes::output_data(std::vector<double> &v1, std::vector<double> &v2,
                            std::vector<double> &v3, std::vector<double> &v4,
                            std::vector<double> &v5, std::vector<double> &v6) {
  // Copy class variables to input/output variables
  std::copy(modeX.begin(), modeX.end(), std::back_inserter(v1));
  std::copy(modeY.begin(), modeY.end(), std::back_inserter(v2));
  std::copy(modeZ.begin(), modeZ.end(), std::back_inserter(v3));
  std::copy(modeT.begin(), modeT.end(), std::back_inserter(v4));
  std::copy(modeFS.begin(), modeFS.end(), std::back_inserter(v5));
  std::copy(modeFST.begin(), modeFST.end(), std::back_inserter(v6));
}

#endif