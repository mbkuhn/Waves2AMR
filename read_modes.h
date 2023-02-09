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
  int itime_now{0};
};

#endif