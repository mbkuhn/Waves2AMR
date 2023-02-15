#ifndef READ_MODES_H
#define READ_MODES_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
class ReadModes {
public:
  ReadModes(std::string, bool nondim = false);

  ReadModes(double dt_out_, double T_stop_, double xlen_, double ylen_,
            double depth_, double g_, double L_, double T_);

  void print_file_constants();

  void read_data(double time);

  void output_data(std::vector<double> &v1, std::vector<double> &v2,
                   std::vector<double> &v3, std::vector<double> &v4,
                   std::vector<double> &v5, std::vector<double> &v6);

  void get_data(double time, std::vector<double> &mX, std::vector<double> &mY,
                std::vector<double> &mZ, std::vector<double> &mT,
                std::vector<double> &mFS, std::vector<double> &mFST);

  // Calculate size of data for each mode variable (# of complex values)
  int get_vector_size() { return vec_size; }

  // Convert time to timestep
  int time2step(double time);

  // Output functions for testing
  int get_n1() { return n1; }
  int get_n2() { return n2; }
  double get_f() { return f_out; }
  double get_Tstop() { return T_stop; }
  double get_xlen() { return xlen; }
  double get_ylen() { return ylen; }
  double get_depth() { return depth; }
  double get_g() { return g; }
  double get_L() { return L; }
  double get_T() { return T; }

private:
  // ASCII functions
  void ascii_initialize();
  void ascii_read(int itime);

  // Dimensionalize read-in quantities
  void dimensionalize();

  // HOS data filename
  std::string m_filename;

  // HOS data dimensions
  int n1, n2;
  double dt_out, f_out, T_stop, xlen, ylen, depth, g, L, T;

  // HOS data vectors
  std::vector<double> modeX, modeY, modeZ, modeT, modeFS, modeFST;

  // HOS working dimensions
  int n1o2p1;
  int nYmode;
  int vec_size;

  // Current time index
  int itime_now;
};

#endif