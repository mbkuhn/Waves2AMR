#ifndef READ_MODES_H
#define READ_MODES_H
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
class ReadModes {
public:
  ReadModes(std::string, bool allmodes = false);

  ReadModes(double dt_out_, double T_stop_, double xlen_, double ylen_,
            double depth_, double g_, double L_, double T_);

  void print_file_constants();

  void read_data(double time);

  void output_data(std::vector<std::complex<double>> &v1,
                   std::vector<std::complex<double>> &v2,
                   std::vector<std::complex<double>> &v3,
                   std::vector<std::complex<double>> &v4,
                   std::vector<std::complex<double>> &v5,
                   std::vector<std::complex<double>> &v6);

  void get_data(double time, std::vector<std::complex<double>> &mX,
                std::vector<std::complex<double>> &mY,
                std::vector<std::complex<double>> &mZ,
                std::vector<std::complex<double>> &mT,
                std::vector<std::complex<double>> &mFS,
                std::vector<std::complex<double>> &mFST);

  void output_data(std::vector<std::complex<double>> &v1,
                   std::vector<std::complex<double>> &v2,
                   std::vector<std::complex<double>> &v3,
                   std::vector<std::complex<double>> &v4);

  void get_data(double time, std::vector<std::complex<double>> &mX,
                std::vector<std::complex<double>> &mY,
                std::vector<std::complex<double>> &mZ,
                std::vector<std::complex<double>> &mFS);

  // Calculate size of data for each mode variable (# of complex values)
  int get_vector_size() { return vec_size; }

  // Convert time to timestep
  int time2step(double time);

  // Convert dimensions for fortran reading
  /* bool fortran2cpp() {} */

  // Output functions for use
  int get_first_dimension() { return (from_fortran ? n2 : n1); }
  int get_second_dimension() { return (from_fortran ? n1 : n2); }

  // Output functions of parameters
  int get_n1() { return n1; }
  int get_n2() { return n2; }
  double get_dtout() { return dt_out; }
  double get_f() { return f_out; }
  double get_Tstop() { return T_stop; }
  double get_xlen() { return xlen; }
  double get_ylen() { return ylen; }
  double get_depth() { return depth; }
  double get_g() { return g; }
  double get_nondim_dtout() { return dt_out / T; }
  double get_nondim_f() { return f_out * T; }
  double get_nondim_Tstop() { return T_stop / T; }
  double get_nondim_xlen() { return xlen / L; }
  double get_nondim_ylen() { return ylen / L; }
  double get_nondim_depth() { return depth / L; }
  double get_nondim_g() { return g / L * T * T; }
  double get_L() { return L; }
  double get_T() { return T; }

private:
  // ASCII functions
  void ascii_initialize();
  void ascii_read(int itime);
  void ascii_read_full(int itime);
  void ascii_read_brief(int itime);

  // Dimensionalize read-in quantities
  void dimensionalize();

  // HOS data filename
  std::string m_filename;

  // HOS data dimensions
  int n1, n2;
  double dt_out, f_out, T_stop, xlen, ylen, depth, g, L, T;

  // HOS data vectors
  std::vector<std::complex<double>> modeX, modeY, modeZ, modeT, modeFS, modeFST;

  // HOS working dimensions
  int n1o2p1;
  int nYmode;
  int vec_size;

  // Current time index
  int itime_now;

  // Relate between format of modes and FFT (only one option now)
  const bool from_fortran = true;
};

#endif