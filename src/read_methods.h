#include "read_modes.h"
#include <iterator>

inline void ReadModes::ascii_initialize() {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());

  double d_n1, d_n2;
  is >> d_n1 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >> g >> L >>
      T;

  // xlen, ylen, depth, and g are nondimensionalized
  // g, L, and T are dimensional

  // Convert values
  n1 = (int)d_n1;
  n2 = (int)d_n2;
  f_out = 1.0 / dt_out;
}

inline void ReadModes::ascii_read(int itime) {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());
  // Skip header (each entry is 18 chars)
  is.ignore(1 + 18 * 10);
  // Skip timesteps (each entry is complex (2) and 6 vars)
  is.ignore(18 * 6 * 2 * vec_size * itime);
  // Read modes
  int idx = 0;
  for (int i2 = 0; i2 < n2; ++i2) {

    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeX[idx + i1];
    }
    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeY[idx + i1];
    }
    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeZ[idx + i1];
    }
    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeT[idx + i1];
    }
    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeFS[idx + i1];
    }
    for (int i1 = 0; i1 < 2 * n1o2p1; ++i1) {
      is >> modeFST[idx + i1];
    }
    idx += 2 * n1o2p1;
  }
};