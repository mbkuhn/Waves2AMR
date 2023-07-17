#include "read_modes.h"
#include <iterator>

void ReadModes::ascii_initialize() {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());

  double d_n1, d_n2;
  is >> d_n1 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >> g >> L >>
      T;

  // xlen, ylen, depth, and g are nondimensionalized
  // L and T are dimensional

  // Convert values
  n1 = (int)d_n1;
  n2 = (int)d_n2;
  f_out = 1.0 / dt_out;
}

void ReadModes::ascii_read(int itime) {

  if (modeT.size() == 0) {
    ascii_read_brief(itime);
  } else {
    ascii_read_full(itime);
  }

}

void ReadModes::ascii_read_full(int itime) {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());
  // Skip timesteps (each entry is complex (2) and 6 vars)
  is.ignore(18 * 6 * 2 * vec_size * (itime + 1));
  // Address edge case of itime = -1
  int i1_init = 0;
  if (itime == -1) {
    i1_init = 5;
    is.ignore(18 * 10);
    for (int i1 = 0; i1 < i1_init; ++i1) {
      modeX[i1].real(0.0);
      modeX[i1].imag(0.0);
    }
  }
  // Read modes
  int idx = 0;
  double buf_r, buf_i;
  for (int i2 = 0; i2 < n2; ++i2) {

    for (int i1 = i1_init; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeX[idx + i1].real(buf_r);
      modeX[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeY[idx + i1].real(buf_r);
      modeY[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeZ[idx + i1].real(buf_r);
      modeZ[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeT[idx + i1].real(buf_r);
      modeT[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeFS[idx + i1].real(buf_r);
      modeFS[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeFST[idx + i1].real(buf_r);
      modeFST[idx + i1].imag(buf_i);
    }
    idx += n1o2p1;
    i1_init = 0;
  }
}

void ReadModes::ascii_read_brief(int itime) {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());
  // Skip timesteps (each entry is complex (2) and 6 vars)
  is.ignore(18 * 6 * 2 * vec_size * (itime + 1));
  // Address edge case of itime = -1
  int i1_init = 0;
  if (itime == -1) {
    i1_init = 5;
    is.ignore(18 * 10);
    for (int i1 = 0; i1 < i1_init; ++i1) {
      modeX[i1].real(0.0);
      modeX[i1].imag(0.0);
    }
  }
  // Read modes
  int idx = 0;
  double buf_r, buf_i;
  for (int i2 = 0; i2 < n2; ++i2) {

    for (int i1 = i1_init; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeX[idx + i1].real(buf_r);
      modeX[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeY[idx + i1].real(buf_r);
      modeY[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeZ[idx + i1].real(buf_r);
      modeZ[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      // Don't need T
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      modeFS[idx + i1].real(buf_r);
      modeFS[idx + i1].imag(buf_i);
    }
    for (int i1 = 0; i1 < n1o2p1; ++i1) {
      is >> buf_r >> buf_i;
      // Don't need T
    }
    idx += n1o2p1;
    i1_init = 0;
  }
}