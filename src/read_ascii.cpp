#include "read_modes.h"
#include <iterator>

bool ReadModes::ascii_initialize() {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());

  bool file_exists = is.good();
  if (!file_exists) {
    return file_exists;
  }

  double d_n1, d_n2;
  is >> d_n1 >> d_n2 >> dt_out >> T_stop >> xlen >> ylen >> depth >> g >> L >>
      T;

  // xlen, ylen, depth, and g are nondimensionalized
  // L and T are dimensional

  // Convert values
  n1 = (int)d_n1;
  n2 = (int)d_n2;
  f_out = 1.0 / dt_out;

  return file_exists;
}

// Output is a flag where true = successful read, false = eof found
bool ReadModes::ascii_read(const int itime) {

  if (modeT.size() == 0) {
    return ascii_read_brief(itime);
  } else {
    return ascii_read_full(itime);
  }
}

bool ReadModes::ascii_read_full(const int itime) {
  bool eof_not_found = true;
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());
  // Skip timesteps (each entry is complex (2) and 6 vars)
  is.ignore(18 * 6 * 2 * vec_size * (itime + 1));
  // Check for eof, exit early if not found
  eof_not_found = eof_not_found && !is.eof();
  if (!eof_not_found) {
    return eof_not_found;
  }
  // Address edge case of itime = -1
  int i1_init = 0;
  if (itime == -1) {
    i1_init = 5;
    is.ignore(18 * 10);
    for (int i1 = 0; i1 < i1_init; ++i1) {
      modeX[i1].real(0.0);
      modeX[i1].imag(0.0);
    }
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
      return eof_not_found;
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
    // Check for eof, exit early if found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
      return eof_not_found;
    }
  }
  // Return eof_not_found value, should be true at this point
  return eof_not_found;
}

bool ReadModes::ascii_read_brief(const int itime) {
  bool eof_not_found = true;
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());
  // Skip timesteps (each entry is complex (2) and 6 vars)
  is.ignore(18 * 6 * 2 * vec_size * (itime + 1));
  // Check for eof, exit early if not found
  eof_not_found = eof_not_found && !is.eof();
  if (!eof_not_found) {
    return eof_not_found;
  }
  // Address edge case of itime = -1
  int i1_init = 0;
  if (itime == -1) {
    i1_init = 5;
    is.ignore(18 * 10);
    for (int i1 = 0; i1 < i1_init; ++i1) {
      modeX[i1].real(0.0);
      modeX[i1].imag(0.0);
    }
    // Check for eof, exit early if not found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
      return eof_not_found;
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
    // Check for eof, exit early if found
    eof_not_found = eof_not_found && !is.eof();
    if (!eof_not_found) {
      return eof_not_found;
    }
  }
  // Return eof_not_found value, should be true at this point
  return eof_not_found;
}