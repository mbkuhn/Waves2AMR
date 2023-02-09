#include "read_modes.h"

void ReadModes::ascii_initialize() {
  std::stringstream fname;
  fname << m_filename;
  std::ifstream is(fname.str());

  double d_n1, d_n2;
  is >> d_n1 >> d_n2 >> f_out >> T_stop >> xlen >> ylen >> depth >> g >> L >> T;
  n1 = (int)d_n1;
  n2 = (int)d_n2;

  // Get working dimensions
  n1o2p1 = n1/2 + 1;

  // Set size of mode vectors
  
}

void ReadModes::ascii_read(int itime) {

  /*std::copy(
    std::istream_iterator<double>(istream), 
    2*n1o2p1, 
    std::back_inserter(modeX));*/
};