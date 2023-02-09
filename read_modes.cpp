#include "read_modes.h"

ReadModes::ReadModes(std::string filename) : m_filename(filename) {
  // TODO: Determine filetype

  // Initialize (TODO: according to file type)
  ascii_initialize();
}

void ReadModes::read_data(double time) {
  // Read (TODO: according to file type)
  ascii_read(time);
}

void ReadModes::read_data(double time, std::vector<double> &mX,
                          std::vector<double> &mY, std::vector<double> &mZ,
                          std::vector<double> &mT, std::vector<double> &mFS,
                          std::vector<double> &mFST) {
  // Read data
  read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mT, mFS, mFST);
}

void ReadModes::print_file_constants() {
  std::cout << "f_out " << f_out << " T " << T << " T_stop " << T_stop
            << std::endl;
  std::cout << "n1 " << n1 << " n2 " << n2 << std::endl;
  std::cout << "xlen " << xlen << " ylen " << ylen << std::endl;
  std::cout << "depth " << depth << " g " << g << " L " << L << std::endl;
}

void ReadModes::output_data(std::vector<double> &v1, std::vector<double> &v2,
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