#include "../src/read_methods.h"

int main() {
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  ReadModes rmodes(fname);

  rmodes.print_file_constants();
}