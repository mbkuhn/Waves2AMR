#include "../src/read_modes.h"

int main() {
  std::string fname = "/Users/mkuhn/testruns_data/HOS/Results_gdeskos_eagle/"
                      "modes_HOS_SWENSE.dat";
  ReadModes rmodes(fname);

  rmodes.print_file_constants();
}