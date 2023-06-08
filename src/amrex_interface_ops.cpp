#include "amrex_interface_ops.h"
using namespace amrex;
amrex::FArrayBox data_amrex::copy_to_fab(int n0, int n1, amrex::Real xlen,
                                         amrex::Real ylen,
                                         std::vector<double> input_vec) {
  Box box(IntVect{0, 0}, IntVect{n0, n1});
  amrex::FArrayBox dataslab(box, 1);
  /* More to be added */
  return dataslab;
}