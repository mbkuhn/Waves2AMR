#ifndef AMREX_INTERFACE_OPS_H
#define AMREX_INTERFACE_OPS_H
#include "AMReX_FArrayBox.H"
#include <vector>
using namespace amrex;
namespace data_amrex {

amrex::FArrayBox copy_to_fab(int n0, int n1, amrex::Real xlen, amrex::Real ylen,
                             std::vector<double> input_vec);

} // namespace data_amrex

#endif