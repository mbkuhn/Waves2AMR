#ifndef AMREX_INTERFACE_OPS_H
#define AMREX_INTERFACE_OPS_H
#include "AMReX_FArrayBox.H"
#include <vector>
namespace data_amrex {

void copy_to_fab(int n0, int n1,
                 amrex::Gpu::DeviceVector<amrex::Real> input_vec,
                 amrex::FArrayBox &fab);

void copy_to_fab(int n0, int n1,
                             amrex::Gpu::DeviceVector<amrex::Real> input_vec0,
                             amrex::Gpu::DeviceVector<amrex::Real> input_vec1,
                             amrex::Gpu::DeviceVector<amrex::Real> input_vec2,
                             amrex::FArrayBox &fab);

} // namespace data_amrex

#endif