#include "amrex_interface_ops.h"
void data_amrex::copy_to_fab(int n0, int n1,
                             amrex::Gpu::DeviceVector<amrex::Real> input_vec,
                             amrex::FArrayBox &fab) {
  amrex::Box bx(amrex::IntVect{0, 0, 0}, amrex::IntVect{n0, n1, 0});
  fab.resize(bx, 1);
  amrex::Array4<amrex::Real> const &data_slab = fab.array();
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    data_slab(i, j, k, 0) = input_vec[i * n1 + j];
  });
}

void data_amrex::copy_to_fab(int n0, int n1,
                        amrex::Gpu::DeviceVector<amrex::Real> input_vec0,
                        amrex::Gpu::DeviceVector<amrex::Real> input_vec1,
                        amrex::Gpu::DeviceVector<amrex::Real> input_vec2,
                        amrex::FArrayBox &fab) {
  amrex::Box bx(amrex::IntVect{0, 0, 0}, amrex::IntVect{n0, n1, 0});
  fab.resize(bx, 3);
  amrex::Array4<amrex::Real> const &data_slab = fab.array();
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    data_slab(i, j, k, 0) = input_vec0[i * n1 + j];
    data_slab(i, j, k, 1) = input_vec1[i * n1 + j];
    data_slab(i, j, k, 2) = input_vec2[i * n1 + j];
  });
}