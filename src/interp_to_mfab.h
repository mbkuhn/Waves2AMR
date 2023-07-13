#ifndef INTERP_TO_MFAB_H
#define INTERP_TO_MFAB_H
#include "AMReX_MultiFab.H"

namespace interp_to_mfab {

int create_height_vector(amrex::Vector<amrex::Real> &hvec, int n,
                         const amrex::Real dz0, const amrex::Real z_wlev,
                         const amrex::Real z_btm, int n_above = 1);

int get_local_height_indices(
    amrex::Vector<int> &indvec, amrex::Vector<amrex::Real> hvec,
    amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec);

void interp_velocity_to_multifab(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, amrex::Vector<int> indvec,
    amrex::Vector<amrex::Real> hvec,
    amrex::Gpu::DeviceVector<amrex::Real> uvec,
    amrex::Gpu::DeviceVector<amrex::Real> vvec,
    amrex::Gpu::DeviceVector<amrex::Real> wvec,
    amrex::Vector<amrex::MultiFab *> vfield,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec);

amrex::Real linear_interp(const amrex::Real a000, const amrex::Real a100,
                          const amrex::Real a010, const amrex::Real a001,
                          const amrex::Real a110, const amrex::Real a101,
                          const amrex::Real a011, const amrex::Real a111,
                          const amrex::Real xc, const amrex::Real yc,
                          const amrex::Real zc, const amrex::Real x0,
                          const amrex::Real y0, const amrex::Real z0,
                          const amrex::Real x1, const amrex::Real y1,
                          const amrex::Real z1);

} // namespace interp_to_mfab

#endif