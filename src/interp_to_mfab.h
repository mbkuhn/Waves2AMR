#ifndef INTERP_TO_MFAB_H
#define INTERP_TO_MFAB_H
#include "AMReX_MultiFab.H"

namespace interp_to_mfab {

int create_height_vector(amrex::Vector<amrex::Real> &hvec, int n,
                         const amrex::Real dz0, const amrex::Real z_wlev,
                         const amrex::Real z_btm, int n_above = 1);

amrex::Vector<int> get_local_height_indices();

void interp_velocity_to_multifab();

} // namespace interp_to_mfab

#endif