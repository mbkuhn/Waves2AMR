#include "interp_to_mfab.h"
#include <limits>

// Using user-specified parameters, create a vector of z heights. These are
// intended to be the heights where the IFFT is performed to get the velocity
// field, which will then be interpolated to the mesh points.
int interp_to_mfab::create_height_vector(amrex::Vector<amrex::Real> &hvec,
                                         int n, const amrex::Real dz0,
                                         const amrex::Real z_wlev,
                                         const amrex::Real z_btm, int n_above) {
  int flag = 0; // 0 means nothing is wrong
  hvec.resize(n);
  const int n_below = n - n_above;
  // z_wlev is average water level: boundary between cells above and below

  // Check that there are not too many cells
  const amrex::Real l = z_wlev - z_btm;
  if (dz0 * n_below > l) {
    flag = 1;
    return flag;
  }

  // Cells above interface are const spacing (dz0)
  amrex::Real z = z_wlev + 0.5 * dz0;
  for (int k = n_above - 1; k >= 0; --k) {
    hvec[k] = z;
    z += dz0;
  }

  // Get spacing factor for cells below
  amrex::Real r = 1.05;
  amrex::Real err = 1.0;
  constexpr amrex::Real tol = 1e-2;
  constexpr int iter_max = 1000;
  {
    int iter = 0;
    while (iter < 1000 && err > 1e-2) {
      r = std::pow(1.0 - (1.0 - r) * (l / dz0), 1.0 / n_below);
      err = std::abs(dz0 * (1.0 - std::pow(r, n_below)) / (1.0 - r) - l) * l;
      ++iter;
    }

    // Check result for flaws, exit
    if (err > tol) {
      flag = 2;
    } else if (r < 1.0) {
      flag = 3;
    } else if (std::isnan(r) || std::isinf(r)) {
      flag = 4;
    } else if (l <= 0.0) {
      flag = 5;
    }
    if (flag > 0) {
      // Array is flawed. pass flag to abort program
      return flag;
    }
  }

  // Cells below interface are spaced with geometric series
  amrex::Real dz = dz0;
  z = -0.5 * dz;
  for (int k = n_above; k < n; ++k) {
    hvec[k] = z;
    // Add half of current cell size
    z -= 0.5 * dz;
    // Get next cell size, add half to get next cell center
    dz *= r;
    z -= 0.5 * dz;
  }

  return flag;
}

// Loop through the mfabs of a field to get which z heights are local to the
// current process
int interp_to_mfab::get_local_height_indices(
    amrex::Vector<int> &indvec, amrex::Vector<amrex::Real> hvec,
    amrex::Vector<const amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {

  // Size of hvec
  int nheights = hvec.size();
  // Number of levels
  int nlevels = field_fabs.size();
  // This library assumes height is in z (index of 2)
  constexpr int idim = 2;

  // Bounds of local AMR mesh
  amrex::Real mesh_zlo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_zhi = -1. * std::numeric_limits<double>::infinity();

  // Loop through levels and mfabs and get max/min bounds
  for (int nl = 0; nl < nlevels; ++nl) {
    for (amrex::MFIter mfi(*field_fabs[nl]); mfi.isValid(); ++mfi) {
      const auto &bx = mfi.growntilebox();
      const amrex::Real mfab_hi =
          (problo_vec[nl])[idim] + bx.bigEnd(idim) * (dx_vec[nl])[idim];
      const amrex::Real mfab_lo =
          (problo_vec[nl])[idim] + bx.smallEnd(idim) * (dx_vec[nl])[idim];
      mesh_zlo = std::min(mfab_lo, mesh_zlo);
      mesh_zhi = std::max(mfab_hi, mesh_zhi);
    }
  }

  // Loop through height vector and get first and last indices
  int itop = -1; // top index is lowest ind, highest height
  int ibtm = -1; // btm index is highest ind, lowest height
  for (int nh = 0; nh < nheights; ++nh) {
    if (itop == -1 && hvec[nh] <= mesh_zhi && hvec[nh] >= mesh_zlo) {
      itop = nh;
      ibtm = itop;
    }
    if (ibtm != -1 && hvec[nh] >= mesh_zlo) {
      ibtm = nh;
    }
  }

  // If there are no overlapping points
  if (itop + ibtm < 0) {
    return 1;
  }

  // Make vector of indices
  indvec.resize(ibtm - itop + 1);
  for (int i = 0; i < ibtm - itop + 1; ++i) {
    indvec[i] = itop + i;
  }

  return 0;
}