#include "interp_to_mfab.h"
#include "AMReX_Gpu.H"
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
    amrex::Vector<amrex::MultiFab *> field_fabs,
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

  // Expand indices to surround mfab points if possible
  itop = amrex::max(0, itop - 1);
  ibtm = amrex::min(ibtm + 1, nheights - 1);

  // Make vector of indices
  indvec.resize(ibtm - itop + 1);
  for (int i = 0; i < ibtm - itop + 1; ++i) {
    indvec[i] = itop + i;
  }

  return 0;
}

// Loop through and populate the multifab with interpolated velocity
void interp_to_mfab::interp_velocity_to_multifab(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, amrex::Vector<int> indvec,
    amrex::Vector<amrex::Real> hvec,
    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> uvec,
    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> vvec,
    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> wvec,
    amrex::Vector<amrex::MultiFab *> vfield,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {

  // Number of levels
  int nlevels = vfield.size();
  // Number of heights relevant to this processor
  int nhts = indvec.size();
  // Loop through cells and perform interpolation
  for (int nl = 0; nl < nlevels; ++nl) {
    auto &vlev = *(vfield[nl]);
    const auto problo = problo_vec[nl];
    const auto dx = dx_vec[nl];
    for (amrex::MFIter mfi(vlev); mfi.isValid(); ++mfi) {
      auto bx = mfi.growntilebox();
      amrex::Array4<amrex::Real> varr = vlev.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // Location of cell
        amrex::Real xc = problo[0] + (i + 0.5) * dx[0];
        amrex::Real yc = problo[1] + (j + 0.5) * dx[1];
        const amrex::Real zc = problo[2] + (k + 0.5) * dx[2];
        // HOS data assumed to be periodic in x and y
        const amrex::Real spd_Lx = spd_nx * spd_dx;
        const amrex::Real spd_Ly = spd_ny * spd_dy;
        xc = ((xc > spd_Lx) ? xc - spd_Lx : xc);
        xc = ((xc < 0.) ? xc + spd_Lx : xc);
        yc = ((yc > spd_Ly) ? yc - spd_Ly : yc);
        yc = ((yc < 0.) ? yc + spd_Ly : yc);
        // Initial positions and indices of HOS spatial data vectors
        int i0 = xc / spd_dx;
        int j0 = yc / spd_dy;
        int k_abv = indvec[0];
        int i1 = i0 + 1, j1 = j0 + 1, k_blw = k_abv + 1;
        amrex::Real x0 = spd_dx * i0, x1 = spd_dx * i1;
        amrex::Real y0 = spd_dy * j0, y1 = spd_dy * j1;
        amrex::Real z0 = hvec[k_blw], z1 = hvec[k_abv];
        // Should there be an offset?
        // Get surrounding indices (go forward, go backward)
        while (i0 < spd_nx - 2 && x0 - spd_dx < xc) {
          ++i0;
          x0 = spd_dx * i0;
        }
        while (i0 > 0 && x0 > xc) {
          --i0;
          x0 = spd_dx * i0;
        }
        while (j0 < spd_ny - 2 && y0 - spd_dy < yc) {
          ++j0;
          y0 = spd_dy * j0;
        }
        while (j0 > 0 && y0 > yc) {
          --j0;
          y0 = spd_dy * j0;
        }
        // Heights are in descending order!!
        while (k_abv < indvec[0] + nhts - 2 && hvec[k_abv + 1] > zc) {
          ++k_abv;
          z1 = hvec[k_abv];
        }
        while (k_abv > indvec[0] && z1 < zc) {
          --k_abv;
          z1 = hvec[k_abv];
        }
        // Get points above
        i1 = i0 + 1;
        x1 = spd_dx * i1;
        j1 = j0 + 1;
        y1 = spd_dy * j1;
        k_blw = k_abv + 1;
        z0 = hvec[k_blw];
        // Periodicity for indices
        i1 = (i1 >= spd_nx) ? i1 - spd_nx : i1;
        j1 = (j1 >= spd_ny) ? j1 - spd_ny : j1;
        // Form indices for 1D vector of 2D data
        const int idx00 = i0 * spd_ny + j0;
        const int idx10 = i1 * spd_ny + j0;
        const int idx01 = i0 * spd_ny + j1;
        const int idx11 = i1 * spd_ny + j1;
        // Get surrounding data (ijk) = [k][ij]
        const amrex::Real u000 = (uvec[k_blw])[idx00];
        const amrex::Real u100 = (uvec[k_blw])[idx10];
        const amrex::Real u010 = (uvec[k_blw])[idx01];
        const amrex::Real u001 = (uvec[k_abv])[idx00];
        const amrex::Real u110 = (uvec[k_blw])[idx11];
        const amrex::Real u101 = (uvec[k_abv])[idx10];
        const amrex::Real u011 = (uvec[k_abv])[idx01];
        const amrex::Real u111 = (uvec[k_abv])[idx11];
        const amrex::Real v000 = (vvec[k_blw])[idx00];
        const amrex::Real v100 = (vvec[k_blw])[idx10];
        const amrex::Real v010 = (vvec[k_blw])[idx01];
        const amrex::Real v001 = (vvec[k_abv])[idx00];
        const amrex::Real v110 = (vvec[k_blw])[idx11];
        const amrex::Real v101 = (vvec[k_abv])[idx10];
        const amrex::Real v011 = (vvec[k_abv])[idx01];
        const amrex::Real v111 = (vvec[k_abv])[idx11];
        const amrex::Real w000 = (wvec[k_blw])[idx00];
        const amrex::Real w100 = (wvec[k_blw])[idx10];
        const amrex::Real w010 = (wvec[k_blw])[idx01];
        const amrex::Real w001 = (wvec[k_abv])[idx00];
        const amrex::Real w110 = (wvec[k_blw])[idx11];
        const amrex::Real w101 = (wvec[k_abv])[idx10];
        const amrex::Real w011 = (wvec[k_abv])[idx01];
        const amrex::Real w111 = (wvec[k_abv])[idx11];
        // Interpolate and store
        varr(i, j, k, 0) =
            linear_interp(u000, u100, u010, u001, u110, u101, u011, u111, xc,
                          yc, zc, x0, y0, z0, x1, y1, z1);
        varr(i, j, k, 1) =
            linear_interp(v000, v100, v010, v001, v110, v101, v011, v111, xc,
                          yc, zc, x0, y0, z0, x1, y1, z1);
        varr(i, j, k, 2) =
            linear_interp(w000, w100, w010, w001, w110, w101, w011, w111, xc,
                          yc, zc, x0, y0, z0, x1, y1, z1);
      });
    }
  }
}

amrex::Real interp_to_mfab::linear_interp(
    const amrex::Real a000, const amrex::Real a100, const amrex::Real a010,
    const amrex::Real a001, const amrex::Real a110, const amrex::Real a101,
    const amrex::Real a011, const amrex::Real a111, const amrex::Real xc,
    const amrex::Real yc, const amrex::Real zc, const amrex::Real x0,
    const amrex::Real y0, const amrex::Real z0, const amrex::Real x1,
    const amrex::Real y1, const amrex::Real z1) {

  // Interpolation weights in each direction (linear basis)
  const amrex::Real wx_hi = (xc - x0) / (x1 - x0);
  const amrex::Real wy_hi = (yc - y0) / (y1 - y0);
  const amrex::Real wz_hi = (zc - z0) / (z1 - z0);

  const amrex::Real wx_lo = 1.0 - wx_hi;
  const amrex::Real wy_lo = 1.0 - wy_hi;
  const amrex::Real wz_lo = 1.0 - wz_hi;

  return (wx_lo * wy_lo * wz_lo * a000 + wx_lo * wy_lo * wz_hi * a001 +
          wx_lo * wy_hi * wz_lo * a010 + wx_lo * wy_hi * wz_hi * a011 +
          wx_hi * wy_lo * wz_lo * a100 + wx_hi * wy_lo * wz_hi * a101 +
          wx_hi * wy_hi * wz_lo * a110 + wx_hi * wy_hi * wz_hi * a111);
}