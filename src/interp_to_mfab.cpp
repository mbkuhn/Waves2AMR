#include "interp_to_mfab.h"
#include "AMReX_Gpu.H"
#include <limits>

// Using user-specified parameters, create a vector of z heights. These are
// intended to be the heights where the IFFT is performed to get the velocity
// field, which will then be interpolated to the mesh points.
int interp_to_mfab::create_height_vector(amrex::Vector<amrex::Real> &hvec,
                                         const int n, const amrex::Real dz0,
                                         const amrex::Real z_wlev,
                                         const amrex::Real z_btm, int n_above) {
  // Ensure that routine is not misused
  if (n == 0 || dz0 < 1e-10 || z_wlev <= z_btm || n_above >= n) {
    amrex::Abort("interp_to_mfab::create_height_vector() has faulty inputs, "
                 "preventing height vector from being formed. n = " +
                 std::to_string(n) + ", dz0 = " + std::to_string(dz0) +
                 ", z_wlev = " + std::to_string(z_wlev) +
                 ", z_btm = " + std::to_string(z_btm) +
                 ", n_above = " + std::to_string(n_above));
  }

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
    amrex::Vector<amrex::Geometry> geom) {

  // Bounds of local AMR mesh
  amrex::Real mesh_zlo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_zhi = -1. * std::numeric_limits<double>::infinity();

  // Number of levels
  const int nlevels = field_fabs.size();
  // Loop through levels and mfabs and get max/min bounds
  for (int nl = 0; nl < nlevels; ++nl) {
    auto problo = geom[nl].ProbLoArray();
    auto dx = geom[nl].CellSizeArray();
    get_mfab_mesh_bounds(*field_fabs[nl], problo, dx, mesh_zlo, mesh_zhi);
  }

  return local_height_vec_ops(indvec, hvec, mesh_zlo, mesh_zhi);
}

// Loop through the mfabs of a field to get which z heights are local to the
// current process
int interp_to_mfab::get_local_height_indices(
    amrex::Vector<int> &indvec, amrex::Vector<amrex::Real> hvec,
    amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {

  // Bounds of local AMR mesh
  amrex::Real mesh_zlo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_zhi = -1. * std::numeric_limits<double>::infinity();

  // Number of levels
  const int nlevels = field_fabs.size();
  // Loop through levels and mfabs and get max/min bounds
  for (int nl = 0; nl < nlevels; ++nl) {
    get_mfab_mesh_bounds(*field_fabs[nl], problo_vec[nl], dx_vec[nl], mesh_zlo,
                         mesh_zhi);
  }

  return local_height_vec_ops(indvec, hvec, mesh_zlo, mesh_zhi);
}

// Loop through the mfabs of a field to get which z heights are local to the
// current process
int interp_to_mfab::get_local_height_indices(
    amrex::Vector<int> &indvec, amrex::Vector<amrex::Real> hvec,
    amrex::MultiFab &mfab, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {

  // Bounds of local AMR mesh
  amrex::Real mesh_zlo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_zhi = -1. * std::numeric_limits<double>::infinity();

  // This function works on a single mfab
  get_mfab_mesh_bounds(mfab, problo, dx, mesh_zlo, mesh_zhi);

  return local_height_vec_ops(indvec, hvec, mesh_zlo, mesh_zhi);
}

// !! -- This assumes cell-centered locations are what matters -- !! //
void interp_to_mfab::get_mfab_mesh_bounds(
    amrex::MultiFab &mfab, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real &mesh_zlo,
    amrex::Real &mesh_zhi, int idim) {
  for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
    const auto &bx = mfi.growntilebox();
    const amrex::Real mfab_hi =
        problo[idim] + (bx.bigEnd(idim) + 0.5) * dx[idim];
    const amrex::Real mfab_lo =
        problo[idim] + (bx.smallEnd(idim) + 0.5) * dx[idim];
    mesh_zlo = std::min(mfab_lo, mesh_zlo);
    mesh_zhi = std::max(mfab_hi, mesh_zhi);
  }
}

int interp_to_mfab::local_height_vec_ops(amrex::Vector<int> &indvec,
                                         amrex::Vector<amrex::Real> hvec,
                                         amrex::Real &mesh_zlo,
                                         amrex::Real &mesh_zhi) {
  // Loop through height vector and get first and last indices
  int itop = -1;           // top index is lowest ind, highest height
  int ibtm = -1;           // btm index is highest ind, lowest height
  bool noovlp_btw = false; // bool for edge case
  int nheights = hvec.size();
  for (int nh = 0; nh < nheights; ++nh) {
    if (itop == -1 && hvec[nh] <= mesh_zhi && hvec[nh] >= mesh_zlo) {
      itop = nh;
      ibtm = itop;
    }
    if (ibtm != -1 && hvec[nh] >= mesh_zlo) {
      ibtm = nh;
    }
    // Where no points overlap with multifab, but multifab is between points
    if (nh != nheights - 1) {
      if (itop == -1 && hvec[nh] > mesh_zhi && hvec[nh + 1] < mesh_zlo) {
        itop = nh;
        ibtm = nh + 1;
        noovlp_btw = true;
      }
    }
  }

  // If there are no overlapping points
  if (itop + ibtm < 0) {
    return 0;
  }

  // Expand indices to surround mfab points if possible
  if (!noovlp_btw) {
    itop = amrex::max(0, itop - 1);
    ibtm = amrex::min(ibtm + 1, nheights - 1);
  }

  // Make vector of indices
  indvec.resize(ibtm - itop + 1);
  for (int i = 0; i < ibtm - itop + 1; ++i) {
    indvec[i] = itop + i;
  }

  // Means that index vector has valid points, can be used
  return 1;
}

int interp_to_mfab::check_lateral_overlap(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::Geometry> geom, bool is_hi) {

  // Bounds of local AMR mesh
  amrex::Real mesh_lo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_hi = -1. * std::numeric_limits<double>::infinity();

  // Assume no overlap
  int flag = 0;
  // Number of levels
  const int nlevels = field_fabs.size();
  // Loop through levels and mfabs and get max/min bounds
  for (int nl = 0; nl < nlevels; ++nl) {
    auto problo = geom[nl].ProbLoArray();
    auto probhi = geom[nl].ProbHiArray();
    auto dx = geom[nl].CellSizeArray();
    // Bounds of region to check
    amrex::Real region_lo = problo[idim];
    amrex::Real region_hi = probhi[idim];
    // Depending on flag, distance is from hi or from lo
    if (is_hi) {
      region_lo = region_hi - dist;
    } else {
      region_hi = region_lo + dist;
    }
    // Get actual local mesh bounds
    get_mfab_mesh_bounds(*field_fabs[nl], problo, dx, mesh_lo, mesh_hi, idim);
    // Compare mesh bounds to region bounds
    if (region_lo < mesh_hi && region_hi > mesh_lo) {
      // Overlap exists
      flag = 1;
    }
  }

  return flag;
}

int interp_to_mfab::check_lateral_overlap_lo(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::Geometry> geom) {
  return check_lateral_overlap(dist, idim, field_fabs, geom, false);
}

int interp_to_mfab::check_lateral_overlap_hi(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::Geometry> geom) {
  return check_lateral_overlap(dist, idim, field_fabs, geom, true);
}

int interp_to_mfab::check_lateral_overlap(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> probhi_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec,
    bool is_hi) {

  // Bounds of local AMR mesh
  amrex::Real mesh_lo = std::numeric_limits<double>::infinity();
  amrex::Real mesh_hi = -1. * std::numeric_limits<double>::infinity();

  // Assume no overlap
  int flag = 0;
  // Number of levels
  const int nlevels = field_fabs.size();
  // Loop through levels and mfabs and get max/min bounds
  for (int nl = 0; nl < nlevels; ++nl) {
    auto problo = problo_vec[nl];
    auto probhi = probhi_vec[nl];
    auto dx = dx_vec[nl];
    // Bounds of region to check
    amrex::Real region_lo = problo[idim];
    amrex::Real region_hi = probhi[idim];
    // Depending on flag, distance is from hi or from lo
    if (is_hi) {
      region_lo = region_hi - dist;
    } else {
      region_hi = region_lo + dist;
    }
    // Get actual local mesh bounds
    get_mfab_mesh_bounds(*field_fabs[nl], problo, dx, mesh_lo, mesh_hi, idim);
    // Compare mesh bounds to region bounds
    if (region_lo < mesh_hi && region_hi > mesh_lo) {
      // Overlap exists
      flag = 1;
    }
  }

  return flag;
}

int interp_to_mfab::check_lateral_overlap_lo(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> probhi_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {
  return check_lateral_overlap(dist, idim, field_fabs, problo_vec, probhi_vec,
                               dx_vec, false);
}

int interp_to_mfab::check_lateral_overlap_hi(
    amrex::Real dist, int idim, amrex::Vector<amrex::MultiFab *> field_fabs,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> probhi_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {
  return check_lateral_overlap(dist, idim, field_fabs, problo_vec, probhi_vec,
                               dx_vec, true);
}

// Loop through and populate the multifab with levelset data, calculated by
// interpolating eta at each point
void interp_to_mfab::interp_eta_to_levelset_field(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, const amrex::Real zsl,
    amrex::Gpu::DeviceVector<amrex::Real> etavec,
    amrex::Vector<amrex::MultiFab *> lsfield,
    amrex::Vector<amrex::Geometry> geom) {

  // Number of levels
  const int nlevels = lsfield.size();

  // Loop through cells and perform interpolation
  for (int nl = 0; nl < nlevels; ++nl) {
    auto &lslev = *(lsfield[nl]);
    const auto problo = geom[nl].ProbLoArray();
    const auto dx = geom[nl].CellSizeArray();

    interp_eta_to_levelset_multifab(spd_nx, spd_ny, spd_dx, spd_dy, zsl, etavec,
                                    lslev, problo, dx);
  }
}

// Loop through and populate the multifab with levelset data, calculated by
// interpolating eta at each point
void interp_to_mfab::interp_eta_to_levelset_field(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, const amrex::Real zsl,
    amrex::Gpu::DeviceVector<amrex::Real> etavec,
    amrex::Vector<amrex::MultiFab *> lsfield,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {

  // Number of levels
  const int nlevels = lsfield.size();

  // Loop through cells and perform interpolation
  for (int nl = 0; nl < nlevels; ++nl) {
    auto &lslev = *(lsfield[nl]);
    const auto problo = problo_vec[nl];
    const auto dx = dx_vec[nl];

    interp_eta_to_levelset_multifab(spd_nx, spd_ny, spd_dx, spd_dy, zsl, etavec,
                                    lslev, problo, dx);
  }
}

// Loop through and populate the multifab with interpolated velocity
void interp_to_mfab::interp_velocity_to_field(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, amrex::Vector<int> indvec,
    amrex::Vector<amrex::Real> hvec, amrex::Gpu::DeviceVector<amrex::Real> uvec,
    amrex::Gpu::DeviceVector<amrex::Real> vvec,
    amrex::Gpu::DeviceVector<amrex::Real> wvec,
    amrex::Vector<amrex::MultiFab *> vfield,
    amrex::Vector<amrex::Geometry> geom) {

  // Number of levels
  const int nlevels = vfield.size();
  // Number of heights relevant to this processor
  const int nhts = indvec.size();
  // Copy hvec and indvec to device
  amrex::Gpu::DeviceVector<int> indvec_dvc(indvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, indvec.begin(), indvec.end(),
                   indvec_dvc.begin());
  amrex::Gpu::DeviceVector<amrex::Real> hvec_dvc(hvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, hvec.begin(), hvec.end(),
                   hvec_dvc.begin());
  // Get pointers to device vectors
  const auto *indvec_ptr = indvec_dvc.data();
  const auto *hvec_ptr = hvec_dvc.data();
  const auto *uvec_ptr = uvec.data();
  const auto *vvec_ptr = vvec.data();
  const auto *wvec_ptr = wvec.data();

  // Loop through cells and perform interpolation
  for (int nl = 0; nl < nlevels; ++nl) {
    auto &vlev = *(vfield[nl]);
    const auto problo = geom[nl].ProbLoArray();
    const auto dx = geom[nl].CellSizeArray();

    // Interpolate velocity for each multifab
    interp_velocity_to_multifab(spd_nx, spd_ny, spd_dx, spd_dy, indvec, hvec,
                                uvec, vvec, wvec, vlev, problo, dx);
  }
}

// Loop through and populate the multifab with interpolated velocity
void interp_to_mfab::interp_velocity_to_field(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, amrex::Vector<int> indvec,
    amrex::Vector<amrex::Real> hvec, amrex::Gpu::DeviceVector<amrex::Real> uvec,
    amrex::Gpu::DeviceVector<amrex::Real> vvec,
    amrex::Gpu::DeviceVector<amrex::Real> wvec,
    amrex::Vector<amrex::MultiFab *> vfield,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo_vec,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx_vec) {

  // Number of levels
  const int nlevels = vfield.size();
  // Number of heights relevant to this processor
  const int nhts = indvec.size();
  // Copy hvec and indvec to device
  amrex::Gpu::DeviceVector<int> indvec_dvc(indvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, indvec.begin(), indvec.end(),
                   indvec_dvc.begin());
  amrex::Gpu::DeviceVector<amrex::Real> hvec_dvc(hvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, hvec.begin(), hvec.end(),
                   hvec_dvc.begin());
  // Get pointers to device vectors
  const auto *indvec_ptr = indvec_dvc.data();
  const auto *hvec_ptr = hvec_dvc.data();
  const auto *uvec_ptr = uvec.data();
  const auto *vvec_ptr = vvec.data();
  const auto *wvec_ptr = wvec.data();

  // Loop through cells and perform interpolation
  for (int nl = 0; nl < nlevels; ++nl) {
    auto &vlev = *(vfield[nl]);
    const auto problo = problo_vec[nl];
    const auto dx = dx_vec[nl];

    // Interpolate velocity for each multifab
    interp_velocity_to_multifab(spd_nx, spd_ny, spd_dx, spd_dy, indvec, hvec,
                                uvec, vvec, wvec, vlev, problo, dx);
  }
}

// Loop through and populate the vector of multifabs with levelset data
void interp_to_mfab::interp_eta_to_levelset_multifab(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, const amrex::Real zsl,
    amrex::Gpu::DeviceVector<amrex::Real> etavec, amrex::MultiFab &lsfab,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {

  // Get pointer to device vector
  const auto *etavec_ptr = etavec.data();

  for (amrex::MFIter mfi(lsfab); mfi.isValid(); ++mfi) {
    auto bx = mfi.growntilebox();
    amrex::Array4<amrex::Real> lsarr = lsfab.array(mfi);
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
      int i1 = i0 + 1, j1 = j0 + 1;
      amrex::Real x0 = spd_dx * i0, x1 = spd_dx * i1;
      amrex::Real y0 = spd_dy * j0, y1 = spd_dy * j1;
      // Should there be an offset?
      // Get surrounding indices (go forward, go backward)
      while (i0 < spd_nx - 2 && x0 - spd_dx < xc) {
        ++i0;
        x0 = spd_dx * i0;
      }
      while ((i0 > 0 && x0 > xc) || i0 >= spd_nx) {
        --i0;
        x0 = spd_dx * i0;
      }
      while (j0 < spd_ny - 2 && y0 - spd_dy < yc) {
        ++j0;
        y0 = spd_dy * j0;
      }
      while ((j0 > 0 && y0 > yc) || j0 >= spd_ny) {
        --j0;
        y0 = spd_dy * j0;
      }
      // Get points above
      i1 = i0 + 1;
      x1 = spd_dx * i1;
      j1 = j0 + 1;
      y1 = spd_dy * j1;
      // Periodicity for indices
      i1 = (i1 >= spd_nx) ? i1 - spd_nx : i1;
      j1 = (j1 >= spd_ny) ? j1 - spd_ny : j1;
      // Form indices for 1D vector of 2D data
      // Row-major is a consequence of performing 2D FFT in C++
      const int idx00 = i0 + j0 * spd_nx;
      const int idx10 = i1 + j0 * spd_nx;
      const int idx01 = i0 + j1 * spd_nx;
      const int idx11 = i1 + j1 * spd_nx;
      // Get surrounding data
      const amrex::Real e00 = etavec_ptr[idx00];
      const amrex::Real e10 = etavec_ptr[idx10];
      const amrex::Real e01 = etavec_ptr[idx01];
      const amrex::Real e11 = etavec_ptr[idx11];
      // Interpolate eta and calculate levelset
      lsarr(i, j, k) =
          linear_interp2D(e00, e10, e01, e11, xc, yc, x0, y0, x1, y1) + zsl -
          zc;
    });
  }
}

// Loop through and populate the multifab with interpolated velocity
void interp_to_mfab::interp_velocity_to_multifab(
    const int spd_nx, const int spd_ny, const amrex::Real spd_dx,
    const amrex::Real spd_dy, amrex::Vector<int> indvec,
    amrex::Vector<amrex::Real> hvec, amrex::Gpu::DeviceVector<amrex::Real> uvec,
    amrex::Gpu::DeviceVector<amrex::Real> vvec,
    amrex::Gpu::DeviceVector<amrex::Real> wvec, amrex::MultiFab &vfab,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {
  // Number of heights relevant to this processor
  const int nhts = indvec.size();
  // Copy hvec and indvec to device
  amrex::Gpu::DeviceVector<int> indvec_dvc(indvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, indvec.begin(), indvec.end(),
                   indvec_dvc.begin());
  amrex::Gpu::DeviceVector<amrex::Real> hvec_dvc(hvec.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, hvec.begin(), hvec.end(),
                   hvec_dvc.begin());
  // Get pointers to device vectors
  const auto *indvec_ptr = indvec_dvc.data();
  const auto *hvec_ptr = hvec_dvc.data();
  const auto *uvec_ptr = uvec.data();
  const auto *vvec_ptr = vvec.data();
  const auto *wvec_ptr = wvec.data();

  // Loop through cells and perform interpolation
  for (amrex::MFIter mfi(vfab); mfi.isValid(); ++mfi) {
    auto bx = mfi.growntilebox();
    amrex::Array4<amrex::Real> varr = vfab.array(mfi);
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
      // Indvec maps from ordinary indices [0 .. number of overlapping heights]
      // to indices in the height vector [lowest overlapping index .. highest
      // overlapping index]. It does not correspond to the indices of heights in
      // the spatial hos data; those correspond to the original, ordinary
      // indices. Also, indvec is always continuous, allowing conversion of
      // indices by an offset.
      int k_abv = indvec_ptr[0];
      int i1 = i0 + 1, j1 = j0 + 1, k_blw = k_abv + 1;
      amrex::Real x0 = spd_dx * i0, x1 = spd_dx * i1;
      amrex::Real y0 = spd_dy * j0, y1 = spd_dy * j1;
      amrex::Real z0 = hvec_ptr[k_blw], z1 = hvec_ptr[k_abv];
      // Get surrounding indices (go forward, go backward)
      while (i0 < spd_nx - 2 && x0 - spd_dx < xc) {
        ++i0;
        x0 = spd_dx * i0;
      }
      while ((i0 > 0 && x0 > xc) || i0 >= spd_nx) {
        --i0;
        x0 = spd_dx * i0;
      }
      while (j0 < spd_ny - 2 && y0 - spd_dy < yc) {
        ++j0;
        y0 = spd_dy * j0;
      }
      while ((j0 > 0 && y0 > yc) || j0 >= spd_ny) {
        --j0;
        y0 = spd_dy * j0;
      }
      // Heights are in descending order!!
      while (k_abv < indvec_ptr[0] + nhts - 2 && hvec_ptr[k_abv + 1] > zc) {
        ++k_abv;
        z1 = hvec_ptr[k_abv];
      }
      while (k_abv > indvec_ptr[0] && z1 < zc) {
        --k_abv;
        z1 = hvec_ptr[k_abv];
      }
      // Get points above
      i1 = i0 + 1;
      x1 = spd_dx * i1;
      j1 = j0 + 1;
      y1 = spd_dy * j1;
      k_blw = k_abv + 1;
      z0 = hvec_ptr[k_blw];
      // Periodicity for indices
      i1 = (i1 >= spd_nx) ? i1 - spd_nx : i1;
      j1 = (j1 >= spd_ny) ? j1 - spd_ny : j1;
      // Offset k index to correctly access spatial data
      const int ok_blw = k_blw - indvec_ptr[0];
      const int ok_abv = k_abv - indvec_ptr[0];
      // Form indices for 1D vector of 3D data
      // Row-major is a consequence of performing 2D FFT in C++
      const int idx000 = i0 + j0 * spd_nx + ok_blw * spd_nx * spd_ny;
      const int idx100 = i1 + j0 * spd_nx + ok_blw * spd_nx * spd_ny;
      const int idx010 = i0 + j1 * spd_nx + ok_blw * spd_nx * spd_ny;
      const int idx110 = i1 + j1 * spd_nx + ok_blw * spd_nx * spd_ny;
      const int idx001 = i0 + j0 * spd_nx + ok_abv * spd_nx * spd_ny;
      const int idx101 = i1 + j0 * spd_nx + ok_abv * spd_nx * spd_ny;
      const int idx011 = i0 + j1 * spd_nx + ok_abv * spd_nx * spd_ny;
      const int idx111 = i1 + j1 * spd_nx + ok_abv * spd_nx * spd_ny;
      // Get surrounding data
      const amrex::Real u000 = uvec_ptr[idx000];
      const amrex::Real u100 = uvec_ptr[idx100];
      const amrex::Real u010 = uvec_ptr[idx010];
      const amrex::Real u001 = uvec_ptr[idx001];
      const amrex::Real u110 = uvec_ptr[idx110];
      const amrex::Real u101 = uvec_ptr[idx101];
      const amrex::Real u011 = uvec_ptr[idx011];
      const amrex::Real u111 = uvec_ptr[idx111];
      const amrex::Real v000 = vvec_ptr[idx000];
      const amrex::Real v100 = vvec_ptr[idx100];
      const amrex::Real v010 = vvec_ptr[idx010];
      const amrex::Real v001 = vvec_ptr[idx001];
      const amrex::Real v110 = vvec_ptr[idx110];
      const amrex::Real v101 = vvec_ptr[idx101];
      const amrex::Real v011 = vvec_ptr[idx011];
      const amrex::Real v111 = vvec_ptr[idx111];
      const amrex::Real w000 = wvec_ptr[idx000];
      const amrex::Real w100 = wvec_ptr[idx100];
      const amrex::Real w010 = wvec_ptr[idx010];
      const amrex::Real w001 = wvec_ptr[idx001];
      const amrex::Real w110 = wvec_ptr[idx110];
      const amrex::Real w101 = wvec_ptr[idx101];
      const amrex::Real w011 = wvec_ptr[idx011];
      const amrex::Real w111 = wvec_ptr[idx111];
      // Interpolate and store
      varr(i, j, k, 0) =
          linear_interp(u000, u100, u010, u001, u110, u101, u011, u111, xc, yc,
                        zc, x0, y0, z0, x1, y1, z1);
      varr(i, j, k, 1) =
          linear_interp(v000, v100, v010, v001, v110, v101, v011, v111, xc, yc,
                        zc, x0, y0, z0, x1, y1, z1);
      varr(i, j, k, 2) =
          linear_interp(w000, w100, w010, w001, w110, w101, w011, w111, xc, yc,
                        zc, x0, y0, z0, x1, y1, z1);

      // If lower edge of cell is above highest points from hos spatial
      // data, populate velocity with zeros
      if (zc - 0.5 * dx[2] > z1) {
        varr(i, j, k, 0) = 0.0;
        varr(i, j, k, 1) = 0.0;
        varr(i, j, k, 2) = 0.0;
      }
    });
  }
}

AMREX_GPU_HOST_DEVICE amrex::Real interp_to_mfab::linear_interp(
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

AMREX_GPU_HOST_DEVICE amrex::Real
interp_to_mfab::linear_interp2D(const amrex::Real a00, const amrex::Real a10,
                                const amrex::Real a01, const amrex::Real a11,
                                const amrex::Real xc, const amrex::Real yc,
                                const amrex::Real x0, const amrex::Real y0,
                                const amrex::Real x1, const amrex::Real y1) {

  // Interpolation weights in each direction (linear basis)
  const amrex::Real wx_hi = (xc - x0) / (x1 - x0);
  const amrex::Real wy_hi = (yc - y0) / (y1 - y0);

  const amrex::Real wx_lo = 1.0 - wx_hi;
  const amrex::Real wy_lo = 1.0 - wy_hi;

  return (wx_lo * wy_lo * a00 + wx_hi * wy_lo * a10 + wx_lo * wy_hi * a01 +
          wx_hi * wy_hi * a11);
}