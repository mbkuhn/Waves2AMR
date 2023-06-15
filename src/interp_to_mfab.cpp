#include "interp_to_mfab.h"

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