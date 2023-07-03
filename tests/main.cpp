#include "../Waves2AMR.h"

int main(int argc, char *argv[]) {
  // Set up AMReX
  amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {});

  // Name of modes file
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Initialize mode reader
  ReadModes rmodes(fname);
  int n0 = rmodes.get_first_dimension();
  int n1 = rmodes.get_second_dimension();
  // Print constants to screen
  std::cout << "HOS simulation constants\n";
  rmodes.print_file_constants();

  // Initialize variables to store modes
  int vsize = rmodes.get_vector_size();
  double initval = 0.0;
  std::vector<std::complex<double>> mX(vsize, initval);
  std::vector<std::complex<double>> mY(vsize, initval);
  std::vector<std::complex<double>> mZ(vsize, initval);
  std::vector<std::complex<double>> mFS(vsize, initval);

  // Timestep stored: t = dt
  double dt_out = rmodes.get_dtout();
  rmodes.get_data(dt_out, mX, mY, mZ, mFS);

  // Set up fftw_complex ptr for eta and get plan
  fftw_plan plan;
  fftw_complex *eta_modes =
      modes_hosgrid::allocate_plan_copy(n0, n1, plan, mFS);

  // Allocate ptrs for velocity as well, copy is built-in later
  auto u_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto v_modes = modes_hosgrid::allocate_complex(n0, n1);
  auto w_modes = modes_hosgrid::allocate_complex(n0, n1);

  // Set up output vectors
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u0(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v0(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w0(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u1(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v1(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w1(n0 * n1, 0.0);

  // Perform fftw for eta
  modes_hosgrid::populate_hos_eta(n0, n1, plan, eta_modes, eta);
  // Get nondim dimensions
  double depth = rmodes.get_depth();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();

  // Perform fftw for velocity at one height
  double ht0 = -0.75325763322349282;
  modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht0, mX, mY, mZ,
                                  plan, u_modes, v_modes, w_modes, u0, v0, w0);

  // Perform fftw for velocity at another height
  double ht1 = -1.3822500981745969;
  modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht1, mX, mY, mZ,
                                  plan, u_modes, v_modes, w_modes, u1, v1, w1);

  // Transfer to host
  std::vector<amrex::Real> etal, u0l, v0l, w0l, u1l, v1l, w1l;
  etal.resize(n0 * n1);
  u0l.resize(n0 * n1);
  v0l.resize(n0 * n1);
  w0l.resize(n0 * n1);
  u1l.resize(n0 * n1);
  v1l.resize(n0 * n1);
  w1l.resize(n0 * n1);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(), &etal[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, u0.begin(), u0.end(), &u0l[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, v0.begin(), v0.end(), &v0l[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, w0.begin(), w0.end(), &w0l[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, u1.begin(), u1.end(), &u1l[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, v1.begin(), v1.end(), &v1l[0]);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, w1.begin(), w1.end(), &w1l[0]);

  // Get max, min of each quantity and print
  double max_eta = -100.0;
  double min_eta = 100.0;
  double max_u0 = -100.0;
  double min_u0 = 100.0;
  double max_v0 = -100.0;
  double min_v0 = 100.0;
  double max_w0 = -100.0;
  double min_w0 = 100.0;
  double max_u1 = -100.0;
  double min_u1 = 100.0;
  double max_v1 = -100.0;
  double min_v1 = 100.0;
  double max_w1 = -100.0;
  double min_w1 = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etal[idx]);
      min_eta = std::min(min_eta, etal[idx]);
      max_u0 = std::max(max_u0, u0l[idx]);
      min_u0 = std::min(min_u0, u0l[idx]);
      max_v0 = std::max(max_v0, v0l[idx]);
      min_v0 = std::min(min_v0, v0l[idx]);
      max_w0 = std::max(max_w0, w0l[idx]);
      min_w0 = std::min(min_w0, w0l[idx]);
      max_u1 = std::max(max_u1, u1l[idx]);
      min_u1 = std::min(min_u1, u1l[idx]);
      max_v1 = std::max(max_v1, v1l[idx]);
      min_v1 = std::min(min_v1, v1l[idx]);
      max_w1 = std::max(max_w1, w1l[idx]);
      min_w1 = std::min(min_w1, w1l[idx]);
    }
  }

  std::cout << std::endl << "Max and min nondim quantities\n";
  std::cout << "  eta: " << max_eta << " " << min_eta << std::endl;
  std::cout << "at ht = " << ht0 << std::endl;
  std::cout << "  u  : " << max_u0 << " " << min_u0 << std::endl;
  std::cout << "  v  : " << max_v0 << " " << min_v0 << std::endl;
  std::cout << "  w  : " << max_w0 << " " << min_w0 << std::endl;
  std::cout << "at ht = " << ht1 << std::endl;
  std::cout << "  u  : " << max_u1 << " " << min_u1 << std::endl;
  std::cout << "  v  : " << max_v1 << " " << min_v1 << std::endl;
  std::cout << "  w  : " << max_w1 << " " << min_w1 << std::endl;

  // --- Workflow for AMR-Wind --- //
  // Create heights where velocity will be sampled
  auto nheights = 40;
  const amrex::Real dz0 = 0.05;
  amrex::Vector<amrex::Real> hvec;
  int flag = interp_to_mfab::create_height_vector(hvec, nheights, dz0, 0.0,
                                                  -depth * rmodes.get_L());
  // Fail if flag indicates it should
  if (flag > 0) {
    amrex::Abort("create_height_vector error, failure code " +
                 std::to_string(flag));
  }
  // Create vector of multifab to represent part of AMR-Wind mesh
  int nz = 8;
  amrex::BoxArray ba(amrex::Box(amrex::IntVect{0, 0, 12 * nz},
                                amrex::IntVect{nz - 1, nz - 1, 13 * nz - 1}));
  amrex::DistributionMapping dm{ba};
  const int ncomp = 3;
  const int nghost = 3;
  // Just do one level in this test
  amrex::MultiFab mf(ba, dm, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> velocity_field{&mf};

  // Make vectors of GpuArrays for geometry information
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev{0.1, 0.1, 0.1};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> dx{dx_lev};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo_all{0., 0., -10.};
  amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> problo{
      problo_all};

  // Get indices of heights that overlap
  amrex::Vector<int> indvec;
  flag = interp_to_mfab::get_local_height_indices(indvec, hvec, velocity_field,
                                                  problo, dx);
  // Flag should indicate that there are overlapping points
  if (flag > 0) {
    amrex::Abort(
        "get_local_height_indices: no valid points between MF and hvec");
  }

  // Create vector of velocities to sample at each height
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> hos_u_vec;
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> hos_v_vec;
  amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> hos_w_vec;
  hos_u_vec.resize(indvec.size());
  hos_v_vec.resize(indvec.size());
  hos_w_vec.resize(indvec.size());

  // Loop through heights to check and print
  int n_hvec = 0;
  int n_mfab = nz + nghost - 1;
  int n_ivec = 0;
  std::cout << std::endl << "Heights, descending order: \n";
  for (int n = 0; n < nz + hvec.size(); ++n) {
    const amrex::Real h_mfab = (problo[0])[2] + (dx[0])[2] * (12 * nz + n_mfab);
    if (hvec[n_hvec] > h_mfab || n_mfab < -nghost) {
      std::cout << "hvec " << n_hvec << " " << hvec[n_hvec];
      if (n_ivec < indvec.size() && indvec[n_ivec] == n_hvec) {
        std::cout << "          ivec " << n_ivec;
        ++n_ivec;
      }
      std::cout << std::endl;
      ++n_hvec;
    } else {
      std::cout << "mfab " << n_mfab << "            " << h_mfab << std::endl;
      --n_mfab;
    }
  }

  // Sample velocities
  for (int iht = 0; iht < indvec.size(); ++iht) {
    // Resize vector within vector
    hos_u_vec[iht].resize(n0 * n1);
    hos_v_vec[iht].resize(n0 * n1);
    hos_w_vec[iht].resize(n0 * n1);
    // Get sample height
    amrex::Real ht = hvec[indvec[iht]];
    // Sample velocity
    modes_hosgrid::populate_hos_vel(
        n0, n1, xlen, ylen, depth, ht0, mX, mY, mZ, plan, u_modes, v_modes,
        w_modes, hos_u_vec[iht], hos_v_vec[iht], hos_w_vec[iht]);
    // Dimensionalize velocities (maybe should be included in populate?)
  }

  // Interpolate to multifab
  const amrex::Real spd_dx = xlen / n0;
  const amrex::Real spd_dy = ylen / n1;
  interp_to_mfab::interp_velocity_to_multifab(
      n0, n1, spd_dx, spd_dy, indvec, hvec, hos_u_vec, hos_v_vec, hos_w_vec,
      velocity_field, problo, dx);

  // Delete ptrs and plan
  delete[] eta_modes;
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  fftw_destroy_plan(plan);

  // Finalize AMReX
  amrex::Finalize();
}