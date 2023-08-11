#include "Waves2AMR.h"

int main(int argc, char *argv[]) {
  // Set up AMReX
  amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {});

  // Name of modes file
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Initialize mode reader and dimensionalize params
  ReadModes rmodes(fname, false);
  int n0 = rmodes.get_first_dimension();
  int n1 = rmodes.get_second_dimension();
  double depth = rmodes.get_depth();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();
  double dimL = rmodes.get_L();
  double dimT = rmodes.get_T();
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

  // --- Workflow for AMR-Wind --- //
  // Create heights where velocity will be sampled
  auto nheights = 40;
  const amrex::Real dz0 = 0.05;
  amrex::Vector<amrex::Real> hvec;
  int flag =
      interp_to_mfab::create_height_vector(hvec, nheights, dz0, 0.0, -depth);
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
  amrex::MultiFab mf_ls(ba, dm, 1, nghost);
  amrex::MultiFab mf_v(ba, dm, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> phi_field{&mf_ls};
  amrex::Vector<amrex::MultiFab *> velocity_field{&mf_v};

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

  // Perform fftw for eta
  amrex::Gpu::DeviceVector<amrex::Real> hos_eta_vec(n0 * n1, 0.0);
  modes_hosgrid::populate_hos_eta(n0, n1, dimL, plan, eta_modes, hos_eta_vec);

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

  // Create vector of velocities to sample
  amrex::Gpu::DeviceVector<amrex::Real> hos_u_vec;
  amrex::Gpu::DeviceVector<amrex::Real> hos_v_vec;
  amrex::Gpu::DeviceVector<amrex::Real> hos_w_vec;

  // Resize 1D velocity vectors
  int nht = indvec.size();
  hos_u_vec.resize(n0 * n1 * nht);
  hos_v_vec.resize(n0 * n1 * nht);
  hos_w_vec.resize(n0 * n1 * nht);
  // Sample velocities
  int indv = 0;
  for (int iht = 0; iht < nht; ++iht) {
    // Get sample height
    amrex::Real ht = hvec[indvec[iht]];
    // Sample velocity
    modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht, dimL, dimT,
                                    mX, mY, mZ, plan, u_modes, v_modes, w_modes,
                                    hos_u_vec, hos_v_vec, hos_w_vec, indv);
  }

  // Interpolate to multifab
  const amrex::Real spd_dx = xlen / n0;
  const amrex::Real spd_dy = ylen / n1;
  const amrex::Real zero_sea_level = 0.0;
  interp_to_mfab::interp_eta_to_levelset_field(n0, n1, spd_dx, spd_dy,
                                               zero_sea_level, hos_eta_vec,
                                               phi_field, problo, dx);
  interp_to_mfab::interp_velocity_to_field(n0, n1, spd_dx, spd_dy, indvec, hvec,
                                           hos_u_vec, hos_v_vec, hos_w_vec,
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