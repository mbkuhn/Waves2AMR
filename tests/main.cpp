#include "../Waves2AMR.h"

int main() {
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
  std::vector<std::complex<double>> mT(vsize, initval);
  std::vector<std::complex<double>> mFS(vsize, initval);
  std::vector<std::complex<double>> mFST(vsize, initval);

  // Timestep stored: t = dt
  double dt_out = rmodes.get_dtout();
  rmodes.get_data(dt_out, mX, mY, mZ, mT, mFS, mFST);

  // Set up fftw_complex ptr for eta and get plan
  fftw_plan plan;
  fftw_complex *eta_modes =
      modes_hosgrid::allocate_plan_copy(n0, n1, plan, mFS);

  // Set up ptrs for velocity as well
  fftw_complex *u_modes = modes_hosgrid::allocate_copy(n0, n1, mX);
  fftw_complex *v_modes = modes_hosgrid::allocate_copy(n0, n1, mY);
  fftw_complex *w_modes = modes_hosgrid::allocate_copy(n0, n1, mZ);

  // Set up output vectors
  std::vector<double> eta, u0, u1, v0, v1, w0, w1;
  eta.resize((n0 * n1));
  u0.resize((n0 * n1));
  u1.resize((n0 * n1));
  v0.resize((n0 * n1));
  v1.resize((n0 * n1));
  w0.resize((n0 * n1));
  w1.resize((n0 * n1));

  // Perform fftw for eta
  modes_hosgrid::populate_hos_eta(n0, n1, plan, eta_modes, eta);
  // Perform fftw for velocity at one height
  double depth = rmodes.get_depth();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();
  double ht0 = 0.0;
  modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht0, plan, u_modes,
                                  v_modes, w_modes, u0, v0, w0);

  // Perform fftw for velocity at another height
  double ht1 = -0.5 * depth;
  modes_hosgrid::populate_hos_vel(n0, n1, xlen, ylen, depth, ht1, plan, u_modes,
                                  v_modes, w_modes, u1, v1, w1);

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
      max_eta = std::max(max_eta, eta[idx]);
      min_eta = std::min(min_eta, eta[idx]);
      max_u0 = std::max(max_u0, u0[idx]);
      min_u0 = std::min(min_u0, u0[idx]);
      max_v0 = std::max(max_v0, v0[idx]);
      min_v0 = std::min(min_v0, v0[idx]);
      max_w0 = std::max(max_w0, w0[idx]);
      min_w0 = std::min(min_w0, w0[idx]);
      max_u1 = std::max(max_u1, u1[idx]);
      min_u1 = std::min(min_u1, u1[idx]);
      max_v1 = std::max(max_v1, v1[idx]);
      min_v1 = std::min(min_v1, v1[idx]);
      max_w1 = std::max(max_w1, w1[idx]);
      min_w1 = std::min(min_w1, w1[idx]);
    }
  }

  std::cout << std::endl << "Max and min quantities\n";
  std::cout << "  eta: " << max_eta << " " << min_eta << std::endl;
  std::cout << "at ht = " << ht0 << std::endl;
  std::cout << "  u  : " << max_u0 << " " << min_u0 << std::endl;
  std::cout << "  v  : " << max_v0 << " " << min_v0 << std::endl;
  std::cout << "  w  : " << max_w0 << " " << min_w0 << std::endl;
  std::cout << "at ht = " << ht1 << std::endl;
  std::cout << "  u  : " << max_u1 << " " << min_u1 << std::endl;
  std::cout << "  v  : " << max_v1 << " " << min_v1 << std::endl;
  std::cout << "  w  : " << max_w1 << " " << min_w1 << std::endl;

  // Delete ptrs and plan
  delete[] eta_modes;
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  fftw_destroy_plan(plan);
}