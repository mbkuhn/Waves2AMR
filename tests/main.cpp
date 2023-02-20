#include "../Waves2AMR.h"

int main() {
  // Name of modes file
  std::string fname = "../tests/modes_HOS_SWENSE.dat";
  // Initialize mode reader
  ReadModes rmodes(fname);
  int nx = rmodes.get_n1();
  int ny = rmodes.get_n2();
  // Print constants to screen
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

  // Timestep stored: t = 100.0
  rmodes.get_data(100.0, mX, mY, mZ, mT, mFS, mFST);

  // Set up fftw_complex ptr for eta and get plan
  fftw_plan plan;
  fftw_complex *eta_modes =
      modes_hosgrid::allocate_plan_copy(nx, ny, plan, mFS);

  // Set up ptrs for velocity as well
  fftw_complex *u_modes = modes_hosgrid::allocate_copy(nx, ny, mX);
  fftw_complex *v_modes = modes_hosgrid::allocate_copy(nx, ny, mY);
  fftw_complex *w_modes = modes_hosgrid::allocate_copy(nx, ny, mZ);

  /*
  // Set up output vectors
  std::vector<double> eta, u0, u1, v0, v1, w0, w1;
  eta.resize((nx * ny));
  u0.resize((nx * ny));
  u1.resize((nx * ny));
  v0.resize((nx * ny));
  v1.resize((nx * ny));
  w0.resize((nx * ny));
  w1.resize((nx * ny));

  // Perform fftw for eta
  modes_hosgrid::populate_hos_eta(nx, ny, plan, eta_modes, eta);
  // Perform fftw for velocity at one height
  double depth = rmodes.get_depth();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();
  double ht = -0.1 * depth;
  modes_hosgrid::populate_hos_vel(nx, ny, xlen, ylen, depth, ht, plan, u_modes,
                                  v_modes, w_modes, u0, v0, w0);

  // Perform fftw for velocity at another height
  ht = -0.5 * depth;
  modes_hosgrid::populate_hos_vel(nx, ny, xlen, ylen, depth, ht, plan, u_modes,
                                  v_modes, w_modes, u1, v1, w1);

  // Get max, min of each quantity and print


  // Delete ptrs and plan
  delete[] eta_modes, u_modes, v_modes, w_modes;
  fftw_destroy_plan(plan); */
}