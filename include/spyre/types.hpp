#pragma once

#include <complex>
#include <vector>
#include <Eigen/Core>

namespace spyre {

// precision typedef - change to float for single precision
using real_t = double;
using complex_t = std::complex<real_t>;

// eigen types
using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;
using complex_vector_t = Eigen::VectorXcd;
using complex_matrix_t = Eigen::MatrixXcd;

// array types for grid data (row-major for cache efficiency in lat loops)
using grid_data_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// spherical harmonic coefficient storage
// stored as triangular array: coeffs[l*(l+1)/2 + m] for m >= 0
using sh_coeffs_t = complex_vector_t;

// index helper for spherical harmonic coefficients
inline size_t sh_index(int l, int m) {
    // only stores m >= 0, negative m obtained by conjugate symmetry
    return static_cast<size_t>(l * (l + 1) / 2 + m);
}

// total number of coefficients for max degree l_max
inline size_t sh_num_coeffs(int l_max) {
    return static_cast<size_t>((l_max + 1) * (l_max + 2) / 2);
}

} // namespace spyre
