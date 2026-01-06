#include "spyre/grid.hpp"
#include <cmath>
#include <stdexcept>

namespace spyre {

namespace {

// compute gauss-legendre nodes and weights
// uses newton iteration to find roots of legendre polynomial
void gauss_legendre_nodes(int n, vector_t& nodes, vector_t& weights) {
    nodes.resize(n);
    weights.resize(n);

    const double eps = 1e-14;
    const int max_iter = 100;

    int m = (n + 1) / 2;

    for (int i = 0; i < m; ++i) {
        // initial guess from asymptotic formula
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));

        double p1, p2, p3, pp;

        for (int iter = 0; iter < max_iter; ++iter) {
            // evaluate legendre polynomial and derivative via recurrence
            p1 = 1.0;
            p2 = 0.0;

            for (int j = 1; j <= n; ++j) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }

            // derivative
            pp = n * (z * p1 - p2) / (z * z - 1.0);

            double z_old = z;
            z = z_old - p1 / pp;

            if (std::abs(z - z_old) < eps) break;
        }

        // symmetric pair
        nodes(i) = -z;
        nodes(n - 1 - i) = z;

        weights(i) = 2.0 / ((1.0 - z * z) * pp * pp);
        weights(n - 1 - i) = weights(i);
    }
}

} // anonymous namespace

gauss_legendre_grid::gauss_legendre_grid(int l_max)
    : l_max_(l_max) {
    if (l_max < 1) {
        throw std::invalid_argument("l_max must be at least 1");
    }

    // shtns convention: n_lat = l_max + 1 for gauss grid
    n_lat_ = l_max + 1;
    // n_lon must be >= 2*l_max + 1 for dealiasing, use power of 2 for fft
    n_lon_ = 2 * (l_max + 1);

    // compute gauss-legendre nodes in [-1, 1]
    vector_t cos_theta;
    gauss_legendre_nodes(n_lat_, cos_theta, weights_);

    // convert to colatitude [0, pi]
    // shtns expects theta from 0 (north pole) to pi (south pole)
    // gauss_legendre_nodes returns cos(theta) from -1 to 1, so we need to reverse
    lats_.resize(n_lat_);
    vector_t weights_reversed(n_lat_);
    for (int i = 0; i < n_lat_; ++i) {
        // reverse the ordering: first element should be smallest theta (north pole)
        lats_(i) = std::acos(cos_theta(n_lat_ - 1 - i));
        weights_reversed(i) = weights_(n_lat_ - 1 - i);
    }
    weights_ = weights_reversed;

    // uniform longitude grid [0, 2*pi)
    lons_.resize(n_lon_);
    for (int j = 0; j < n_lon_; ++j) {
        lons_(j) = 2.0 * M_PI * j / n_lon_;
    }

    // scale weights for spherical integration (includes sin(theta) jacobian)
    // integral = sum_ij f(i,j) * weights(i) * 2*pi/n_lon
    for (int i = 0; i < n_lat_; ++i) {
        weights_(i) *= 2.0 * M_PI / n_lon_;
    }
}

std::unique_ptr<grid> grid::create(const std::string& type, int l_max) {
    if (type == "gauss" || type == "gauss_legendre") {
        return std::make_unique<gauss_legendre_grid>(l_max);
    }
    throw std::invalid_argument("unknown grid type: " + type);
}

} // namespace spyre
