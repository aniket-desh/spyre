#include "spyre/grid.hpp"
#include <cmath>
#include <stdexcept>

namespace spyre {

equiangular_grid::equiangular_grid(int n_lat, int n_lon)
    : n_lat_(n_lat), n_lon_(n_lon) {
    if (n_lat < 2 || n_lon < 4) {
        throw std::invalid_argument("grid too small");
    }

    // l_max limited by nyquist
    l_max_ = std::min(n_lat - 1, n_lon / 2 - 1);

    // uniform colatitude grid (0, pi) excluding poles
    lats_.resize(n_lat);
    for (int i = 0; i < n_lat; ++i) {
        lats_(i) = M_PI * (i + 0.5) / n_lat;
    }

    // uniform longitude grid [0, 2*pi)
    lons_.resize(n_lon);
    for (int j = 0; j < n_lon; ++j) {
        lons_(j) = 2.0 * M_PI * j / n_lon;
    }

    // trapezoidal weights (simple approximation)
    weights_.resize(n_lat);
    double dtheta = M_PI / n_lat;
    double dphi = 2.0 * M_PI / n_lon;
    for (int i = 0; i < n_lat; ++i) {
        weights_(i) = std::sin(lats_(i)) * dtheta * dphi;
    }
}

} // namespace spyre
