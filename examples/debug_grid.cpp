// debug grid points
#include "spyre/spyre.hpp"
#include <shtns.h>
#include <iostream>
#include <cmath>

int main() {
    using namespace spyre;

    int lmax = 7;
    auto g = std::make_shared<gauss_legendre_grid>(lmax);

    std::cout << "our grid: nlat=" << g->n_lat() << " nlon=" << g->n_lon() << "\n";
    std::cout << "our colatitudes (theta):\n";
    for (int i = 0; i < g->n_lat(); ++i) {
        std::cout << "  " << i << ": theta=" << g->lat(i) << " cos=" << std::cos(g->lat(i)) << "\n";
    }

    // create shtns config
    shtns_cfg cfg = shtns_create(lmax, lmax, 1, sht_orthonormal);
    shtns_set_grid(cfg, sht_gauss, 0.0, g->n_lat(), g->n_lon());

    std::cout << "\nshtns grid: nlat=" << cfg->nlat << " nlon=" << cfg->nphi << "\n";
    std::cout << "shtns cos(theta) values:\n";
    for (unsigned i = 0; i < cfg->nlat; ++i) {
        std::cout << "  " << i << ": cos=" << cfg->ct[i] << " theta=" << std::acos(cfg->ct[i]) << "\n";
    }

    shtns_destroy(cfg);
    return 0;
}
