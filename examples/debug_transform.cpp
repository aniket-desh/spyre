// debug transform
#include "spyre/spyre.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace spyre;

    int lmax = 7;
    auto g = std::make_shared<gauss_legendre_grid>(lmax);

    std::cout << "grid: " << g->n_lat() << " x " << g->n_lon() << "\n";

    // create constant field = 1
    field f(g);
    f.from_function([](double, double) { return 1.0; });

    std::cout << "constant field before transform:\n";
    std::cout << "  f(0,0)=" << f(0,0) << " f(0,1)=" << f(0,1) << "\n";
    std::cout << "  min=" << f.min() << " max=" << f.max() << "\n";

    // transform
    spherical_transform sht(g);
    std::cout << "sht l_max=" << sht.l_max() << " num_coeffs=" << sht.num_coeffs() << "\n";

    auto coeffs = sht.allocate_coeffs();
    sht.forward(f, coeffs);

    std::cout << "coefficients (should be ~sqrt(4*pi) at (0,0) and 0 elsewhere):\n";
    std::cout << "  c[0]=" << coeffs(0) << " (Y_0^0, expect " << std::sqrt(4*M_PI) << ")\n";
    for (int i = 1; i < std::min(10, (int)coeffs.size()); ++i) {
        if (std::abs(coeffs(i)) > 1e-10) {
            std::cout << "  c[" << i << "]=" << coeffs(i) << "\n";
        }
    }

    // inverse transform
    field f2(g);
    sht.inverse(coeffs, f2);

    std::cout << "after inverse transform:\n";
    std::cout << "  f2(0,0)=" << f2(0,0) << " f2(0,1)=" << f2(0,1) << "\n";
    std::cout << "  min=" << f2.min() << " max=" << f2.max() << "\n";

    double max_err = 0;
    for (int i = 0; i < g->n_lat(); ++i) {
        for (int j = 0; j < g->n_lon(); ++j) {
            max_err = std::max(max_err, std::abs(f(i,j) - f2(i,j)));
        }
    }
    std::cout << "roundtrip error: " << max_err << "\n";

    return 0;
}
