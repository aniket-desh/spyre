// debug transform with sin(theta)*cos(phi)
#include "spyre/spyre.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace spyre;

    int lmax = 31;
    auto g = std::make_shared<gauss_legendre_grid>(lmax);

    std::cout << "grid: " << g->n_lat() << " x " << g->n_lon() << "\n";

    // create field = sin(theta)*cos(phi)
    field f(g);
    f.from_function([](double theta, double phi) {
        return std::sin(theta) * std::cos(phi);
    });

    std::cout << "field sin(theta)*cos(phi):\n";
    std::cout << "  min=" << f.min() << " max=" << f.max() << "\n";

    // transform
    spherical_transform sht(g);

    auto coeffs = sht.allocate_coeffs();
    sht.forward(f, coeffs);

    // sin(theta)*cos(phi) = -sqrt(2pi/3) * Re(Y_1^1)
    // so coefficient at l=1, m=1 should be nonzero
    std::cout << "first 20 nonzero coefficients:\n";
    for (int i = 0; i < std::min(20, (int)coeffs.size()); ++i) {
        if (std::abs(coeffs(i)) > 1e-10) {
            std::cout << "  c[" << i << "]=" << coeffs(i) << "\n";
        }
    }

    // inverse transform
    field f2(g);
    sht.inverse(coeffs, f2);

    std::cout << "after inverse transform:\n";
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
