// quick test with small grid
#include "spyre/spyre.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace spyre;

    // small grid for quick test
    auto g = std::make_shared<gauss_legendre_grid>(15);
    std::cout << "grid: " << g->n_lat() << " x " << g->n_lon() << "\n";

    // field test
    field f(g);
    f.from_function([](double theta, double phi) {
        return std::sin(theta) * std::cos(phi);
    });
    std::cout << "field: min=" << f.min() << " max=" << f.max() << "\n";

    // transform test
    spherical_transform sht(g);
    auto coeffs = sht.allocate_coeffs();
    sht.forward(f, coeffs);
    std::cout << "coeffs size: " << coeffs.size() << "\n";

    field f2(g);
    sht.inverse(coeffs, f2);

    // check roundtrip error
    double max_err = 0;
    for (int i = 0; i < g->n_lat(); ++i) {
        for (int j = 0; j < g->n_lon(); ++j) {
            max_err = std::max(max_err, std::abs(f(i, j) - f2(i, j)));
        }
    }
    std::cout << "roundtrip error: " << max_err << "\n";

    // heat solver test
    heat_solver solver(g, 0.1, time_integrator::imex_euler);
    solver.set_initial_condition(f);
    solver.step(0.1);
    std::cout << "after step: min=" << solver.solution().min()
              << " max=" << solver.solution().max() << "\n";

    std::cout << "all tests passed!\n";
    return 0;
}
