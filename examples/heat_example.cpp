// heat equation example in c++

#include "spyre/spyre.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace spyre;

    // create grid with l_max = 63
    auto g = std::make_shared<gauss_legendre_grid>(63);

    std::cout << "grid: " << g->n_lat() << " x " << g->n_lon()
              << " (l_max = " << g->l_max() << ")\n";

    // create initial condition: gaussian bump
    field u0(g);
    u0.from_function([](double theta, double phi) {
        // centered at (pi/2, 0)
        double d_theta = theta - M_PI / 2;
        double d_phi = phi;
        if (d_phi > M_PI) d_phi -= 2 * M_PI;

        double sigma = 0.3;
        return std::exp(-(d_theta * d_theta + d_phi * d_phi) / (2 * sigma * sigma));
    });

    std::cout << "initial: min=" << u0.min() << " max=" << u0.max() << "\n";

    // create heat solver
    double kappa = 0.01;
    heat_solver solver(g, kappa, time_integrator::imex_euler);
    solver.set_initial_condition(u0);

    // solve
    double t_final = 5.0;
    double dt = 0.01;
    auto history = solver.solve(t_final, dt, 50);

    std::cout << "solved " << history.size() << " snapshots\n";
    std::cout << "final: min=" << solver.solution().min()
              << " max=" << solver.solution().max() << "\n";

    return 0;
}
