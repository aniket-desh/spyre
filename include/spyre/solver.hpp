#pragma once

#include "spyre/types.hpp"
#include "spyre/field.hpp"
#include "spyre/transform.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace spyre {

// time integration method
enum class time_integrator {
    euler,       // forward euler (first order)
    rk4,         // classical runge-kutta (fourth order)
    ssp_rk3,     // strong stability preserving rk3
    imex_euler,  // implicit-explicit euler
};

// solution history for animation
struct solution_history {
    std::vector<real_t> times;
    std::vector<field> fields;

    void clear() { times.clear(); fields.clear(); }
    size_t size() const { return times.size(); }
};

// base pde solver
class pde_solver {
public:
    pde_solver(std::shared_ptr<grid> g);
    virtual ~pde_solver() = default;

    // set initial condition
    void set_initial_condition(const field& u0);

    // single time step
    virtual void step(real_t dt) = 0;

    // solve from current time to t_final
    solution_history solve(real_t t_final, real_t dt, int save_every = 1);

    // accessors
    const field& solution() const { return u_; }
    field& solution() { return u_; }
    real_t time() const { return t_; }

    // reset time
    void reset() { t_ = 0.0; }

protected:
    std::shared_ptr<grid> grid_;
    std::unique_ptr<spherical_transform> sht_;
    field u_;     // current solution
    real_t t_;    // current time
};

// heat equation: du/dt = kappa * laplacian(u)
class heat_solver : public pde_solver {
public:
    heat_solver(std::shared_ptr<grid> g, real_t kappa = 1.0,
                time_integrator method = time_integrator::imex_euler);

    void step(real_t dt) override;

    // parameters
    void set_diffusivity(real_t kappa) { kappa_ = kappa; }
    real_t diffusivity() const { return kappa_; }

private:
    real_t kappa_;
    time_integrator method_;

    // spectral coefficients for implicit solve
    sh_coeffs_t coeffs_;
    sh_coeffs_t rhs_coeffs_;

    void step_explicit(real_t dt);
    void step_implicit(real_t dt);
};

// advection equation: du/dt + v . grad(u) = 0
class advection_solver : public pde_solver {
public:
    // advection with constant velocity field
    advection_solver(std::shared_ptr<grid> g,
                     real_t v_theta, real_t v_phi,
                     time_integrator method = time_integrator::rk4);

    // advection with variable velocity field
    advection_solver(std::shared_ptr<grid> g,
                     std::function<real_t(real_t, real_t)> v_theta,
                     std::function<real_t(real_t, real_t)> v_phi,
                     time_integrator method = time_integrator::rk4);

    void step(real_t dt) override;

private:
    field v_theta_;
    field v_phi_;
    time_integrator method_;

    // rk4 workspace
    field k1_, k2_, k3_, k4_;
    field temp_;

    // compute rhs: -v . grad(u)
    void compute_rhs(const field& u, field& rhs);
};

// reaction-diffusion: du/dt = D * laplacian(u) + f(u)
class reaction_diffusion_solver : public pde_solver {
public:
    using reaction_func = std::function<real_t(real_t)>;

    reaction_diffusion_solver(std::shared_ptr<grid> g,
                              real_t diffusivity,
                              reaction_func reaction,
                              time_integrator method = time_integrator::ssp_rk3);

    void step(real_t dt) override;

private:
    real_t D_;
    reaction_func f_;
    time_integrator method_;

    sh_coeffs_t coeffs_;
    field rhs_;
};

} // namespace spyre
