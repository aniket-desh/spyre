#include "spyre/solver.hpp"
#include <cmath>
#include <stdexcept>

namespace spyre {

// base solver implementation
pde_solver::pde_solver(std::shared_ptr<grid> g)
    : grid_(std::move(g))
    , sht_(std::make_unique<spherical_transform>(grid_))
    , u_(grid_)
    , t_(0.0) {
}

void pde_solver::set_initial_condition(const field& u0) {
    u_ = u0;
    t_ = 0.0;
}

solution_history pde_solver::solve(real_t t_final, real_t dt, int save_every) {
    solution_history history;

    int step_count = 0;
    while (t_ < t_final) {
        // adjust last step
        real_t current_dt = std::min(dt, t_final - t_);

        // save if needed
        if (step_count % save_every == 0) {
            history.times.push_back(t_);
            history.fields.push_back(u_);
        }

        // advance
        step(current_dt);
        t_ += current_dt;
        ++step_count;
    }

    // save final state
    history.times.push_back(t_);
    history.fields.push_back(u_);

    return history;
}

// heat solver implementation
heat_solver::heat_solver(std::shared_ptr<grid> g, real_t kappa, time_integrator method)
    : pde_solver(std::move(g))
    , kappa_(kappa)
    , method_(method)
    , coeffs_(sht_->allocate_coeffs())
    , rhs_coeffs_(sht_->allocate_coeffs()) {
}

void heat_solver::step(real_t dt) {
    switch (method_) {
        case time_integrator::euler:
        case time_integrator::rk4:
            step_explicit(dt);
            break;
        case time_integrator::imex_euler:
            step_implicit(dt);
            break;
        default:
            throw std::runtime_error("unsupported time integrator for heat equation");
    }
}

void heat_solver::step_explicit(real_t dt) {
    // forward euler: u^{n+1} = u^n + dt * kappa * laplacian(u^n)
    // or rk4 for higher order

    if (method_ == time_integrator::euler) {
        // transform to spectral
        sht_->forward(u_, coeffs_);

        // apply laplacian in spectral space
        sht_->laplacian(coeffs_, rhs_coeffs_);

        // update: coeffs += dt * kappa * laplacian
        coeffs_ += dt * kappa_ * rhs_coeffs_;

        // transform back
        sht_->inverse(coeffs_, u_);

    } else if (method_ == time_integrator::rk4) {
        // rk4 for heat equation
        field k1(grid_), k2(grid_), k3(grid_), k4(grid_);
        field temp(grid_);

        // k1 = kappa * laplacian(u)
        k1 = laplacian(u_, *sht_);
        k1 *= kappa_;

        // k2 = kappa * laplacian(u + 0.5*dt*k1)
        temp = u_ + 0.5 * dt * k1;
        k2 = laplacian(temp, *sht_);
        k2 *= kappa_;

        // k3 = kappa * laplacian(u + 0.5*dt*k2)
        temp = u_ + 0.5 * dt * k2;
        k3 = laplacian(temp, *sht_);
        k3 *= kappa_;

        // k4 = kappa * laplacian(u + dt*k3)
        temp = u_ + dt * k3;
        k4 = laplacian(temp, *sht_);
        k4 *= kappa_;

        // u = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        u_ += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
}

void heat_solver::step_implicit(real_t dt) {
    // backward euler (fully implicit):
    // u^{n+1} = u^n + dt * kappa * laplacian(u^{n+1})
    //
    // in spectral space:
    // u_lm^{n+1} = u_lm^n + dt * kappa * (-l(l+1)) * u_lm^{n+1}
    // u_lm^{n+1} * (1 + dt * kappa * l(l+1)) = u_lm^n
    // u_lm^{n+1} = u_lm^n / (1 + dt * kappa * l(l+1))

    sht_->forward(u_, coeffs_);

    int l_max = sht_->l_max();
    int m_max = sht_->m_max();

    // apply implicit operator
    for (int l = 0; l <= l_max; ++l) {
        real_t factor = 1.0 / (1.0 + dt * kappa_ * l * (l + 1));
        for (int m = 0; m <= std::min(l, m_max); ++m) {
            size_t idx = sht_->coeff_index(l, m);
            coeffs_(static_cast<Eigen::Index>(idx)) *= factor;
        }
    }

    sht_->inverse(coeffs_, u_);
}

// advection solver implementation
advection_solver::advection_solver(std::shared_ptr<grid> g, real_t v_theta, real_t v_phi,
                                   time_integrator method)
    : pde_solver(std::move(g))
    , v_theta_(grid_)
    , v_phi_(grid_)
    , method_(method)
    , k1_(grid_)
    , k2_(grid_)
    , k3_(grid_)
    , k4_(grid_)
    , temp_(grid_) {
    // constant velocity field
    v_theta_.from_function([v_theta](real_t, real_t) { return v_theta; });
    v_phi_.from_function([v_phi](real_t, real_t) { return v_phi; });
}

advection_solver::advection_solver(std::shared_ptr<grid> g,
                                   std::function<real_t(real_t, real_t)> v_theta_func,
                                   std::function<real_t(real_t, real_t)> v_phi_func,
                                   time_integrator method)
    : pde_solver(std::move(g))
    , v_theta_(grid_)
    , v_phi_(grid_)
    , method_(method)
    , k1_(grid_)
    , k2_(grid_)
    , k3_(grid_)
    , k4_(grid_)
    , temp_(grid_) {
    v_theta_.from_function(v_theta_func);
    v_phi_.from_function(v_phi_func);
}

void advection_solver::step(real_t dt) {
    // rk4 for advection
    if (method_ == time_integrator::rk4) {
        compute_rhs(u_, k1_);

        temp_ = u_ + 0.5 * dt * k1_;
        compute_rhs(temp_, k2_);

        temp_ = u_ + 0.5 * dt * k2_;
        compute_rhs(temp_, k3_);

        temp_ = u_ + dt * k3_;
        compute_rhs(temp_, k4_);

        u_ += (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_);

    } else if (method_ == time_integrator::euler) {
        compute_rhs(u_, k1_);
        u_ += dt * k1_;
    }
}

void advection_solver::compute_rhs(const field& u, field& rhs) {
    // rhs = -v . grad(u)
    // use spectral gradient

    vector_field grad_u = gradient(u, *sht_);

    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        real_t sin_theta = std::sin(grid_->lat(i));
        for (int j = 0; j < nlon; ++j) {
            // velocity dot gradient
            // on sphere: v . grad(u) = v_theta * du/dtheta + v_phi * du/dphi / sin(theta)
            real_t advection = v_theta_(i, j) * grad_u.theta()(i, j);
            if (sin_theta > 1e-10) {
                advection += v_phi_(i, j) * grad_u.phi()(i, j) / sin_theta;
            }
            rhs(i, j) = -advection;
        }
    }
}

// reaction-diffusion solver
reaction_diffusion_solver::reaction_diffusion_solver(std::shared_ptr<grid> g,
                                                     real_t diffusivity,
                                                     reaction_func reaction,
                                                     time_integrator method)
    : pde_solver(std::move(g))
    , D_(diffusivity)
    , f_(std::move(reaction))
    , method_(method)
    , coeffs_(sht_->allocate_coeffs())
    , rhs_(grid_) {
}

void reaction_diffusion_solver::step(real_t dt) {
    // strang splitting or imex
    // here: simple first-order splitting

    // step 1: diffusion (implicit)
    sht_->forward(u_, coeffs_);

    int l_max = sht_->l_max();
    int m_max = sht_->m_max();

    for (int l = 0; l <= l_max; ++l) {
        real_t factor = 1.0 / (1.0 + dt * D_ * l * (l + 1));
        for (int m = 0; m <= std::min(l, m_max); ++m) {
            size_t idx = sht_->coeff_index(l, m);
            coeffs_(static_cast<Eigen::Index>(idx)) *= factor;
        }
    }

    sht_->inverse(coeffs_, u_);

    // step 2: reaction (explicit euler)
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        for (int j = 0; j < nlon; ++j) {
            u_(i, j) += dt * f_(u_(i, j));
        }
    }
}

} // namespace spyre
