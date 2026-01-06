#pragma once

#include "spyre/types.hpp"
#include "spyre/grid.hpp"
#include "spyre/field.hpp"
#include <memory>

// forward declaration of shtns
struct shtns_info;

namespace spyre {

// spherical harmonic transform wrapper around shtns
class spherical_transform {
public:
    explicit spherical_transform(std::shared_ptr<grid> g);
    ~spherical_transform();

    // no copy (shtns context is not copyable)
    spherical_transform(const spherical_transform&) = delete;
    spherical_transform& operator=(const spherical_transform&) = delete;

    // move ok
    spherical_transform(spherical_transform&& other) noexcept;
    spherical_transform& operator=(spherical_transform&& other) noexcept;

    // forward transform: physical -> spectral
    void forward(const field& f, sh_coeffs_t& coeffs) const;

    // inverse transform: spectral -> physical
    void inverse(const sh_coeffs_t& coeffs, field& f) const;

    // allocate coefficient array of correct size
    sh_coeffs_t allocate_coeffs() const;

    // spectral operations (operate directly on coefficients)

    // laplacian: nabla^2 f = -l(l+1) f_lm / R^2
    void laplacian(const sh_coeffs_t& in, sh_coeffs_t& out, real_t radius = 1.0) const;

    // gradient: returns theta and phi components
    void gradient(const sh_coeffs_t& in,
                  sh_coeffs_t& out_theta,
                  sh_coeffs_t& out_phi) const;

    // divergence of vector field
    void divergence(const sh_coeffs_t& v_theta,
                    const sh_coeffs_t& v_phi,
                    sh_coeffs_t& out) const;

    // curl of vector field (returns radial component)
    void curl(const sh_coeffs_t& v_theta,
              const sh_coeffs_t& v_phi,
              sh_coeffs_t& out) const;

    // access parameters
    int l_max() const { return l_max_; }
    int m_max() const { return m_max_; }
    size_t num_coeffs() const { return num_coeffs_; }

    // get coefficient index for (l, m)
    size_t coeff_index(int l, int m) const;

    // compute power spectrum: P(l) = sum_m |f_lm|^2
    vector_t power_spectrum(const sh_coeffs_t& coeffs) const;

private:
    std::shared_ptr<grid> grid_;
    shtns_info* shtns_;  // shtns context
    int l_max_;
    int m_max_;
    size_t num_coeffs_;

    // workspace for transforms
    mutable std::vector<double> work_real_;
    mutable std::vector<std::complex<double>> work_complex_;

    void init_shtns();
    void cleanup();
};

// convenience functions for field operations

// compute laplacian of field
field laplacian(const field& f, const spherical_transform& sht, real_t radius = 1.0);

// compute gradient of scalar field
vector_field gradient(const field& f, const spherical_transform& sht);

// compute divergence of vector field
field divergence(const vector_field& v, const spherical_transform& sht);

} // namespace spyre
