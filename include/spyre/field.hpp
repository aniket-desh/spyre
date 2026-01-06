#pragma once

#include "spyre/types.hpp"
#include "spyre/grid.hpp"
#include <functional>
#include <memory>

namespace spyre {

// forward declaration
class spherical_transform;

// scalar field on the sphere
class field {
public:
    // construct from grid
    explicit field(std::shared_ptr<grid> g);

    // copy and move
    field(const field& other);
    field(field&& other) noexcept;
    field& operator=(const field& other);
    field& operator=(field&& other) noexcept;

    ~field();

    // grid access
    const grid& get_grid() const { return *grid_; }
    std::shared_ptr<grid> grid_ptr() const { return grid_; }

    // data access (physical space)
    grid_data_t& data() { return data_; }
    const grid_data_t& data() const { return data_; }

    // element access
    real_t& operator()(int i, int j) { return data_(i, j); }
    real_t operator()(int i, int j) const { return data_(i, j); }

    // initialization from function f(lat, lon)
    void from_function(std::function<real_t(real_t, real_t)> f);

    // initialization from spherical harmonic (for testing)
    void from_spherical_harmonic(int l, int m);

    // arithmetic
    field& operator+=(const field& other);
    field& operator-=(const field& other);
    field& operator*=(real_t scalar);
    field& operator*=(const field& other);  // pointwise

    // norms
    real_t max_norm() const;
    real_t l2_norm() const;  // requires quadrature weights

    // statistics
    real_t min() const;
    real_t max() const;
    real_t mean() const;

private:
    std::shared_ptr<grid> grid_;
    grid_data_t data_;
};

// arithmetic operators
field operator+(const field& a, const field& b);
field operator-(const field& a, const field& b);
field operator*(const field& a, real_t s);
field operator*(real_t s, const field& a);

// vector field on the sphere (theta and phi components)
class vector_field {
public:
    explicit vector_field(std::shared_ptr<grid> g);

    const grid& get_grid() const { return *grid_; }

    // component access
    field& theta() { return theta_; }
    field& phi() { return phi_; }
    const field& theta() const { return theta_; }
    const field& phi() const { return phi_; }

    // magnitude |v|
    field magnitude() const;

private:
    std::shared_ptr<grid> grid_;
    field theta_;
    field phi_;
};

} // namespace spyre
