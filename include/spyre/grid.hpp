#pragma once

#include "spyre/types.hpp"
#include <memory>

namespace spyre {

// base grid class
class grid {
public:
    virtual ~grid() = default;

    // grid dimensions
    virtual int n_lat() const = 0;
    virtual int n_lon() const = 0;
    virtual int l_max() const = 0;

    // coordinate accessors
    virtual const vector_t& latitudes() const = 0;   // colatitudes in [0, pi]
    virtual const vector_t& longitudes() const = 0;  // longitudes in [0, 2pi)

    // quadrature weights for integration
    virtual const vector_t& weights() const = 0;

    // grid point coordinates
    virtual real_t lat(int i) const = 0;
    virtual real_t lon(int j) const = 0;

    // factory method
    static std::unique_ptr<grid> create(const std::string& type, int l_max);
};

// gauss-legendre grid (optimal for spectral transforms)
class gauss_legendre_grid : public grid {
public:
    explicit gauss_legendre_grid(int l_max);

    int n_lat() const override { return n_lat_; }
    int n_lon() const override { return n_lon_; }
    int l_max() const override { return l_max_; }

    const vector_t& latitudes() const override { return lats_; }
    const vector_t& longitudes() const override { return lons_; }
    const vector_t& weights() const override { return weights_; }

    real_t lat(int i) const override { return lats_(i); }
    real_t lon(int j) const override { return lons_(j); }

private:
    int l_max_;
    int n_lat_;
    int n_lon_;
    vector_t lats_;    // colatitudes (theta)
    vector_t lons_;    // longitudes (phi)
    vector_t weights_; // gauss-legendre quadrature weights
};

// equiangular grid (uniform spacing, good for visualization)
class equiangular_grid : public grid {
public:
    equiangular_grid(int n_lat, int n_lon);

    int n_lat() const override { return n_lat_; }
    int n_lon() const override { return n_lon_; }
    int l_max() const override { return l_max_; }

    const vector_t& latitudes() const override { return lats_; }
    const vector_t& longitudes() const override { return lons_; }
    const vector_t& weights() const override { return weights_; }

    real_t lat(int i) const override { return lats_(i); }
    real_t lon(int j) const override { return lons_(j); }

private:
    int l_max_;
    int n_lat_;
    int n_lon_;
    vector_t lats_;
    vector_t lons_;
    vector_t weights_;
};

} // namespace spyre
