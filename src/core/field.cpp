#include "spyre/field.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace spyre {

field::field(std::shared_ptr<grid> g)
    : grid_(std::move(g))
    , data_(grid_->n_lat(), grid_->n_lon()) {
    data_.setZero();
}

field::field(const field& other)
    : grid_(other.grid_)
    , data_(other.data_) {
}

field::field(field&& other) noexcept
    : grid_(std::move(other.grid_))
    , data_(std::move(other.data_)) {
}

field& field::operator=(const field& other) {
    if (this != &other) {
        grid_ = other.grid_;
        data_ = other.data_;
    }
    return *this;
}

field& field::operator=(field&& other) noexcept {
    if (this != &other) {
        grid_ = std::move(other.grid_);
        data_ = std::move(other.data_);
    }
    return *this;
}

field::~field() = default;

void field::from_function(std::function<real_t(real_t, real_t)> f) {
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    #ifdef SPYRE_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < nlat; ++i) {
        real_t theta = grid_->lat(i);
        for (int j = 0; j < nlon; ++j) {
            real_t phi = grid_->lon(j);
            data_(i, j) = f(theta, phi);
        }
    }
}

void field::from_spherical_harmonic(int l, int m) {
    // Y_l^m = P_l^m(cos(theta)) * exp(i*m*phi)
    // for real fields, use real part for m >= 0, imaginary for m < 0
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    int abs_m = std::abs(m);
    if (abs_m > l) {
        throw std::invalid_argument("|m| must be <= l");
    }

    for (int i = 0; i < nlat; ++i) {
        real_t theta = grid_->lat(i);
        real_t cos_theta = std::cos(theta);

        // compute associated legendre polynomial P_l^|m|
        // using recurrence relation
        real_t plm = 1.0;

        // P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
        real_t sin_theta = std::sin(theta);
        real_t pmm = 1.0;
        for (int k = 1; k <= abs_m; ++k) {
            pmm *= -(2*k - 1) * sin_theta;
        }

        if (l == abs_m) {
            plm = pmm;
        } else {
            // P_{m+1}^m = x(2m+1) P_m^m
            real_t pmmp1 = cos_theta * (2*abs_m + 1) * pmm;
            if (l == abs_m + 1) {
                plm = pmmp1;
            } else {
                // recurrence for higher l
                real_t pll = 0.0;
                for (int ll = abs_m + 2; ll <= l; ++ll) {
                    pll = (cos_theta * (2*ll - 1) * pmmp1 - (ll + abs_m - 1) * pmm) / (ll - abs_m);
                    pmm = pmmp1;
                    pmmp1 = pll;
                }
                plm = pll;
            }
        }

        // normalization factor for orthonormal spherical harmonics
        real_t norm = 1.0;
        for (int k = l - abs_m + 1; k <= l + abs_m; ++k) {
            norm *= k;
        }
        norm = std::sqrt((2*l + 1) / (4.0 * M_PI * norm));
        if (abs_m > 0) norm *= std::sqrt(2.0);

        plm *= norm;

        for (int j = 0; j < nlon; ++j) {
            real_t phi = grid_->lon(j);
            if (m >= 0) {
                data_(i, j) = plm * std::cos(m * phi);
            } else {
                data_(i, j) = plm * std::sin(abs_m * phi);
            }
        }
    }
}

field& field::operator+=(const field& other) {
    data_ += other.data_;
    return *this;
}

field& field::operator-=(const field& other) {
    data_ -= other.data_;
    return *this;
}

field& field::operator*=(real_t scalar) {
    data_ *= scalar;
    return *this;
}

field& field::operator*=(const field& other) {
    data_.array() *= other.data_.array();
    return *this;
}

real_t field::max_norm() const {
    return data_.cwiseAbs().maxCoeff();
}

real_t field::l2_norm() const {
    const auto& w = grid_->weights();
    real_t sum = 0.0;
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        for (int j = 0; j < nlon; ++j) {
            sum += data_(i, j) * data_(i, j) * w(i);
        }
    }
    return std::sqrt(sum);
}

real_t field::min() const {
    return data_.minCoeff();
}

real_t field::max() const {
    return data_.maxCoeff();
}

real_t field::mean() const {
    const auto& w = grid_->weights();
    real_t sum = 0.0;
    real_t total_weight = 0.0;
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        for (int j = 0; j < nlon; ++j) {
            sum += data_(i, j) * w(i);
            total_weight += w(i);
        }
    }
    return sum / total_weight;
}

field operator+(const field& a, const field& b) {
    field result(a);
    result += b;
    return result;
}

field operator-(const field& a, const field& b) {
    field result(a);
    result -= b;
    return result;
}

field operator*(const field& a, real_t s) {
    field result(a);
    result *= s;
    return result;
}

field operator*(real_t s, const field& a) {
    return a * s;
}

// vector field implementation
vector_field::vector_field(std::shared_ptr<grid> g)
    : grid_(g)
    , theta_(g)
    , phi_(g) {
}

field vector_field::magnitude() const {
    field result(grid_);
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        for (int j = 0; j < nlon; ++j) {
            real_t vt = theta_(i, j);
            real_t vp = phi_(i, j);
            result(i, j) = std::sqrt(vt*vt + vp*vp);
        }
    }
    return result;
}

} // namespace spyre
