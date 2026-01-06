#include "spyre/transform.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

#ifdef SPYRE_USE_SHTNS
#include <shtns.h>
#endif

namespace spyre {

#ifdef SPYRE_USE_SHTNS

// shtns-based implementation

spherical_transform::spherical_transform(std::shared_ptr<grid> g)
    : grid_(std::move(g))
    , shtns_(nullptr)
    , l_max_(grid_->l_max())
    , m_max_(grid_->l_max())
    , num_coeffs_(0) {
    init_shtns();
}

spherical_transform::~spherical_transform() {
    cleanup();
}

spherical_transform::spherical_transform(spherical_transform&& other) noexcept
    : grid_(std::move(other.grid_))
    , shtns_(other.shtns_)
    , l_max_(other.l_max_)
    , m_max_(other.m_max_)
    , num_coeffs_(other.num_coeffs_)
    , work_real_(std::move(other.work_real_))
    , work_complex_(std::move(other.work_complex_)) {
    other.shtns_ = nullptr;
}

spherical_transform& spherical_transform::operator=(spherical_transform&& other) noexcept {
    if (this != &other) {
        cleanup();
        grid_ = std::move(other.grid_);
        shtns_ = other.shtns_;
        l_max_ = other.l_max_;
        m_max_ = other.m_max_;
        num_coeffs_ = other.num_coeffs_;
        work_real_ = std::move(other.work_real_);
        work_complex_ = std::move(other.work_complex_);
        other.shtns_ = nullptr;
    }
    return *this;
}

void spherical_transform::init_shtns() {
    // use shtns_create + shtns_set_grid for proper initialization
    shtns_cfg cfg = shtns_create(l_max_, m_max_, 1, sht_orthonormal);
    if (!cfg) {
        throw std::runtime_error("failed to create shtns config");
    }

    // set up gauss grid with our dimensions
    shtns_set_grid(cfg, sht_gauss, 0.0, grid_->n_lat(), grid_->n_lon());

    shtns_ = cfg;
    num_coeffs_ = static_cast<size_t>(cfg->nlm);
    work_real_.resize(static_cast<size_t>(grid_->n_lat() * grid_->n_lon()));
    work_complex_.resize(num_coeffs_);
}

void spherical_transform::cleanup() {
    if (shtns_) {
        shtns_destroy(shtns_);
        shtns_ = nullptr;
    }
}

void spherical_transform::forward(const field& f, sh_coeffs_t& coeffs) const {
    if (!shtns_) throw std::runtime_error("shtns not initialized");

    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    // shtns expects data as: work[j*nlat + i] where j is lon index, i is lat index
    // (phi varies slowest, theta varies fastest)
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            work_real_[static_cast<size_t>(j * nlat + i)] = f(i, j);
        }
    }

    if (coeffs.size() != static_cast<Eigen::Index>(num_coeffs_)) {
        coeffs.resize(static_cast<Eigen::Index>(num_coeffs_));
    }

    spat_to_SH(shtns_, work_real_.data(), reinterpret_cast<cplx*>(coeffs.data()));
}

void spherical_transform::inverse(const sh_coeffs_t& coeffs, field& f) const {
    if (!shtns_) throw std::runtime_error("shtns not initialized");

    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    // copy to workspace (shtns may modify input)
    std::memcpy(work_complex_.data(), coeffs.data(), num_coeffs_ * sizeof(cplx));
    SH_to_spat(shtns_, reinterpret_cast<cplx*>(work_complex_.data()), work_real_.data());

    // shtns outputs data as: work[j*nlat + i] where j is lon index, i is lat index
    // (phi varies slowest, theta varies fastest)
    for (int i = 0; i < nlat; ++i) {
        for (int j = 0; j < nlon; ++j) {
            f(i, j) = work_real_[static_cast<size_t>(j * nlat + i)];
        }
    }
}

#else

// fallback implementation (no shtns)

spherical_transform::spherical_transform(std::shared_ptr<grid> g)
    : grid_(std::move(g))
    , shtns_(nullptr)
    , l_max_(grid_->l_max())
    , m_max_(grid_->l_max())
    , num_coeffs_(sh_num_coeffs(l_max_)) {
    work_real_.resize(static_cast<size_t>(grid_->n_lat() * grid_->n_lon()));
    work_complex_.resize(num_coeffs_);
}

spherical_transform::~spherical_transform() = default;

spherical_transform::spherical_transform(spherical_transform&& other) noexcept
    : grid_(std::move(other.grid_))
    , shtns_(nullptr)
    , l_max_(other.l_max_)
    , m_max_(other.m_max_)
    , num_coeffs_(other.num_coeffs_)
    , work_real_(std::move(other.work_real_))
    , work_complex_(std::move(other.work_complex_)) {
}

spherical_transform& spherical_transform::operator=(spherical_transform&& other) noexcept {
    if (this != &other) {
        grid_ = std::move(other.grid_);
        l_max_ = other.l_max_;
        m_max_ = other.m_max_;
        num_coeffs_ = other.num_coeffs_;
        work_real_ = std::move(other.work_real_);
        work_complex_ = std::move(other.work_complex_);
    }
    return *this;
}

namespace {

// associated legendre polynomial P_l^m(x) with normalization for spherical harmonics
double assoc_legendre(int l, int m, double x) {
    if (m < 0 || m > l) return 0.0;

    double pmm = 1.0;
    if (m > 0) {
        double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if (l == m) return pmm;

    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;

    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

// normalization factor for spherical harmonics
double sh_norm(int l, int m) {
    double num = (2.0 * l + 1.0) / (4.0 * M_PI);
    for (int k = l - m + 1; k <= l + m; ++k) {
        num /= k;
    }
    return std::sqrt(num);
}

} // anonymous namespace

void spherical_transform::forward(const field& f, sh_coeffs_t& coeffs) const {
    // direct summation (slow but correct)
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();
    const auto& weights = grid_->weights();

    if (coeffs.size() != static_cast<Eigen::Index>(num_coeffs_)) {
        coeffs.resize(static_cast<Eigen::Index>(num_coeffs_));
    }
    coeffs.setZero();

    for (int l = 0; l <= l_max_; ++l) {
        for (int m = 0; m <= l; ++m) {
            complex_t sum(0.0, 0.0);
            double norm = sh_norm(l, m);

            for (int i = 0; i < nlat; ++i) {
                double theta = grid_->lat(i);
                double cos_theta = std::cos(theta);
                double plm = assoc_legendre(l, m, cos_theta) * norm;

                for (int j = 0; j < nlon; ++j) {
                    double phi = grid_->lon(j);
                    complex_t ylm_conj(plm * std::cos(m * phi), -plm * std::sin(m * phi));
                    sum += f(i, j) * ylm_conj * weights(i);
                }
            }

            coeffs(static_cast<Eigen::Index>(sh_index(l, m))) = sum;
        }
    }
}

void spherical_transform::inverse(const sh_coeffs_t& coeffs, field& f) const {
    const int nlat = grid_->n_lat();
    const int nlon = grid_->n_lon();

    for (int i = 0; i < nlat; ++i) {
        double theta = grid_->lat(i);
        double cos_theta = std::cos(theta);

        for (int j = 0; j < nlon; ++j) {
            double phi = grid_->lon(j);
            double sum = 0.0;

            for (int l = 0; l <= l_max_; ++l) {
                double norm = sh_norm(l, 0);
                double pl0 = assoc_legendre(l, 0, cos_theta) * norm;
                sum += std::real(coeffs(static_cast<Eigen::Index>(sh_index(l, 0)))) * pl0;

                for (int m = 1; m <= l; ++m) {
                    norm = sh_norm(l, m);
                    double plm = assoc_legendre(l, m, cos_theta) * norm;
                    complex_t c = coeffs(static_cast<Eigen::Index>(sh_index(l, m)));
                    sum += 2.0 * plm * (std::real(c) * std::cos(m * phi) - std::imag(c) * std::sin(m * phi));
                }
            }

            f(i, j) = sum;
        }
    }
}

#endif // SPYRE_USE_SHTNS

// common implementations

sh_coeffs_t spherical_transform::allocate_coeffs() const {
    return sh_coeffs_t::Zero(static_cast<Eigen::Index>(num_coeffs_));
}

size_t spherical_transform::coeff_index(int l, int m) const {
#ifdef SPYRE_USE_SHTNS
    return static_cast<size_t>(LiM(shtns_, l, m));
#else
    return sh_index(l, m);
#endif
}

void spherical_transform::laplacian(const sh_coeffs_t& in, sh_coeffs_t& out, real_t radius) const {
    out.resize(in.size());
    real_t r2_inv = 1.0 / (radius * radius);

#ifdef SPYRE_USE_SHTNS
    // use shtns coefficient indexing via LiM macro
    for (int l = 0; l <= l_max_; ++l) {
        real_t eigenval = -static_cast<real_t>(l * (l + 1)) * r2_inv;
        for (int m = 0; m <= std::min(l, m_max_); ++m) {
            size_t idx = static_cast<size_t>(LiM(shtns_, l, m));
            out(static_cast<Eigen::Index>(idx)) = eigenval * in(static_cast<Eigen::Index>(idx));
        }
    }
#else
    for (int l = 0; l <= l_max_; ++l) {
        real_t eigenval = -static_cast<real_t>(l * (l + 1)) * r2_inv;
        for (int m = 0; m <= std::min(l, m_max_); ++m) {
            size_t idx = sh_index(l, m);
            out(static_cast<Eigen::Index>(idx)) = eigenval * in(static_cast<Eigen::Index>(idx));
        }
    }
#endif
}

void spherical_transform::gradient(const sh_coeffs_t& in,
                                    sh_coeffs_t& out_theta,
                                    sh_coeffs_t& out_phi) const {
    out_theta = allocate_coeffs();
    out_phi = allocate_coeffs();
    // placeholder - full implementation needs recursion relations
}

void spherical_transform::divergence(const sh_coeffs_t& v_theta,
                                      const sh_coeffs_t& v_phi,
                                      sh_coeffs_t& out) const {
    out = allocate_coeffs();
    // placeholder
}

void spherical_transform::curl(const sh_coeffs_t& v_theta,
                                const sh_coeffs_t& v_phi,
                                sh_coeffs_t& out) const {
    out = allocate_coeffs();
    // placeholder
}

vector_t spherical_transform::power_spectrum(const sh_coeffs_t& coeffs) const {
    vector_t spectrum = vector_t::Zero(l_max_ + 1);

    for (int l = 0; l <= l_max_; ++l) {
        real_t power = 0.0;
        for (int m = 0; m <= std::min(l, m_max_); ++m) {
#ifdef SPYRE_USE_SHTNS
            size_t idx = static_cast<size_t>(LiM(shtns_, l, m));
#else
            size_t idx = sh_index(l, m);
#endif
            complex_t c = coeffs(static_cast<Eigen::Index>(idx));
            real_t norm_sq = std::norm(c);
            power += (m == 0) ? norm_sq : 2.0 * norm_sq;
        }
        spectrum(l) = power;
    }

    return spectrum;
}

// convenience functions

field laplacian(const field& f, const spherical_transform& sht, real_t radius) {
    sh_coeffs_t coeffs = sht.allocate_coeffs();
    sh_coeffs_t lap_coeffs = sht.allocate_coeffs();

    sht.forward(f, coeffs);
    sht.laplacian(coeffs, lap_coeffs, radius);

    field result(f.grid_ptr());
    sht.inverse(lap_coeffs, result);

    return result;
}

vector_field gradient(const field& f, const spherical_transform& sht) {
    vector_field result(f.grid_ptr());
    // placeholder
    return result;
}

field divergence(const vector_field& v, const spherical_transform& sht) {
    auto g = std::make_shared<gauss_legendre_grid>(v.get_grid().l_max());
    field result(g);
    // placeholder
    return result;
}

} // namespace spyre
