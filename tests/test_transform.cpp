#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "spyre/transform.hpp"
#include <cmath>

using namespace spyre;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("spherical transform construction", "[transform]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    spherical_transform sht(g);

    REQUIRE(sht.l_max() == 31);
    REQUIRE(sht.m_max() == 31);
    REQUIRE(sht.num_coeffs() > 0);
}

TEST_CASE("transform roundtrip", "[transform]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    spherical_transform sht(g);
    field f(g), f_recovered(g);

    // smooth test function
    f.from_function([](double theta, double phi) {
        return std::sin(theta) * std::cos(phi) + 0.5 * std::cos(2 * theta);
    });

    sh_coeffs_t coeffs = sht.allocate_coeffs();
    sht.forward(f, coeffs);
    sht.inverse(coeffs, f_recovered);

    // should recover original to high precision
    for (int i = 0; i < g->n_lat(); ++i) {
        for (int j = 0; j < g->n_lon(); ++j) {
            REQUIRE_THAT(f_recovered(i, j), WithinRel(f(i, j), 1e-10));
        }
    }
}

TEST_CASE("laplacian in spectral space", "[transform]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    spherical_transform sht(g);
    field f(g);

    // Y_l^m has eigenvalue -l(l+1) for laplacian
    // use Y_2^0 which is proportional to 3*cos^2(theta) - 1
    f.from_spherical_harmonic(2, 0);

    sh_coeffs_t coeffs = sht.allocate_coeffs();
    sh_coeffs_t lap_coeffs = sht.allocate_coeffs();

    sht.forward(f, coeffs);
    sht.laplacian(coeffs, lap_coeffs);

    // laplacian coeffs should be -l(l+1) * original = -6 * original
    // check dominant coefficient
    field lap_f(g);
    sht.inverse(lap_coeffs, lap_f);

    // lap(Y_2^0) = -6 * Y_2^0
    for (int i = 0; i < g->n_lat(); ++i) {
        for (int j = 0; j < g->n_lon(); ++j) {
            REQUIRE_THAT(lap_f(i, j), WithinRel(-6.0 * f(i, j), 1e-6));
        }
    }
}

TEST_CASE("power spectrum", "[transform]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    spherical_transform sht(g);
    field f(g);

    // single spherical harmonic should have power only at one l
    f.from_spherical_harmonic(5, 2);

    sh_coeffs_t coeffs = sht.allocate_coeffs();
    sht.forward(f, coeffs);
    vector_t spectrum = sht.power_spectrum(coeffs);

    // power should be concentrated at l=5
    double power_at_5 = spectrum(5);
    double total_power = spectrum.sum();

    REQUIRE(power_at_5 > 0.99 * total_power);
}
