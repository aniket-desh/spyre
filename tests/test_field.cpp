#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "spyre/field.hpp"
#include <cmath>

using namespace spyre;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("field construction", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(15);
    field f(g);

    REQUIRE(f.data().rows() == g->n_lat());
    REQUIRE(f.data().cols() == g->n_lon());
}

TEST_CASE("field from function", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(15);
    field f(g);

    // constant function
    f.from_function([](double, double) { return 1.0; });

    for (int i = 0; i < g->n_lat(); ++i) {
        for (int j = 0; j < g->n_lon(); ++j) {
            REQUIRE_THAT(f(i, j), WithinAbs(1.0, 1e-14));
        }
    }
}

TEST_CASE("field statistics", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(15);
    field f(g);

    f.from_function([](double theta, double phi) {
        return std::sin(theta) * std::cos(phi);
    });

    REQUIRE(f.min() < 0.0);
    REQUIRE(f.max() > 0.0);
    REQUIRE(f.max_norm() > 0.0);
}

TEST_CASE("field arithmetic", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(15);
    field f1(g), f2(g);

    f1.from_function([](double, double) { return 1.0; });
    f2.from_function([](double, double) { return 2.0; });

    SECTION("addition") {
        field f3 = f1 + f2;
        REQUIRE_THAT(f3(0, 0), WithinAbs(3.0, 1e-14));
    }

    SECTION("subtraction") {
        field f3 = f2 - f1;
        REQUIRE_THAT(f3(0, 0), WithinAbs(1.0, 1e-14));
    }

    SECTION("scalar multiplication") {
        field f3 = f1 * 5.0;
        REQUIRE_THAT(f3(0, 0), WithinAbs(5.0, 1e-14));
    }

    SECTION("in-place operations") {
        f1 += f2;
        REQUIRE_THAT(f1(0, 0), WithinAbs(3.0, 1e-14));

        f1 *= 2.0;
        REQUIRE_THAT(f1(0, 0), WithinAbs(6.0, 1e-14));
    }
}

TEST_CASE("field l2 norm", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    field f(g);

    // constant field: integral = 4*pi * c^2, norm = sqrt(4*pi) * c
    f.from_function([](double, double) { return 1.0; });

    double expected_norm = std::sqrt(4.0 * M_PI);
    REQUIRE_THAT(f.l2_norm(), WithinRel(expected_norm, 1e-6));
}

TEST_CASE("spherical harmonic field", "[field]") {
    auto g = std::make_shared<gauss_legendre_grid>(31);
    field f(g);

    // Y_0^0 = 1/(2*sqrt(pi)) -- constant
    f.from_spherical_harmonic(0, 0);
    double expected = 1.0 / (2.0 * std::sqrt(M_PI));

    // should be approximately constant
    double variation = f.max() - f.min();
    REQUIRE(variation < 1e-10);
}
