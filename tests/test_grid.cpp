#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "spyre/grid.hpp"
#include <cmath>

using namespace spyre;
using Catch::Matchers::WithinRel;

TEST_CASE("gauss-legendre grid construction", "[grid]") {
    SECTION("basic properties") {
        gauss_legendre_grid g(31);

        REQUIRE(g.l_max() == 31);
        REQUIRE(g.n_lat() == 32);
        REQUIRE(g.n_lon() == 64);
    }

    SECTION("latitude bounds") {
        gauss_legendre_grid g(15);

        // colatitudes should be in (0, pi)
        for (int i = 0; i < g.n_lat(); ++i) {
            REQUIRE(g.lat(i) > 0.0);
            REQUIRE(g.lat(i) < M_PI);
        }
    }

    SECTION("longitude bounds") {
        gauss_legendre_grid g(15);

        // longitudes should be in [0, 2*pi)
        for (int j = 0; j < g.n_lon(); ++j) {
            REQUIRE(g.lon(j) >= 0.0);
            REQUIRE(g.lon(j) < 2.0 * M_PI);
        }
    }

    SECTION("quadrature weights sum to 4*pi") {
        gauss_legendre_grid g(31);

        double sum = 0.0;
        for (int i = 0; i < g.n_lat(); ++i) {
            sum += g.weights()(i) * g.n_lon();
        }

        REQUIRE_THAT(sum, WithinRel(4.0 * M_PI, 1e-10));
    }
}

TEST_CASE("equiangular grid construction", "[grid]") {
    SECTION("basic properties") {
        equiangular_grid g(64, 128);

        REQUIRE(g.n_lat() == 64);
        REQUIRE(g.n_lon() == 128);
    }

    SECTION("uniform spacing") {
        equiangular_grid g(32, 64);

        double dtheta = M_PI / g.n_lat();
        double dphi = 2.0 * M_PI / g.n_lon();

        for (int i = 1; i < g.n_lat(); ++i) {
            REQUIRE_THAT(g.lat(i) - g.lat(i - 1), WithinRel(dtheta, 1e-10));
        }

        for (int j = 1; j < g.n_lon(); ++j) {
            REQUIRE_THAT(g.lon(j) - g.lon(j - 1), WithinRel(dphi, 1e-10));
        }
    }
}

TEST_CASE("grid factory", "[grid]") {
    auto g1 = grid::create("gauss", 15);
    REQUIRE(g1->l_max() == 15);

    auto g2 = grid::create("gauss_legendre", 31);
    REQUIRE(g2->l_max() == 31);

    REQUIRE_THROWS(grid::create("invalid", 15));
}
