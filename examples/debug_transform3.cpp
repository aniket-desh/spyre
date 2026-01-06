// debug transform memory layout - direct shtns test
#include <shtns.h>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    int lmax = 7;
    int nlat = lmax + 1;
    int nlon = 2 * (lmax + 1);

    std::cout << "grid: nlat=" << nlat << " nlon=" << nlon << "\n";

    // create shtns config
    shtns_cfg cfg = shtns_create(lmax, lmax, 1, sht_orthonormal);
    shtns_set_grid(cfg, sht_gauss, 0.0, nlat, nlon);

    // create test data: sin(theta)*cos(phi)
    std::vector<double> spatial(nlat * nlon);
    for (int i = 0; i < nlat; ++i) {
        double theta = std::acos(cfg->ct[i]);
        for (int j = 0; j < nlon; ++j) {
            double phi = 2 * M_PI * j / nlon;
            spatial[i * nlon + j] = std::sin(theta) * std::cos(phi);
        }
    }

    std::cout << "spatial data at a few points:\n";
    std::cout << "  spatial[0]=" << spatial[0] << " (i=0,j=0, theta=" << std::acos(cfg->ct[0]) << ", phi=0)\n";
    std::cout << "  spatial[1]=" << spatial[1] << " (i=0,j=1)\n";
    std::cout << "  spatial[nlon]=" << spatial[nlon] << " (i=1,j=0, theta=" << std::acos(cfg->ct[1]) << ")\n";

    // transform
    std::vector<std::complex<double>> coeffs(cfg->nlm);
    spat_to_SH(cfg, spatial.data(), reinterpret_cast<cplx*>(coeffs.data()));

    std::cout << "coefficients (nlm=" << cfg->nlm << "):\n";
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (std::abs(coeffs[i]) > 1e-10) {
            int l = cfg->li[i];
            int m = cfg->mi[i];
            std::cout << "  c[" << i << "] (l=" << l << ",m=" << m << ")=" << coeffs[i] << "\n";
        }
    }

    // inverse transform
    std::vector<double> spatial2(nlat * nlon);
    std::vector<std::complex<double>> coeffs_copy = coeffs;  // shtns may modify
    SH_to_spat(cfg, reinterpret_cast<cplx*>(coeffs_copy.data()), spatial2.data());

    double max_err = 0;
    for (int i = 0; i < nlat * nlon; ++i) {
        max_err = std::max(max_err, std::abs(spatial[i] - spatial2[i]));
    }
    std::cout << "roundtrip error: " << max_err << "\n";

    shtns_destroy(cfg);
    return 0;
}
