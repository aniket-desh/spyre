#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "spyre/spyre.hpp"

namespace py = pybind11;
using namespace spyre;

// helper to convert field to numpy array (zero-copy view)
py::array_t<double> field_to_numpy(field& f) {
    auto& data = f.data();
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(data.rows()),
        static_cast<py::ssize_t>(data.cols())
    };
    std::vector<py::ssize_t> strides = {
        static_cast<py::ssize_t>(data.cols() * sizeof(double)),
        static_cast<py::ssize_t>(sizeof(double))
    };
    return py::array_t<double>(
        py::buffer_info(
            data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            shape,
            strides
        )
    );
}

// helper to set field from numpy array
void field_from_numpy(field& f, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto r = arr.unchecked<2>();
    if (r.shape(0) != f.data().rows() || r.shape(1) != f.data().cols()) {
        throw std::runtime_error("array shape mismatch");
    }
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        for (py::ssize_t j = 0; j < r.shape(1); ++j) {
            f(static_cast<int>(i), static_cast<int>(j)) = r(i, j);
        }
    }
}

PYBIND11_MODULE(_spyre_core, m) {
    m.doc() = "spyre c++ core bindings";

    // grid base class
    py::class_<grid, std::shared_ptr<grid>>(m, "grid")
        .def_property_readonly("n_lat", &grid::n_lat)
        .def_property_readonly("n_lon", &grid::n_lon)
        .def_property_readonly("l_max", &grid::l_max)
        .def("lat", &grid::lat)
        .def("lon", &grid::lon)
        .def_property_readonly("latitudes", [](const grid& g) {
            return py::array_t<double>(g.latitudes().size(), g.latitudes().data());
        })
        .def_property_readonly("longitudes", [](const grid& g) {
            return py::array_t<double>(g.longitudes().size(), g.longitudes().data());
        })
        .def_property_readonly("weights", [](const grid& g) {
            return py::array_t<double>(g.weights().size(), g.weights().data());
        });

    // gauss-legendre grid
    py::class_<gauss_legendre_grid, grid, std::shared_ptr<gauss_legendre_grid>>(m, "gauss_legendre_grid")
        .def(py::init<int>(), py::arg("l_max"));

    // equiangular grid
    py::class_<equiangular_grid, grid, std::shared_ptr<equiangular_grid>>(m, "equiangular_grid")
        .def(py::init<int, int>(), py::arg("n_lat"), py::arg("n_lon"));

    // field class
    py::class_<field>(m, "field")
        .def(py::init<std::shared_ptr<grid>>(), py::arg("grid"))
        .def_property_readonly("grid", &field::grid_ptr)
        .def_property("data",
            [](field& f) { return field_to_numpy(f); },
            [](field& f, py::array_t<double> arr) { field_from_numpy(f, arr); })
        .def("from_function", [](field& f, py::function func) {
            f.from_function([&func](real_t lat, real_t lon) {
                return func(lat, lon).cast<real_t>();
            });
        })
        .def("from_spherical_harmonic", &field::from_spherical_harmonic)
        .def("__getitem__", [](const field& f, std::tuple<int, int> idx) {
            return f(std::get<0>(idx), std::get<1>(idx));
        })
        .def("__setitem__", [](field& f, std::tuple<int, int> idx, real_t val) {
            f(std::get<0>(idx), std::get<1>(idx)) = val;
        })
        .def("__add__", [](const field& a, const field& b) { return a + b; })
        .def("__sub__", [](const field& a, const field& b) { return a - b; })
        .def("__mul__", [](const field& a, real_t s) { return a * s; })
        .def("__rmul__", [](const field& a, real_t s) { return s * a; })
        .def("__iadd__", &field::operator+=)
        .def("__isub__", &field::operator-=)
        .def("__imul__", py::overload_cast<real_t>(&field::operator*=))
        .def("max_norm", &field::max_norm)
        .def("l2_norm", &field::l2_norm)
        .def("min", &field::min)
        .def("max", &field::max)
        .def("mean", &field::mean)
        .def("copy", [](const field& f) { return field(f); });

    // vector field
    py::class_<vector_field>(m, "vector_field")
        .def(py::init<std::shared_ptr<grid>>(), py::arg("grid"))
        .def_property_readonly("theta", py::overload_cast<>(&vector_field::theta))
        .def_property_readonly("phi", py::overload_cast<>(&vector_field::phi))
        .def("magnitude", &vector_field::magnitude);

    // spherical transform
    py::class_<spherical_transform>(m, "spherical_transform")
        .def(py::init<std::shared_ptr<grid>>(), py::arg("grid"))
        .def_property_readonly("l_max", &spherical_transform::l_max)
        .def_property_readonly("m_max", &spherical_transform::m_max)
        .def_property_readonly("num_coeffs", &spherical_transform::num_coeffs)
        .def("forward", [](const spherical_transform& sht, const field& f) {
            sh_coeffs_t coeffs = sht.allocate_coeffs();
            sht.forward(f, coeffs);
            return py::array_t<std::complex<double>>(coeffs.size(), coeffs.data());
        })
        .def("inverse", [](const spherical_transform& sht, py::array_t<std::complex<double>> coeffs_arr, field& f) {
            auto r = coeffs_arr.unchecked<1>();
            sh_coeffs_t coeffs(r.size());
            for (py::ssize_t i = 0; i < r.size(); ++i) {
                coeffs(i) = r(i);
            }
            sht.inverse(coeffs, f);
        })
        .def("power_spectrum", [](const spherical_transform& sht, py::array_t<std::complex<double>> coeffs_arr) {
            auto r = coeffs_arr.unchecked<1>();
            sh_coeffs_t coeffs(r.size());
            for (py::ssize_t i = 0; i < r.size(); ++i) {
                coeffs(i) = r(i);
            }
            vector_t spectrum = sht.power_spectrum(coeffs);
            return py::array_t<double>(spectrum.size(), spectrum.data());
        });

    // time integrator enum
    py::enum_<time_integrator>(m, "time_integrator")
        .value("euler", time_integrator::euler)
        .value("rk4", time_integrator::rk4)
        .value("ssp_rk3", time_integrator::ssp_rk3)
        .value("imex_euler", time_integrator::imex_euler);

    // solution history
    py::class_<solution_history>(m, "solution_history")
        .def_readonly("times", &solution_history::times)
        .def_readonly("fields", &solution_history::fields)
        .def("__len__", &solution_history::size);

    // heat solver
    py::class_<heat_solver>(m, "heat_solver")
        .def(py::init<std::shared_ptr<grid>, real_t, time_integrator>(),
             py::arg("grid"),
             py::arg("diffusivity") = 1.0,
             py::arg("method") = time_integrator::imex_euler)
        .def("set_initial_condition", &heat_solver::set_initial_condition)
        .def("step", &heat_solver::step)
        .def("solve", &heat_solver::solve,
             py::arg("t_final"),
             py::arg("dt"),
             py::arg("save_every") = 1)
        .def_property_readonly("solution", py::overload_cast<>(&heat_solver::solution, py::const_))
        .def_property_readonly("time", &heat_solver::time)
        .def("reset", &heat_solver::reset)
        .def_property("diffusivity", &heat_solver::diffusivity, &heat_solver::set_diffusivity);

    // convenience functions
    m.def("laplacian", &laplacian,
          py::arg("f"),
          py::arg("sht"),
          py::arg("radius") = 1.0,
          "compute laplacian of scalar field");
}
