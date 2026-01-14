# spyre $\odot$
a fast, efficient library for solving pdes on a sphere with python bindings.

## features

- **spherical harmonic transforms** - wrapped shtns for O(n² log n) transforms
- **spectral pde solvers** - heat equation, advection, reaction-diffusion
- **time integration** - euler, rk4, ssp-rk3, imex schemes
- **visualization** - 3d sphere plots, 2d map projections, animations
- **python bindings** - numpy-compatible via pybind11

## installation

### python package

```bash
# install dependencies (macos)
brew install cmake eigen fftw

# build shtns from source
git clone https://bitbucket.org/nschaeff/shtns.git
cd shtns && ./configure --enable-openmp && make && sudo make install

# install spyre python package
pip install python/

# or with visualization dependencies
pip install python/[viz]

# or with all optional dependencies
pip install python/[all]
```

### c++ library only

```bash
# build c++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## usage

### python

```python
import numpy as np
import spyre

# create sphere (gauss-legendre grid, l_max=63)
grid = spyre.sphere(l_max=63)

# define initial condition
u0 = spyre.field(grid)
u0.from_function(lambda theta, phi: np.exp(-((theta - np.pi/4)**2 + phi**2) / 0.1))

# solve heat equation
solver = spyre.heat_solver(grid, diffusivity=0.01)
solver.set_initial_condition(u0)
history = solver.solve(t_final=5.0, dt=0.01, save_every=10)

# animate on 3d sphere
spyre.animate(history, projection="3d", cmap="inferno", output="heat.mp4")
```

### c++

```cpp
#include "spyre/spyre.hpp"

using namespace spyre;

int main() {
    auto grid = std::make_shared<gauss_legendre_grid>(63);
    field u(grid);
    u.from_function([](double theta, double phi) {
        return std::exp(-((theta - M_PI/4)*(theta - M_PI/4) + phi*phi) / 0.1);
    });

    heat_solver solver(grid, 0.01);
    solver.set_initial_condition(u);
    auto history = solver.solve(5.0, 0.01, 10);

    return 0;
}
```

## api

### grids
- `gauss_legendre_grid(l_max)` - optimal for spectral methods
- `equiangular_grid(n_lat, n_lon)` - uniform spacing

### fields
- `field(grid)` - scalar field on sphere
- `field.from_function(f)` - initialize from f(theta, phi)
- `field.data` - numpy array access

### transforms
- `spherical_transform(grid)` - forward/inverse sht
- `sht.forward(field)` - spatial to spectral
- `sht.inverse(coeffs, field)` - spectral to spatial
- `sht.power_spectrum(coeffs)` - power by degree l

### solvers
- `heat_solver(grid, diffusivity, method)` - diffusion equation
- `solver.solve(t_final, dt, save_every)` - returns solution history
- `solver.set_initial_condition(field)` - set initial state

### visualization
- `spyre.plot(field, projection="mollweide")` - 2d map (mollweide, orthographic, robinson, platecarree)
- `spyre.plot3d(field, backend="plotly")` - interactive 3d globe (plotly or matplotlib)
- `spyre.plot_coefficients(field)` - power spectrum
- `spyre.animate(history, output="file.mp4")` - animation (3d or 2d projections)
- `spyre.animate_interactive(history)` - interactive animation widget

## examples

See `examples/` directory for comprehensive demos:
- `heat_animation_demo.py` - gaussian blobs diffusing and merging
- `heat_pole_to_pole.py` - meridional heat transfer
- `heat_spherical_harmonics.py` - spectral accuracy test
- `heat_rings.py` - concentric ring diffusion patterns
- `heat_checkerboard.py` - high-frequency diffusion
- `heat_diffusivity_comparison.py` - parameter study

Run any example:
```bash
python examples/heat_pole_to_pole.py
```

## license

mit
