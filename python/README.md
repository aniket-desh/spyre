# spyre - Python Package

A fast, efficient library for solving PDEs on a sphere with Python bindings.

## Installation

### Prerequisites

Before installing, you need to have the following C++ dependencies:

**macOS:**
```bash
brew install cmake eigen fftw

# build shtns from source
git clone https://bitbucket.org/nschaeff/shtns.git
cd shtns && ./configure --enable-openmp && make && sudo make install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install cmake libeigen3-dev libfftw3-dev

# build shtns from source
git clone https://bitbucket.org/nschaeff/shtns.git
cd shtns && ./configure --enable-openmp && make && sudo make install
```

### Install from source

```bash
# clone the repository
git clone https://github.com/spyre-dev/spyre.git
cd spyre

# install the python package
pip install python/

# or with visualization dependencies
pip install python/[viz]

# or with all optional dependencies
pip install python/[all]
```

## Quick Start

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

# visualize
spyre.plot(history[-1], projection="mollweide")
```

## Features

- **Spherical harmonic transforms** - wrapped shtns for O(n² log n) transforms
- **Spectral PDE solvers** - heat equation, advection (more coming)
- **Time integration** - euler, rk4, ssp-rk3, imex schemes
- **Visualization** - 3d sphere plots, 2d map projections, animations
- **NumPy compatible** - seamless integration with scientific Python ecosystem

## Documentation

For detailed documentation, API reference, and examples, see the [main repository](https://github.com/spyre-dev/spyre).

## License

MIT
