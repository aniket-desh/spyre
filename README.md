```
                                                                                                  
                                              ░░░░▒▒░░                                            
                                        ░░▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒                                        
                                      ▓▓▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒░░                                    
                                    ▓▓▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒▓▓░░                                  
                                  ▓▓▓▓▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▓▓                                  
                                ░░██▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▒▒▓▓▓▓                                
                                ▓▓██▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▓▓                                
                                ████▓▓▓▓▓▓▒▒▒▒░░      ░░░░▒▒▒▒▒▒▓▓░░                              
                                ████▓▓▓▓▓▓▓▓▒▒░░      ░░░░▒▒▒▒▓▓▓▓▒▒                              
                                ██████▓▓▓▓▓▓▒▒░░      ░░░░▒▒▒▒▓▓▓▓▒▒                              
                                ██████████▓▓▒▒░░░░  ░░░░▒▒▒▒▒▒▓▓▓▓▒▒                              
                                ██████████▓▓▒▒▒▒░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓░░                              
                                ▓▓████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓                                
                                ░░████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒                                
                                  ▓▓██████▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓                                  
                                    ██████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░                                  
                                    ░░████████▓▓▓▓▓▓▓▓▓▓▓▓██░░                                    
                                        ▓▓████████████████░░                                      
                                            ▒▒▓▓██▓▓▒▒                                            
```

# spyre
fast, efficient library for solving pdes on a sphere with python bindings.

## features

- **spherical harmonic transforms** - wrapped shtns for O(n² log n) transforms
- **spectral pde solvers** - heat equation, advection (more coming)
- **time integration** - euler, rk4, ssp-rk3, imex schemes
- **visualization** - 3d sphere plots, 2d map projections, animations
- **python bindings** - numpy-compatible via pybind11

## dependencies

```
c++: eigen, fftw3, shtns, pybind11
python: numpy, matplotlib
```

## build

```bash
# install dependencies (macos)
brew install eigen fftw

# build shtns from source
git clone https://bitbucket.org/nschaeff/shtns.git
cd shtns && ./configure --enable-openmp && make && sudo make install

# build spyre
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

### visualization
- `spyre.plot(field, projection="mollweide")` - 2d map
- `spyre.plot3d(field)` - interactive 3d globe
- `spyre.animate(history, output="file.mp4")` - animation

## license

mit
