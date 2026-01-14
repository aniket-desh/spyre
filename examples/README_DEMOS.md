# Spyre PDE Animation Demos

This directory contains demonstration scripts showcasing various PDE solvers and initial conditions on the sphere.

## How Animations Work

The animation pipeline has three stages:

1. **PDE Solver**: Solve the heat equation using `spyre.heat_solver`
   - Returns a `solution_history` object containing time snapshots
   - Uses spectral methods with spherical harmonics

2. **Animation**: Pass history to `spyre.animate()`
   - Supports 3D sphere view and 2D map projections
   - Uses matplotlib for rendering
   - Exports to MP4 or GIF

3. **Rendering**: Maps field data onto sphere surface
   - Proper handling of periodic boundaries
   - Smooth color interpolation

## Available Demos

### 1. heat_animation_demo.py
**Original demo** - Two Gaussian blobs diffusing and merging
- Initial: 2 hot spots at different latitudes
- Colormap: inferno
- Output: 3D animation + 2D comparison

### 2. heat_multiple_sources.py
**Multiple Heat Sources** - Four sources at different locations and temperatures
- Shows interaction between multiple diffusion centers
- High resolution (l_max=95)
- Colormap: plasma
- Output: Both 3D and 2D Mollweide animations

### 3. heat_pole_to_pole.py
**Meridional Heat Transfer** - Hot north pole, cold south pole
- Tests meridional (north-south) diffusion
- Creates smooth gradient evolution
- Colormap: coolwarm
- Output: 3D animation + multi-panel comparison plot

### 4. heat_spherical_harmonics.py
**Spectral Accuracy Test** - Superposition of spherical harmonic modes
- Initial: Y_2^0 + 0.5×Y_4^2 + 0.3×Y_8^4
- High modes decay faster (spectral diffusion)
- Includes power spectrum analysis
- Colormap: RdBu_r
- Output: 3D animation + spectral decay plot

### 5. heat_rings.py
**Concentric Rings** - Alternating hot/cold rings from pole
- Beautiful radial diffusion patterns
- Tests azimuthal symmetry
- High resolution (l_max=95)
- Colormap: viridis
- Output: 3D + orthographic projection

### 6. heat_checkerboard.py
**Checkerboard Pattern** - Alternating hot/cold squares
- 8×16 grid of patches
- Tests high-frequency diffusion
- Colormap: coolwarm
- Output: 3D + 2D Mollweide

### 7. heat_diffusivity_comparison.py
**Parameter Study** - Three diffusivities (κ = 0.005, 0.02, 0.08)
- Same initial condition, different rates
- Side-by-side comparison
- Colormap: hot
- Output: 3 animations + comparison grid plot

## Running the Demos

```bash
# Activate virtual environment
source .venv/bin/activate

# Run any demo
python examples/heat_pole_to_pole.py

# Run all demos (warning: takes ~10 minutes)
for demo in examples/heat_*.py; do
    echo "Running $demo..."
    python "$demo"
done
```

## Parameters Explained

### Grid Resolution
- `l_max`: Maximum spherical harmonic degree
- Standard: 63 (64×128 grid)
- High-res: 95 (96×192 grid)

### Diffusivity (κ)
- **Slow** (0.005-0.01): Detailed evolution, long timescales
- **Medium** (0.015-0.02): Balanced visualization
- **Fast** (0.05-0.08): Rapid diffusion

### Time Integration
- `imex_euler`: Implicit-Explicit Euler (recommended for diffusion)
- `rk4`: 4th-order Runge-Kutta (explicit)
- `ssp_rk3`: Strong Stability Preserving RK3

### Animation Settings
- `fps`: Frames per second (24-30 recommended)
- `save_every`: Snapshot interval (higher = fewer frames)
- `cmap`: Matplotlib colormap

## Creating Custom Demos

Template for new demos:

```python
import spyre

# 1. Create grid
grid = spyre.sphere(l_max=63)

# 2. Define initial condition
u0 = spyre.field(grid)
def my_initial_condition(theta, phi):
    # theta: colatitude [0, pi]
    # phi: longitude [0, 2pi]
    return ...  # your function

u0.from_function(my_initial_condition)

# 3. Create solver
solver = spyre.heat_solver(grid, diffusivity=0.02)
solver.set_initial_condition(u0)

# 4. Solve
history = solver.solve(t_final=3.0, dt=0.005, save_every=8)

# 5. Animate
spyre.animate(history, projection="3d", cmap="viridis",
              fps=30, output="my_demo.mp4")
```

## Future Enhancements

Potential demos to add (require additional solver bindings):

1. **Advection**: Solid-body rotation, deformational flow
2. **Reaction-Diffusion**: Turing patterns, Gray-Scott model
3. **Coupled Systems**: Multi-field dynamics
4. **Nonlinear PDEs**: Burgers equation, shallow water

## Tips for Better Visualizations

1. **Color scales**: Use perceptually uniform colormaps (viridis, plasma, cividis)
2. **Resolution**: Match l_max to feature scale (higher for sharp gradients)
3. **Time stepping**: Keep CFL number moderate (dt ~ 0.005 works well)
4. **Frame rate**: 24-30 fps for smooth playback
5. **File size**: Reduce `save_every` for shorter videos

## Citation

If you use these demos in publications, please cite:
```
spyre - Fast Spherical PDE Solver
https://github.com/spyre-dev/spyre
```
