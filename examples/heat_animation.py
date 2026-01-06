#!/opt/homebrew/bin/python3.13
"""
heat equation animation on a sphere
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create sphere with l_max = 63 (64x128 grid)
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: gaussian blob centered at (lat=45°, lon=0°)
u0 = spyre.field(grid)
u0.from_function(lambda theta, phi: np.exp(-((theta - np.pi/4)**2 + phi**2) / 0.1))
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver with diffusivity
diffusivity = 0.01
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve and save snapshots
t_final = 5.0
dt = 0.01
save_every = 10  # save every 10 steps

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# animate on 3d sphere
print("creating animation...")
output_file = "/Users/aniket/Development/spyre/examples/heat_sphere.mp4"
spyre.animate(history, projection="3d", cmap="hot", fps=30, output=output_file)
print(f"saved to {output_file}")
