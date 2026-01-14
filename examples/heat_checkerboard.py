#!/usr/bin/env python3
"""
Checkerboard Pattern Demo
Alternating hot/cold patches on the sphere
Watch the pattern smooth out through diffusion
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create sphere
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: checkerboard pattern
u0 = spyre.field(grid)

def checkerboard(theta, phi):
    """Checkerboard pattern based on lat/lon grid"""
    # divide sphere into squares
    n_lat_squares = 8
    n_lon_squares = 16

    lat_idx = int(theta / np.pi * n_lat_squares)
    lon_idx = int(phi / (2*np.pi) * n_lon_squares)

    # checkerboard: hot if (lat_idx + lon_idx) is even, cold if odd
    if (lat_idx + lon_idx) % 2 == 0:
        return 1.0
    else:
        return 0.0

u0.from_function(checkerboard)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver
diffusivity = 0.015
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 3.0
dt = 0.005
save_every = 8

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create 3D animation
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_checkerboard_3d.mp4"
spyre.animate(history, projection="3d", cmap="coolwarm", fps=30, output=out3d)
print(f"saved to {out3d}")

# create 2D mollweide animation
print("\ncreating 2D animation...")
out2d = "/Users/aniket/Development/spyre/examples/heat_checkerboard_2d.mp4"
spyre.animate(history, projection="mollweide", cmap="coolwarm", fps=30, output=out2d)
print(f"saved to {out2d}")

print("\nDemo complete!")
