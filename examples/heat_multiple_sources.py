#!/usr/bin/env python3
"""
Multiple Heat Sources Demo
Demonstrates heat diffusion from multiple point sources distributed across the sphere
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create high-res sphere
grid = spyre.sphere(l_max=95)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: multiple gaussian heat sources
u0 = spyre.field(grid)

def multiple_sources(theta, phi):
    """Four heat sources at different locations"""
    sources = [
        (np.pi/4, 0.0, 1.0),           # north-east, hot
        (3*np.pi/4, np.pi, 0.8),       # south-west, warm
        (np.pi/2, np.pi/2, 0.6),       # equator-east, medium
        (np.pi/2, 3*np.pi/2, 0.5),     # equator-west, cool
    ]

    result = 0.0
    for theta0, phi0, amplitude in sources:
        # proper angular distance
        dphi = np.minimum(np.abs(phi - phi0), 2*np.pi - np.abs(phi - phi0))
        distance_sq = (theta - theta0)**2 + dphi**2
        result += amplitude * np.exp(-distance_sq / 0.03)

    return result

u0.from_function(multiple_sources)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver with moderate diffusivity
diffusivity = 0.015
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 4.0
dt = 0.005
save_every = 10

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create 3D animation
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_multiple_sources_3d.mp4"
spyre.animate(history, projection="3d", cmap="plasma", fps=30, output=out3d)
print(f"saved to {out3d}")

# create 2D mollweide animation
print("\ncreating 2D mollweide animation...")
out2d = "/Users/aniket/Development/spyre/examples/heat_multiple_sources_2d.mp4"
spyre.animate(history, projection="mollweide", cmap="plasma", fps=30, output=out2d)
print(f"saved to {out2d}")

print("\nDemo complete!")
