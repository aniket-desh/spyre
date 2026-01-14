#!/usr/bin/env python3
"""
Concentric Rings Demo
Alternating hot/cold rings centered on the north pole
Creates beautiful radial diffusion patterns
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create sphere
grid = spyre.sphere(l_max=95)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: concentric rings from pole
u0 = spyre.field(grid)

def concentric_rings(theta, phi):
    """Alternating hot/cold rings based on colatitude"""
    # create rings using sin function of theta
    # theta ranges from 0 (north pole) to pi (south pole)
    rings = np.sin(8 * theta)  # 8 rings from pole to pole

    # make it positive and add offset for better visualization
    return 0.5 + 0.5 * rings

u0.from_function(concentric_rings)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver with low diffusivity for detailed evolution
diffusivity = 0.008
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 3.0
dt = 0.005
save_every = 8

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create 3D animation with viridis colormap
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_rings_3d.mp4"
spyre.animate(history, projection="3d", cmap="viridis", fps=30, output=out3d)
print(f"saved to {out3d}")

# create orthographic projection animation (view from north pole)
print("\ncreating orthographic animation...")
out2d = "/Users/aniket/Development/spyre/examples/heat_rings_ortho.mp4"
spyre.animate(history, projection="orthographic", cmap="viridis", fps=30, output=out2d)
print(f"saved to {out2d}")

print("\nDemo complete!")
