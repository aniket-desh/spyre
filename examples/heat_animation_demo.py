#!/usr/bin/env python3
"""
heat equation animation demo - creates both 3D and 2D versions
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create sphere with l_max = 63 (64x128 grid)
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: gaussian blob centered at (theta=pi/4, phi=0)
# plus a smaller blob on the other side
u0 = spyre.field(grid)
def initial_condition(theta, phi):
    # main blob at theta=pi/4, phi=0
    blob1 = np.exp(-((theta - np.pi/4)**2 + phi**2) / 0.08)
    # second blob at theta=3pi/4, phi=pi
    phi_wrapped = phi - np.pi if phi > np.pi else phi + np.pi
    blob2 = 0.5 * np.exp(-((theta - 3*np.pi/4)**2 + phi_wrapped**2) / 0.08)
    return blob1 + blob2

u0.from_function(initial_condition)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver
diffusivity = 0.02
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 3.0
dt = 0.005
save_every = 8

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# 3D sphere animation
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_3d.mp4"
spyre.animate(history, projection="3d", cmap="inferno", fps=24, output=out3d)
print(f"saved to {out3d}")

# also create static plots of initial and final states
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# initial state
ax = axes[0]
g = history.fields[0].grid
data = np.asarray(history.fields[0].data)
lats = 90 - np.rad2deg(g.latitudes)
lons = np.rad2deg(g.longitudes)
lon_grid, lat_grid = np.meshgrid(lons, lats)
im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="inferno", vmin=0, vmax=1)
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title("t = 0 (initial)")
plt.colorbar(im, ax=ax, shrink=0.8)

# final state
ax = axes[1]
data = np.asarray(history.fields[-1].data)
im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="inferno", vmin=0, vmax=1)
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title(f"t = {t_final} (final)")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("/Users/aniket/Development/spyre/examples/heat_comparison.png", dpi=150)
print("saved comparison plot to heat_comparison.png")
