#!/usr/bin/env python3
"""
Heat equation with adaptive colormap to show diffusion clearly
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre
import matplotlib.pyplot as plt

# create sphere
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: two gaussian blobs
u0 = spyre.field(grid)
def initial_condition(theta, phi):
    blob1 = np.exp(-((theta - np.pi/4)**2 + phi**2) / 0.08)
    phi_wrapped = phi - np.pi if phi > np.pi else phi + np.pi
    blob2 = 0.5 * np.exp(-((theta - 3*np.pi/4)**2 + phi_wrapped**2) / 0.08)
    return blob1 + blob2

u0.from_function(initial_condition)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# solve
diffusivity = 0.02
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

t_final = 3.0
dt = 0.005
save_every = 8

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create comparison plots with ADAPTIVE colormaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

g = history.fields[0].grid
lats = 90 - np.rad2deg(g.latitudes)
lons = np.rad2deg(g.longitudes)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# initial state
ax = axes[0]
data = np.asarray(history.fields[0].data)
im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="inferno", vmin=0, vmax=data.max())
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title("t = 0 (initial)")
plt.colorbar(im, ax=ax, shrink=0.8)

# final state with ADAPTIVE colormap
ax = axes[1]
data = np.asarray(history.fields[-1].data)
im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="inferno", vmin=0, vmax=data.max())
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title(f"t = {t_final} (final) - adaptive colormap")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("heat_comparison_adaptive.png", dpi=150)
print("saved heat_comparison_adaptive.png")

# also create a 4-panel plot showing time evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

times_to_plot = [0, len(history)//3, 2*len(history)//3, -1]
for i, idx in enumerate(times_to_plot):
    ax = axes[i]
    field = history.fields[idx]
    t = history.times[idx]
    data = np.asarray(field.data)

    # use adaptive vmax for each panel
    im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="inferno", vmin=0, vmax=data.max())
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"t = {t:.2f}, max = {data.max():.4f}")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("heat_evolution.png", dpi=150)
print("saved heat_evolution.png")
