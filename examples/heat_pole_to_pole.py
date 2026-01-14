#!/usr/bin/env python3
"""
Pole-to-Pole Heat Transfer Demo
Hot north pole, cold south pole - watch heat diffuse meridionally
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre
import matplotlib.pyplot as plt

# create sphere
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: hot north pole, cold south pole
u0 = spyre.field(grid)

def pole_to_pole(theta, phi):
    """Temperature gradient from north to south pole"""
    # theta = 0 is north pole (hot), theta = pi is south pole (cold)
    # smooth transition with tanh
    return 0.5 * (1.0 - np.tanh(5 * (theta - np.pi/2)))

u0.from_function(pole_to_pole)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver with low diffusivity for slower evolution
diffusivity = 0.01
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 5.0
dt = 0.01
save_every = 5

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create static comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
times_to_plot = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]

for i, idx in enumerate(times_to_plot[:5]):
    ax = axes.flat[i]
    g = history.fields[idx].grid
    data = np.asarray(history.fields[idx].data)
    lats = 90 - np.rad2deg(g.latitudes)
    lons = np.rad2deg(g.longitudes)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"t = {history.times[idx]:.2f}")
    plt.colorbar(im, ax=ax, shrink=0.8)

# hide last subplot
axes.flat[5].axis('off')

plt.tight_layout()
plt.savefig("/Users/aniket/Development/spyre/examples/heat_pole_to_pole.png", dpi=150)
print("saved comparison plot")

# create 3D animation
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_pole_to_pole_3d.mp4"
spyre.animate(history, projection="3d", cmap="coolwarm", fps=24, output=out3d)
print(f"saved to {out3d}")

print("\nDemo complete!")
