#!/usr/bin/env python3
"""
Diffusivity Comparison Demo
Same initial condition with three different diffusivities
Shows how diffusion rate affects heat spreading
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre
import matplotlib.pyplot as plt

# create sphere
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: single hot spot at equator
def hot_spot(theta, phi):
    """Hot spot at equator, 0 degrees longitude"""
    dphi = np.minimum(np.abs(phi), 2*np.pi - np.abs(phi))
    dist_sq = (theta - np.pi/2)**2 + dphi**2
    return np.exp(-dist_sq / 0.05)

# three diffusivities: slow, medium, fast
diffusivities = [0.005, 0.02, 0.08]
histories = []
labels = ['slow (κ=0.005)', 'medium (κ=0.02)', 'fast (κ=0.08)']

for i, kappa in enumerate(diffusivities):
    print(f"\nSolving with diffusivity κ={kappa}...")
    u0 = spyre.field(grid)
    u0.from_function(hot_spot)

    solver = spyre.heat_solver(grid, kappa, spyre.time_integrator.imex_euler)
    solver.set_initial_condition(u0)

    t_final = 2.0
    dt = 0.005
    save_every = 8

    history = solver.solve(t_final, dt, save_every)
    histories.append(history)
    print(f"  final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# create comparison plot at 4 time points
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

time_indices = [0, len(histories[0])//3, 2*len(histories[0])//3, len(histories[0])-1]

for row, (history, label) in enumerate(zip(histories, labels)):
    for col, idx in enumerate(time_indices):
        ax = axes[row, col]
        g = history.fields[idx].grid
        data = np.asarray(history.fields[idx].data)
        lats = 90 - np.rad2deg(g.latitudes)
        lons = np.rad2deg(g.longitudes)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # use same color scale for all
        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="hot", vmin=0, vmax=1)
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("lat (deg)")

        if row == 0:
            ax.set_title(f"t = {history.times[idx]:.2f}")
        if col == 0:
            ax.text(-0.3, 0.5, label, transform=ax.transAxes,
                   rotation=90, va='center', fontsize=12, fontweight='bold')

        if col == 3:  # add colorbar to rightmost plots
            plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Heat Diffusion: Effect of Diffusivity", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("/Users/aniket/Development/spyre/examples/heat_diffusivity_comparison.png", dpi=150)
print("\nsaved comparison plot")

# create animations for each diffusivity
for i, (history, label) in enumerate(zip(histories, labels)):
    print(f"\ncreating animation for {label}...")
    outfile = f"/Users/aniket/Development/spyre/examples/heat_diffusivity_{i+1}_3d.mp4"
    spyre.animate(history, projection="3d", cmap="hot", fps=30, output=outfile)
    print(f"saved to {outfile}")

print("\nDemo complete!")
