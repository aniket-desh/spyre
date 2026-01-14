#!/usr/bin/env python3
"""
Spherical Harmonics Demo
Initial condition is a superposition of spherical harmonics
Tests spectral accuracy - each mode decays at its own rate
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre
import matplotlib.pyplot as plt

# create sphere
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: superposition of spherical harmonics
# Y_l^m decays as exp(-kappa * l(l+1) * t)
u0 = spyre.field(grid)

# create a mix of low and high frequency modes
# Y_2^0 + 0.5*Y_4^2 + 0.3*Y_8^4
u0.from_spherical_harmonic(2, 0)
temp = spyre.field(grid)
temp.from_spherical_harmonic(4, 2)
u0 += 0.5 * temp
temp.from_spherical_harmonic(8, 4)
u0 += 0.3 * temp

# normalize
u_max = u0.max()
u_min = u0.min()
data = np.asarray(u0.data)
data[:] = (data - u_min) / (u_max - u_min)

print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create heat solver
diffusivity = 0.02
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve - watch high frequencies decay faster than low frequencies
t_final = 2.0
dt = 0.005
save_every = 6

history = solver.solve(t_final, dt, save_every)
print(f"solved {len(history)} snapshots over t=[0, {t_final}]")
print(f"final: min={history.fields[-1].min():.4f} max={history.fields[-1].max():.4f}")

# analyze power spectrum evolution
print("\nAnalyzing spectral decay...")
sht = spyre.spherical_transform(grid)
spectra = []
for f in history.fields[::10]:  # sample every 10th frame
    coeffs = sht.forward(f)
    spectrum = sht.power_spectrum(coeffs)
    spectra.append(spectrum)

# plot spectral decay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# power spectrum at different times
for i, idx in enumerate([0, len(spectra)//2, len(spectra)-1]):
    l_vals = np.arange(len(spectra[idx]))
    ax1.semilogy(l_vals, spectra[idx], 'o-', label=f't={history.times[idx*10]:.2f}', markersize=3)

ax1.set_xlabel("spherical harmonic degree l")
ax1.set_ylabel("power P(l)")
ax1.set_title("Spectral decay: high modes decay faster")
ax1.grid(True, alpha=0.3)
ax1.legend()

# field visualization at final time
g = history.fields[-1].grid
data = np.asarray(history.fields[-1].data)
lats = 90 - np.rad2deg(g.latitudes)
lons = np.rad2deg(g.longitudes)
lon_grid, lat_grid = np.meshgrid(lons, lats)
im = ax2.pcolormesh(lon_grid, lat_grid, data, cmap="RdBu_r", shading='auto')
ax2.set_xlabel("longitude (deg)")
ax2.set_ylabel("latitude (deg)")
ax2.set_title(f"Final state t={history.times[-1]:.2f}")
plt.colorbar(im, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig("/Users/aniket/Development/spyre/examples/heat_spherical_harmonics.png", dpi=150)
print("saved spectral analysis plot")

# create 3D animation
print("\ncreating 3D animation...")
out3d = "/Users/aniket/Development/spyre/examples/heat_spherical_harmonics_3d.mp4"
spyre.animate(history, projection="3d", cmap="RdBu_r", fps=30, output=out3d)
print(f"saved to {out3d}")

print("\nDemo complete!")
