#!/usr/bin/env python3
"""
Test energy conservation in heat equation solver
"""

import sys
sys.path.insert(0, "/Users/aniket/Development/spyre/python")

import numpy as np
import spyre

# create sphere with l_max = 63
grid = spyre.sphere(l_max=63)
print(f"grid: {grid.n_lat} x {grid.n_lon}")

# create initial condition: two gaussian blobs
u0 = spyre.field(grid)
def initial_condition(theta, phi):
    # main blob at theta=pi/4, phi=0
    blob1 = np.exp(-((theta - np.pi/4)**2 + phi**2) / 0.08)
    # second blob at theta=3pi/4, phi=pi
    phi_wrapped = phi - np.pi if phi > np.pi else phi + np.pi
    blob2 = 0.5 * np.exp(-((theta - 3*np.pi/4)**2 + phi_wrapped**2) / 0.08)
    return blob1 + blob2

u0.from_function(initial_condition)

# compute initial integral using quadrature weights
weights = grid.weights
data_initial = np.asarray(u0.data)
initial_integral = np.sum(data_initial * weights[:, np.newaxis])
initial_mean = initial_integral / (4 * np.pi)

print(f"\nInitial statistics:")
print(f"  min: {u0.min():.6f}")
print(f"  max: {u0.max():.6f}")
print(f"  integral: {initial_integral:.6f}")
print(f"  mean: {initial_mean:.6f}")
print(f"  expected final uniform value: {initial_mean:.6f}")

# create heat solver with IMEX Euler (the one we fixed)
diffusivity = 0.02
solver = spyre.heat_solver(grid, diffusivity, spyre.time_integrator.imex_euler)
solver.set_initial_condition(u0)

# solve
t_final = 3.0
dt = 0.005
save_every = 100

history = solver.solve(t_final, dt, save_every)

print(f"\nSolution progress:")
for i, (t, field) in enumerate(zip(history.times, history.fields)):
    data = np.asarray(field.data)
    integral = np.sum(data * weights[:, np.newaxis])
    mean = integral / (4 * np.pi)
    conservation_error = abs(integral - initial_integral) / initial_integral

    print(f"  t={t:.2f}: min={field.min():.6f}, max={field.max():.6f}, "
          f"integral={integral:.6f}, mean={mean:.6f}, error={conservation_error:.2e}")

# check final state
final_field = history.fields[-1]
data_final = np.asarray(final_field.data)
final_integral = np.sum(data_final * weights[:, np.newaxis])
final_mean = final_integral / (4 * np.pi)

print(f"\nFinal statistics:")
print(f"  min: {final_field.min():.6f}")
print(f"  max: {final_field.max():.6f}")
print(f"  integral: {final_integral:.6f}")
print(f"  mean: {final_mean:.6f}")

conservation_error = abs(final_integral - initial_integral) / initial_integral
print(f"\nEnergy conservation error: {conservation_error:.2e}")
print(f"Energy retention: {(final_integral / initial_integral) * 100:.2f}%")

if conservation_error < 0.01:
    print("✓ Energy is conserved (error < 1%)")
else:
    print("✗ Energy is NOT conserved!")
