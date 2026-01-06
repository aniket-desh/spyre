"""
heat equation example

demonstrates solving the heat equation on a sphere with spyre
"""

import numpy as np
import spyre

# create sphere grid
g = spyre.sphere(l_max=63, grid_type="gauss")
print(f"grid: {g.n_lat} x {g.n_lon} (l_max = {g.l_max})")

# initial condition: gaussian bump
u0 = spyre.field(g)

def initial_condition(theta, phi):
    # centered at equator
    d_theta = theta - np.pi / 2
    d_phi = phi
    if d_phi > np.pi:
        d_phi -= 2 * np.pi

    sigma = 0.3
    return np.exp(-(d_theta**2 + d_phi**2) / (2 * sigma**2))

u0.from_function(initial_condition)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# create and run solver
solver = spyre.heat_solver(g, diffusivity=0.01)
solver.set_initial_condition(u0)

history = solver.solve(t_final=5.0, dt=0.01, save_every=50)
print(f"solved {len(history)} snapshots")

# plot initial and final states
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    spyre.plot(history.fields[0], projection="mollweide", ax=axes[0])
    axes[0].set_title(f"t = {history.times[0]:.2f}")

    spyre.plot(history.fields[-1], projection="mollweide", ax=axes[1])
    axes[1].set_title(f"t = {history.times[-1]:.2f}")

    plt.tight_layout()
    plt.savefig("heat_equation.png", dpi=150)
    print("saved heat_equation.png")

except ImportError:
    print("matplotlib not available, skipping plot")

# create animation if ffmpeg available
try:
    spyre.animate(history, projection="mollweide", fps=15, output="heat.mp4")
    print("saved heat.mp4")
except Exception as e:
    print(f"animation failed: {e}")
