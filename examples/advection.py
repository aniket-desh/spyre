"""
advection example

demonstrates solid body rotation on a sphere
"""

import numpy as np
import spyre

# create sphere grid
g = spyre.sphere(l_max=63, grid_type="gauss")
print(f"grid: {g.n_lat} x {g.n_lon}")

# initial condition: cosine bell
u0 = spyre.field(g)

# bell center at (pi/3, pi)
theta0, phi0 = np.pi / 3, np.pi

def cosine_bell(theta, phi):
    # great circle distance
    cos_d = np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0) + np.cos(theta) * np.cos(theta0)
    d = np.arccos(np.clip(cos_d, -1, 1))

    r = np.pi / 6  # bell radius
    if d < r:
        return 0.5 * (1 + np.cos(np.pi * d / r))
    return 0.0

u0.from_function(cosine_bell)
print(f"initial: min={u0.min():.4f} max={u0.max():.4f}")

# solid body rotation velocity field
# v_theta = 0, v_phi = constant (rotation around pole)
omega = 2 * np.pi / 12  # one rotation in t=12

def v_theta(theta, phi):
    return 0.0

def v_phi(theta, phi):
    return omega * np.sin(theta)  # angular velocity

# note: advection solver not yet fully implemented
# this is a placeholder showing the api
print("advection solver api demonstration (full impl coming in phase 2)")

# visualize initial condition
try:
    spyre.plot(u0, projection="orthographic", title="cosine bell initial condition")
    import matplotlib.pyplot as plt
    plt.savefig("advection_initial.png", dpi=150)
    print("saved advection_initial.png")
except ImportError:
    pass
