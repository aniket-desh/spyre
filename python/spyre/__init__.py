"""
spyre - fast spherical pde solver library
"""

from __future__ import annotations

__version__ = "0.1.0"

# import c++ bindings
try:
    from ._spyre_core import (
        # grids
        grid,
        gauss_legendre_grid,
        equiangular_grid,
        # fields
        field,
        vector_field,
        # transforms
        spherical_transform,
        # solvers
        time_integrator,
        solution_history,
        heat_solver,
        # functions
        laplacian,
    )
except ImportError as e:
    raise ImportError(
        "failed to import spyre c++ bindings. "
        "ensure the library is properly built: pip install -e ."
    ) from e

# python-level convenience


def sphere(
    n_lat: int | None = None,
    n_lon: int | None = None,
    l_max: int | None = None,
    grid_type: str = "gauss",
):
    """
    create a sphere grid

    args:
        n_lat: number of latitude points (for equiangular)
        n_lon: number of longitude points (for equiangular)
        l_max: max spherical harmonic degree (for gauss)
        grid_type: "gauss" or "equiangular"

    returns:
        grid object
    """
    if grid_type in ("gauss", "gauss_legendre"):
        if l_max is None:
            if n_lat is not None:
                l_max = n_lat - 1
            else:
                raise ValueError("must specify l_max or n_lat for gauss grid")
        return gauss_legendre_grid(l_max)
    elif grid_type == "equiangular":
        if n_lat is None or n_lon is None:
            raise ValueError("must specify n_lat and n_lon for equiangular grid")
        return equiangular_grid(n_lat, n_lon)
    else:
        raise ValueError(f"unknown grid type: {grid_type}")


# lazy imports for optional visualization
def plot(*args, **kwargs):
    """plot scalar field on sphere"""
    from .plotting import plot as _plot
    return _plot(*args, **kwargs)


def plot3d(*args, **kwargs):
    """interactive 3d sphere plot"""
    from .plotting import plot3d as _plot3d
    return _plot3d(*args, **kwargs)


def plot_coefficients(*args, **kwargs):
    """plot power spectrum"""
    from .plotting import plot_coefficients as _plot_coefficients
    return _plot_coefficients(*args, **kwargs)


def animate(*args, **kwargs):
    """animate solution history"""
    from .animation import animate as _animate
    return _animate(*args, **kwargs)


def animate_interactive(*args, **kwargs):
    """interactive animation widget"""
    from .animation import animate_interactive as _animate_interactive
    return _animate_interactive(*args, **kwargs)


__all__ = [
    # version
    "__version__",
    # grids
    "grid",
    "gauss_legendre_grid",
    "equiangular_grid",
    "sphere",
    # fields
    "field",
    "vector_field",
    # transforms
    "spherical_transform",
    # solvers
    "time_integrator",
    "solution_history",
    "heat_solver",
    # functions
    "laplacian",
    # visualization
    "plot",
    "plot3d",
    "plot_coefficients",
    "animate",
    "animate_interactive",
]
