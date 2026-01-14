"""
visualization functions for spyre
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import field, spherical_transform


def plot(
    f: field,
    projection: str = "mollweide",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    central_lon: float = 0.0,
    title: str | None = None,
    colorbar: bool = True,
    ax=None,
    **kwargs,
):
    """
    plot scalar field using map projection

    args:
        f: scalar field to plot
        projection: map projection (mollweide, orthographic, robinson, etc)
        cmap: colormap name
        vmin, vmax: color scale limits
        central_lon: central longitude for projection (degrees)
        title: plot title
        colorbar: show colorbar
        ax: existing axes to plot on
        **kwargs: passed to pcolormesh

    returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    g = f.grid
    data = np.asarray(f.data)

    # convert colatitude to latitude in degrees
    lats = 90 - np.rad2deg(g.latitudes)
    lons = np.rad2deg(g.longitudes)

    # create meshgrid for pcolormesh
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    if has_cartopy and projection != "none":
        # use cartopy projection
        proj_map = {
            "mollweide": ccrs.Mollweide(central_longitude=central_lon),
            "orthographic": ccrs.Orthographic(central_longitude=central_lon),
            "robinson": ccrs.Robinson(central_longitude=central_lon),
            "platecarree": ccrs.PlateCarree(central_longitude=central_lon),
        }

        if projection not in proj_map:
            raise ValueError(f"unknown projection: {projection}")

        proj = proj_map[projection]

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(10, 6))

        im = ax.pcolormesh(
            lon_grid, lat_grid, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        ax.coastlines(linewidth=0.5, color="gray")
        ax.set_global()

    else:
        # fallback to simple 2d plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        ax.set_aspect("equal")

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6, label="")

    if title:
        ax.set_title(title)

    return ax


def plot3d(
    f: field,
    backend: str = "plotly",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    wireframe: bool = False,
    title: str | None = None,
    **kwargs,
):
    """
    interactive 3d sphere plot

    args:
        f: scalar field to plot
        backend: "plotly" or "matplotlib"
        cmap: colormap name
        vmin, vmax: color scale limits
        wireframe: show wireframe
        title: plot title
        **kwargs: passed to surface plot

    returns:
        figure object
    """
    g = f.grid
    data = np.asarray(f.data)

    # convert to cartesian coordinates
    theta = np.asarray(g.latitudes)[:, np.newaxis]  # colatitude
    phi = np.asarray(g.longitudes)[np.newaxis, :]   # longitude

    # close the mesh at the longitude seam by appending first column
    phi_closed = np.concatenate([phi, phi[:, :1] + 2 * np.pi], axis=1)
    theta_closed = np.concatenate([theta, theta], axis=1)
    data_closed = np.concatenate([data, data[:, :1]], axis=1)

    x = np.sin(theta_closed) * np.cos(phi_closed)
    y = np.sin(theta_closed) * np.sin(phi_closed)
    z = np.cos(theta_closed)

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Surface(
                x=x, y=y, z=z,
                surfacecolor=data_closed,
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                **kwargs,
            )
        ])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cm.get_cmap(cmap)(norm(data_closed))

        ax.plot_surface(x, y, z, facecolors=colors, shade=False,
                        rstride=1, cstride=1, antialiased=True, **kwargs)

        if wireframe:
            ax.plot_wireframe(x, y, z, color="gray", linewidth=0.3, alpha=0.3)

        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

        if title:
            ax.set_title(title)

        return fig

    else:
        raise ValueError(f"unknown backend: {backend}")


def plot_coefficients(
    f: field,
    sht: spherical_transform | None = None,
    ax=None,
    **kwargs,
):
    """
    plot power spectrum of scalar field

    args:
        f: scalar field
        sht: spherical transform (created if not provided)
        ax: existing axes
        **kwargs: passed to semilogy

    returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt
    from . import spherical_transform as sht_class

    if sht is None:
        sht = sht_class(f.grid)

    coeffs = sht.forward(f)
    spectrum = sht.power_spectrum(coeffs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    l_vals = np.arange(len(spectrum))
    ax.semilogy(l_vals, spectrum, ".-", **kwargs)

    ax.set_xlabel("spherical harmonic degree l")
    ax.set_ylabel("power P(l)")
    ax.set_title("power spectrum")
    ax.grid(True, alpha=0.3)

    return ax
