"""
animation functions for spyre
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import solution_history


def animate(
    history: solution_history,
    projection: str = "3d",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    fps: int = 30,
    output: str | None = None,
    title: str = "t = {t:.3f}",
    **kwargs,
):
    """
    animate solution history

    args:
        history: solution history from solver
        projection: "3d", "mollweide", "orthographic", etc
        cmap: colormap name
        vmin, vmax: color scale limits (auto if none)
        fps: frames per second
        output: output filename (mp4, gif, or none for display)
        title: title format string (uses {t} for time)
        **kwargs: passed to plotting function

    returns:
        matplotlib animation or html for jupyter
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    if len(history) == 0:
        raise ValueError("empty solution history")

    # compute global color limits if not provided
    if vmin is None or vmax is None:
        all_min = min(f.min() for f in history.fields)
        all_max = max(f.max() for f in history.fields)
        if vmin is None:
            vmin = all_min
        if vmax is None:
            vmax = all_max

    is_3d = projection == "3d"

    if is_3d:
        # 3d animation with matplotlib
        from matplotlib import cm

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        g = history.fields[0].grid
        theta = np.asarray(g.latitudes)[:, np.newaxis]
        phi = np.asarray(g.longitudes)[np.newaxis, :]

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta) * np.ones_like(phi)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        data = np.asarray(history.fields[0].data)
        colors = cm.get_cmap(cmap)(norm(data))

        surf = ax.plot_surface(x, y, z, facecolors=colors, shade=False)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        txt = ax.set_title(title.format(t=history.times[0]))

        def update(frame):
            nonlocal surf
            surf.remove()
            data = np.asarray(history.fields[frame].data)
            colors = cm.get_cmap(cmap)(norm(data))
            surf = ax.plot_surface(x, y, z, facecolors=colors, shade=False)
            txt.set_text(title.format(t=history.times[frame]))
            return [surf, txt]

    else:
        # 2d map projection animation
        try:
            import cartopy.crs as ccrs
            has_cartopy = True
        except ImportError:
            has_cartopy = False

        if has_cartopy:
            proj_map = {
                "mollweide": ccrs.Mollweide(),
                "orthographic": ccrs.Orthographic(),
                "robinson": ccrs.Robinson(),
            }
            proj = proj_map.get(projection, ccrs.Mollweide())
            fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(10, 6))
            transform = ccrs.PlateCarree()
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            transform = None

        g = history.fields[0].grid
        lats = 90 - np.rad2deg(g.latitudes)
        lons = np.rad2deg(g.longitudes)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        data = np.asarray(history.fields[0].data)

        if has_cartopy:
            im = ax.pcolormesh(
                lon_grid, lat_grid, data,
                transform=transform,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            ax.coastlines(linewidth=0.5, color="gray")
            ax.set_global()
        else:
            im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, ax=ax, shrink=0.6)
        txt = ax.set_title(title.format(t=history.times[0]))

        def update(frame):
            data = np.asarray(history.fields[frame].data)
            im.set_array(data.ravel())
            txt.set_text(title.format(t=history.times[frame]))
            return [im, txt]

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000 // fps, blit=False)

    if output:
        if output.endswith(".gif"):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)
        plt.close()
        return output
    else:
        plt.close()
        # return html for jupyter display
        from IPython.display import HTML
        return HTML(anim.to_jshtml())


def animate_interactive(
    history: solution_history,
    projection: str = "3d",
    cmap: str = "viridis",
    **kwargs,
):
    """
    create interactive animation widget for jupyter

    args:
        history: solution history
        projection: "3d" or map projection
        cmap: colormap
        **kwargs: passed to plotting

    returns:
        ipywidgets widget
    """
    try:
        from ipywidgets import interact, IntSlider, Play, jslink, HBox, VBox, Output
    except ImportError:
        raise ImportError("ipywidgets required for interactive animation")

    import matplotlib.pyplot as plt
    from . import plot, plot3d

    vmin = min(f.min() for f in history.fields)
    vmax = max(f.max() for f in history.fields)

    out = Output()

    def update(frame):
        with out:
            out.clear_output(wait=True)
            f = history.fields[frame]
            t = history.times[frame]

            if projection == "3d":
                fig = plot3d(f, backend="matplotlib", cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot(f, projection=projection, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, **kwargs)

            plt.title(f"t = {t:.3f}")
            plt.show()

    slider = IntSlider(min=0, max=len(history) - 1, step=1, value=0, description="frame")
    play = Play(min=0, max=len(history) - 1, step=1, interval=100)
    jslink((play, "value"), (slider, "value"))

    interact(update, frame=slider)

    return VBox([HBox([play, slider]), out])
