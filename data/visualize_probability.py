import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import shape
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def load_probability_surface(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    lat_edges = np.array(data["lat_edges"])
    lon_edges = np.array(data["lon_edges"])
    time_steps = data["time_steps"]

    probabilities = []
    time_labels = []
    for ts in time_steps:
        prob = ts["probability"]
        if prob is None:
            probabilities.append(None)
        else:
            probabilities.append(np.array(prob))
        time_labels.append(ts["time_label"])

    return lat_edges, lon_edges, probabilities, time_labels


def create_probability_dashboard(json_path: str):
    lat_edges, lon_edges, probabilities, time_labels = load_probability_surface(json_path)

    n_times = len(probabilities)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # Load California GeoJSON and draw coastline if cartopy is available
    if HAS_CARTOPY:
        try:
            with open("data/California.geojson", "r") as f:
                cal_data = json.load(f)
            geoms = []
            for feat in cal_data.get("features", []):
                geom_data = feat.get("geometry")
                if geom_data is None:
                    continue
                geoms.append(shape(geom_data))
            if geoms:
                land_feature = cfeature.ShapelyFeature(
                    geoms,
                    ccrs.PlateCarree(),
                    facecolor="lightgray",
                    edgecolor="black",
                    alpha=0.5,
                )
                ax.add_feature(land_feature, zorder=2)
        except Exception:
            pass

    # Initial time index
    current_t = 0
    prob0 = probabilities[current_t]

    if prob0 is not None:
        # Mask zero-probability cells so they are transparent
        masked_prob = np.ma.masked_where(prob0 == 0.0, prob0)
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
        ax.pcolormesh(
            lon_grid,
            lat_grid,
            masked_prob,
            cmap="viridis",
            shading="auto",
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
            zorder=1,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Probability Surface - Time Step {current_t}\n{time_labels[current_t]}")

    # Slider for time steps
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Time Step", 0, n_times - 1, valinit=current_t, valstep=1)

    def update(val):
        t = int(slider.val)
        prob = probabilities[t]
        ax.clear()

        # Redraw California coastline if available
        if HAS_CARTOPY:
            try:
                with open("data/California.geojson", "r") as f:
                    cal_data = json.load(f)
                geoms = []
                for feat in cal_data.get("features", []):
                    geom_data = feat.get("geometry")
                    if geom_data is None:
                        continue
                    geoms.append(shape(geom_data))
                if geoms:
                    land_feature = cfeature.ShapelyFeature(
                        geoms,
                        ccrs.PlateCarree(),
                        facecolor="lightgray",
                        edgecolor="black",
                        alpha=0.5,
                    )
                    ax.add_feature(land_feature, zorder=2)
            except Exception:
                pass

        if prob is not None:
            masked_prob = np.ma.masked_where(prob == 0.0, prob)
            lon_grid, lat_grid = np.meshgrid(
                0.5 * (lon_edges[:-1] + lon_edges[1:]),
                0.5 * (lat_edges[:-1] + lat_edges[1:]),
            )
            ax.pcolormesh(
                lon_grid,
                lat_grid,
                masked_prob,
                cmap="viridis",
                shading="auto",
                transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
                zorder=1,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Probability Surface - Time Step {t}\n{time_labels[t]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    json_path = "data/case_study_windsurfer/ex1_probability_surface.json"

    create_probability_dashboard(json_path)
