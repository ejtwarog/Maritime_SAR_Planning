"""
Visualize search and rescue simulation results from JSON export.

Displays:
- Drift particle trajectories
- Probability surface evolution
- Search progression (cells searched)
- Real-time metrics

Usage:
    python visualizations/visualize_search.py ../search_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import shape
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


class SearchVisualizer:
    """Interactive visualization of search simulation results."""

    def __init__(self, json_file: str, geojson_file: str = "data/California.geojson"):
        """
        Initialize visualizer.

        Args:
            json_file: Path to exported simulation JSON
            geojson_file: Path to California coastline GeoJSON
        """
        print(f"Loading results from {json_file}...")
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.metadata = self.data["metadata"]
        self.grid = self.data["grid"]
        self.trajectories = np.array(self.data["trajectories"])
        self.probability_surfaces = [np.array(p) if p is not None else None for p in self.data["probability_surfaces"]]
        self.search_results = self.data["search_results"]

        self.lat_edges = np.array(self.grid["lat_edges"])
        self.lon_edges = np.array(self.grid["lon_edges"])

        self.start_step = self.metadata.get("start_step", 0)
        self.total_time_steps = self.metadata.get("total_time_steps", len(self.trajectories))
        self.n_steps = len(self.search_results)
        self.n_particles = self.metadata["n_particles"]
        self.bounds = self.metadata["bounds"]

        # Load coastline
        self.land_geometry = None
        try:
            with open(geojson_file, "r") as f:
                cal_data = json.load(f)
            geoms = []
            for feat in cal_data.get("features", []):
                geom_data = feat.get("geometry")
                if geom_data is not None:
                    geoms.append(shape(geom_data))
            if geoms:
                from shapely.geometry import MultiPolygon
                self.land_geometry = MultiPolygon(geoms) if len(geoms) > 1 else geoms[0]
        except Exception as e:
            print(f"Warning: Could not load coastline: {e}")

        # Setup figure
        self._setup_figure()
        self.current_time_step = 0

    def _setup_figure(self):
        """Setup matplotlib figure with subplots."""
        if HAS_CARTOPY:
            self.fig = plt.figure(figsize=(16, 10))
            self.ax_traj = self.fig.add_axes([0.05, 0.35, 0.4, 0.6], projection=ccrs.PlateCarree())
            self.ax_prob = self.fig.add_axes([0.52, 0.35, 0.4, 0.6], projection=ccrs.PlateCarree())
        else:
            self.fig, (self.ax_traj, self.ax_prob) = plt.subplots(1, 2, figsize=(16, 10))

        # Setup axes
        for ax in (self.ax_traj, self.ax_prob):
            ax.set_xlim(self.bounds["min_lon"], self.bounds["max_lon"])
            ax.set_ylim(self.bounds["min_lat"], self.bounds["max_lat"])
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            # Add coastline
            if HAS_CARTOPY and self.land_geometry is not None:
                land_feature = cfeature.ShapelyFeature(
                    [self.land_geometry],
                    ccrs.PlateCarree(),
                    facecolor="lightgray",
                    edgecolor="black",
                    alpha=0.5,
                )
                ax.add_feature(land_feature, zorder=2)

        # Slider
        ax_slider = self.fig.add_axes([0.2, 0.25, 0.6, 0.03])
        self.slider = Slider(
            ax_slider,
            "Time Step",
            0,
            self.total_time_steps - 1,
            valinit=0,
            valstep=1,
            color="steelblue",
        )
        self.slider.on_changed(self._on_slider_change)

        # Info text
        self.info_text = self.fig.text(0.05, 0.15, "", fontsize=10, family="monospace")
        self.fig.suptitle("Search and Rescue Simulation Visualization", fontsize=14, fontweight="bold")

    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_time_step = int(self.slider.val)
        if self.current_time_step >= len(self.trajectories):
            self.current_time_step = len(self.trajectories) - 1
        self._update_display()

    def _update_display(self):
        """Update visualization for current time step."""
        self.ax_traj.clear()
        self.ax_prob.clear()

        # Redraw axes setup
        for ax in (self.ax_traj, self.ax_prob):
            ax.set_xlim(self.bounds["min_lon"], self.bounds["max_lon"])
            ax.set_ylim(self.bounds["min_lat"], self.bounds["max_lat"])
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            if HAS_CARTOPY and self.land_geometry is not None:
                land_feature = cfeature.ShapelyFeature(
                    [self.land_geometry],
                    ccrs.PlateCarree(),
                    facecolor="lightgray",
                    edgecolor="black",
                    alpha=0.5,
                )
                ax.add_feature(land_feature, zorder=2)

        # Left panel: Trajectories
        self._draw_trajectories()

        # Right panel: Probability surface + search
        self._draw_probability_and_search()

        # Update info text
        self._update_info_text()

        self.fig.canvas.draw_idle()

    def _draw_trajectories(self):
        """Draw particle trajectories up to current time step."""
        for i in range(self.n_particles):
            traj_lon = self.trajectories[: self.current_time_step + 1, i, 0]
            traj_lat = self.trajectories[: self.current_time_step + 1, i, 1]

            # Filter out NaNs
            mask = ~np.isnan(traj_lon) & ~np.isnan(traj_lat)
            if not np.any(mask):
                continue

            traj_lon = traj_lon[mask]
            traj_lat = traj_lat[mask]

            # Plot trajectory line
            self.ax_traj.plot(traj_lon, traj_lat, "b-", alpha=0.2, linewidth=0.5, zorder=1)

            # Plot current position
            self.ax_traj.scatter(traj_lon[-1], traj_lat[-1], c="red", s=20, edgecolors="darkred", linewidth=0.5, zorder=3)

        time_label = "(Drift only - search not started)" if self.current_time_step < self.start_step else "(Search in progress)"
        self.ax_traj.set_title(f"Particle Trajectories - Step {self.current_time_step}\n{time_label}")

    def _draw_probability_and_search(self):
        """Draw probability surface with search overlay."""
        prob = self.probability_surfaces[self.current_time_step]

        if prob is not None:
            # Draw probability surface
            masked_prob = np.ma.masked_where(prob == 0.0, prob)
            lat_centers = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])
            lon_centers = 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])
            lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

            self.ax_prob.pcolormesh(
                lon_grid,
                lat_grid,
                masked_prob,
                cmap="YlOrRd",
                shading="auto",
                alpha=0.7,
                zorder=1,
            )

        # Only show search visualization if search has started
        if self.current_time_step >= self.start_step:
            search_idx = self.current_time_step - self.start_step
            if search_idx < len(self.search_results):
                search_result = self.search_results[search_idx]
                
                # Only show argmax star on the first search step
                if search_idx == 0:
                    start_lat = search_result["search_start_lat"]
                    start_lon = search_result["search_start_lon"]
                    self.ax_prob.scatter(start_lon, start_lat, c="green", s=100, marker="*", edgecolors="darkgreen", linewidth=1, zorder=4, label="Search Start (Argmax)")

                # Draw searched cells
                cells_searched = search_result["cells_searched_list"]
                if cells_searched:
                    for lat_idx, lon_idx in cells_searched:
                        lat_min = self.lat_edges[lat_idx]
                        lat_max = self.lat_edges[lat_idx + 1]
                        lon_min = self.lon_edges[lon_idx]
                        lon_max = self.lon_edges[lon_idx + 1]

                        rect = Rectangle(
                            (lon_min, lat_min),
                            lon_max - lon_min,
                            lat_max - lat_min,
                            linewidth=0.5,
                            edgecolor="blue",
                            facecolor="none",
                            alpha=0.6,
                            zorder=2,
                        )
                        self.ax_prob.add_patch(rect)

        self.ax_prob.set_title(f"Probability Surface + Search - Step {self.current_time_step}")
        self.ax_prob.legend(loc="upper right")

    def _update_info_text(self):
        """Update info text with current metrics."""
        if self.current_time_step < self.start_step:
            info_str = f"Time Step: {self.current_time_step}/{self.total_time_steps - 1}\n"
            info_str += f"Status: Drift only (search begins at step {self.start_step})\n"
            info_str += f"Particles: {self.n_particles}"
            self.info_text.set_text(info_str)
            return
        
        search_idx = self.current_time_step - self.start_step
        if search_idx >= len(self.search_results):
            return
        
        search_result = self.search_results[search_idx]

        # Compute cumulative unique cells searched
        unique_cells = set()
        for sr in self.search_results[:self.current_time_step+1]:
            for cell in sr['cells_searched_list']:
                unique_cells.add(tuple(cell))

        info_str = f"Algorithm: {self.metadata['algorithm']}\n"
        info_str += f"Search Depth: {self.metadata['search_depth']}\n"
        info_str += f"Particles: {self.n_particles}\n"
        info_str += f"\n"
        info_str += f"Time Step: {self.current_time_step}/{self.total_time_steps - 1}\n"
        info_str += f"Time Label: {search_result['time_label']}\n"
        info_str += f"\n"
        info_str += f"Cells Searched This Step: {search_result['cells_searched']}\n"
        info_str += f"Probability Covered This Step: {search_result['probability_covered']:.6f}\n"
        info_str += f"\n"
        info_str += f"Cumulative Stats:\n"
        info_str += f"Total Cells Searched: {len(set(tuple(c) for result in self.search_results for c in result.get('cells_searched_list', [])))}\n"
        info_str += f"Total Probability: {sum(r['probability_covered'] for r in self.search_results):.6f}\n"
        self.info_text.set_text(info_str)

    def show(self):
        """Display the visualization."""
        self._update_display()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_search.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    visualizer = SearchVisualizer(json_file)
    visualizer.show()
