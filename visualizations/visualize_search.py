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
        
        # Compute global vmax for consistent color scaling across all time steps
        self.global_vmax = 0.0
        for prob in self.probability_surfaces:
            if prob is not None:
                p = np.array(prob)
                self.global_vmax = max(self.global_vmax, np.nanmax(p))

    def _setup_figure(self):
        """Setup matplotlib figure with subplots."""
        if HAS_CARTOPY:
            self.fig = plt.figure(figsize=(12, 9))
            self.ax_prob = self.fig.add_axes([0.1, 0.45, 0.8, 0.5], projection=ccrs.PlateCarree())
        else:
            self.fig, self.ax_prob = plt.subplots(1, 1, figsize=(12, 9))

        # Setup axes
        self.ax_prob.set_xlim(self.bounds["min_lon"], self.bounds["max_lon"])
        self.ax_prob.set_ylim(self.bounds["min_lat"], self.bounds["max_lat"])
        self.ax_prob.set_xlabel("Longitude")
        self.ax_prob.set_ylabel("Latitude")

        # Add coastline
        if HAS_CARTOPY and self.land_geometry is not None:
            land_feature = cfeature.ShapelyFeature(
                [self.land_geometry],
                ccrs.PlateCarree(),
                facecolor="lightgray",
                edgecolor="black",
                alpha=0.5,
            )
            self.ax_prob.add_feature(land_feature, zorder=2)

        # Slider
        ax_slider = self.fig.add_axes([0.2, 0.38, 0.6, 0.03])
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

        # Info text - moved below slider with more space
        self.info_text = self.fig.text(0.1, 0.02, "", fontsize=11, family="monospace", verticalalignment="bottom")
        self.fig.suptitle("Search and Rescue Simulation Visualization", fontsize=14, fontweight="bold")

    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_time_step = int(self.slider.val)
        if self.current_time_step >= len(self.trajectories):
            self.current_time_step = len(self.trajectories) - 1
        self._update_display()

    def _update_display(self):
        """Update visualization for current time step."""
        self.ax_prob.clear()

        # Redraw axes setup
        self.ax_prob.set_xlim(self.bounds["min_lon"], self.bounds["max_lon"])
        self.ax_prob.set_ylim(self.bounds["min_lat"], self.bounds["max_lat"])
        self.ax_prob.set_xlabel("Longitude")
        self.ax_prob.set_ylabel("Latitude")

        if HAS_CARTOPY and self.land_geometry is not None:
            land_feature = cfeature.ShapelyFeature(
                [self.land_geometry],
                ccrs.PlateCarree(),
                facecolor="lightgray",
                edgecolor="black",
                alpha=0.5,
            )
            self.ax_prob.add_feature(land_feature, zorder=2)

        # Draw probability surface + search
        self._draw_probability_and_search()

        # Update info text
        self._update_info_text()

        self.fig.canvas.draw_idle()

    def _draw_probability_and_search(self):
        """Draw probability surface with search overlay."""
        prob = self.probability_surfaces[self.current_time_step]

        if prob is not None:
            # Draw probability surface
            masked_prob = np.ma.masked_where(prob == 0.0, prob)
            lat_centers = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])
            lon_centers = 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])
            lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

            # Use global vmax for consistent color scaling across all time steps
            # This makes it visually obvious when probability decreases after search
            vmax = self.global_vmax if self.global_vmax > 0 else 0.01
            
            self.ax_prob.pcolormesh(
                lon_grid,
                lat_grid,
                masked_prob,
                cmap="YlOrRd",
                shading="auto",
                alpha=0.7,
                vmin=0,
                vmax=vmax,
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

                # Draw searched cells and path connections
                cells_searched = search_result["cells_searched_list"]
                platform_metrics = search_result.get("platform_metrics", {})
                
                if cells_searched:
                    if platform_metrics:
                        # Dual-platform: draw each platform's cells with different colors and connecting paths
                        platform_colors = {
                            "crewed": "blue",
                            "uncrewed": "red",
                        }
                        legend_added = set()
                        
                        for platform_name, metrics in platform_metrics.items():
                            cells_list = metrics.get("cells_list", [])
                            color = platform_colors.get(platform_name, "gray")
                            
                            # Draw connecting path for this platform
                            if len(cells_list) > 1:
                                path_lons = []
                                path_lats = []
                                for lat_idx, lon_idx in cells_list:
                                    lat_center = 0.5 * (self.lat_edges[lat_idx] + self.lat_edges[lat_idx + 1])
                                    lon_center = 0.5 * (self.lon_edges[lon_idx] + self.lon_edges[lon_idx + 1])
                                    path_lons.append(lon_center)
                                    path_lats.append(lat_center)
                                self.ax_prob.plot(path_lons, path_lats, color=color, linewidth=1.5, alpha=0.5, zorder=2)
                            
                            # Draw cell rectangles
                            for lat_idx, lon_idx in cells_list:
                                lat_min = self.lat_edges[lat_idx]
                                lat_max = self.lat_edges[lat_idx + 1]
                                lon_min = self.lon_edges[lon_idx]
                                lon_max = self.lon_edges[lon_idx + 1]

                                rect = Rectangle(
                                    (lon_min, lat_min),
                                    lon_max - lon_min,
                                    lat_max - lat_min,
                                    linewidth=0.5,
                                    edgecolor=color,
                                    facecolor="none",
                                    alpha=0.6,
                                    zorder=2,
                                )
                                self.ax_prob.add_patch(rect)
                            
                            # Add legend entry for this platform (only once)
                            if platform_name not in legend_added:
                                self.ax_prob.plot([], [], color=color, linewidth=2, label=f"{platform_name.capitalize()} Platform")
                                legend_added.add(platform_name)
                    else:
                        # Single platform: draw connecting path
                        path_lons = []
                        path_lats = []
                        
                        for lat_idx, lon_idx in cells_searched:
                            # Get cell center
                            lat_center = 0.5 * (self.lat_edges[lat_idx] + self.lat_edges[lat_idx + 1])
                            lon_center = 0.5 * (self.lon_edges[lon_idx] + self.lon_edges[lon_idx + 1])
                            path_lons.append(lon_center)
                            path_lats.append(lat_center)
                        
                        # Draw path line
                        if len(path_lons) > 1:
                            self.ax_prob.plot(path_lons, path_lats, "b-", linewidth=1.5, alpha=0.5, zorder=2, label="Search Path")
                        
                        # Draw cell rectangles
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
        info_str += f"Initial Particles: {self.n_particles}\n"
        info_str += f"\n"
        info_str += f"Time Step: {self.current_time_step}/{self.total_time_steps - 1}\n"
        info_str += f"Time Label: {search_result['time_label']}\n"
        info_str += f"\n"
        info_str += f"This Step:\n"
        info_str += f"  Cells Searched: {search_result['cells_searched']}\n"
        info_str += f"  Objects Removed: {search_result.get('objects_removed', 0)}\n"
        info_str += f"  Probability Covered: {search_result['probability_covered']:.6f}\n"
        info_str += f"\n"
        
        # Compute cumulative stats up to current step
        total_cells = len(set(tuple(c) for result in self.search_results[:search_idx+1] for c in result.get('cells_searched_list', [])))
        total_removed = sum(r.get('objects_removed', 0) for r in self.search_results[:search_idx+1])
        removal_rate = (total_removed / self.n_particles * 100) if self.n_particles > 0 else 0
        
        info_str += f"Cumulative Stats (through step {self.current_time_step}):\n"
        info_str += f"  Total Cells Searched: {total_cells}\n"
        info_str += f"  Total Objects Removed: {total_removed}/{self.n_particles}\n"
        info_str += f"  Removal Rate: {removal_rate:.2f}%\n"
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
