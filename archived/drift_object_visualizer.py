import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import griddata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from grid_world import GridWorld
from currents import Currents
from drift_object import DriftObjectCollection

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


class DriftDashboard:
    """Interactive dashboard for visualizing particle drift trajectories."""
    
    def __init__(self, grid: GridWorld, currents: Currents, n_particles: int = 5, 
                 max_time_steps: int = 50, figsize=(14, 10),
                 init_center: tuple | None = None, init_sigma_m: float = 250.0):
        """
        Initialize drift dashboard.
        
        Args:
            grid: GridWorld object
            currents: Currents object
            n_particles: Number of particles to sample (default: 1)
            max_time_steps: Maximum time steps to simulate
            figsize: Figure size
        """
        self.grid = grid
        self.currents = currents
        self.n_particles = n_particles
        self.max_time_steps = min(max_time_steps, currents.n_times - 1)
        self.dt = 360  # Time step in seconds (6 minutes)
        self.drift_objects = DriftObjectCollection()

        # Initial distribution parameters
        if init_center is None:
            # Default to grid center if not provided
            self.center_lat = 0.5 * (grid.get_bounds()['min_lat'] + grid.get_bounds()['max_lat'])
            self.center_lon = 0.5 * (grid.get_bounds()['min_lon'] + grid.get_bounds()['max_lon'])
        else:
            self.center_lat, self.center_lon = init_center
        self.init_sigma_m = init_sigma_m
        
        # Get grid info
        self.grid_lat, self.grid_lon = grid.get_grid_points()
        self.grid_shape = grid.get_grid_shape()
        self.bounds = grid.get_bounds()

        # Precompute grid cell edges for probability surface
        lat_2d = self.grid_lat.reshape(self.grid_shape)
        lon_2d = self.grid_lon.reshape(self.grid_shape)
        lat_axis = lat_2d[:, 0]
        lon_axis = lon_2d[0, :]

        lat_edges = np.empty(lat_axis.size + 1)
        lon_edges = np.empty(lon_axis.size + 1)

        lat_edges[1:-1] = 0.5 * (lat_axis[:-1] + lat_axis[1:])
        lat_edges[0] = lat_axis[0] - 0.5 * (lat_axis[1] - lat_axis[0])
        lat_edges[-1] = lat_axis[-1] + 0.5 * (lat_axis[-1] - lat_axis[-2])

        lon_edges[1:-1] = 0.5 * (lon_axis[:-1] + lon_axis[1:])
        lon_edges[0] = lon_axis[0] - 0.5 * (lon_axis[1] - lon_axis[0])
        lon_edges[-1] = lon_axis[-1] + 0.5 * (lon_axis[-1] - lon_axis[-2])

        self.lat_edges = lat_edges
        self.lon_edges = lon_edges
        
        # Initialize particle positions and trajectories
        self.particles_lon = None
        self.particles_lat = None
        self.trajectories = None
        self.current_time_step = 0
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        self.currents.load_land_geometry("data/California.geojson")

        # Axes: left for trajectories, right for probability surface
        self.ax_traj = None
        self.ax_prob = None
        self._setup_map()
        self._setup_controls()
        self._resample_particles()
        self._update_display()
    
    def _setup_map(self):
        """Set up the map axes for trajectories and probability surface."""
        if HAS_CARTOPY:
            self.ax_traj = self.fig.add_axes([0.07, 0.35, 0.4, 0.6], 
                                             projection=ccrs.PlateCarree())
            self.ax_prob = self.fig.add_axes([0.53, 0.35, 0.4, 0.6], 
                                             projection=ccrs.PlateCarree())

            # Land overlay on both axes if available
            if getattr(self.currents, "land_geometry", None) is not None:
                land_feature = cfeature.ShapelyFeature(
                    [self.currents.land_geometry],
                    ccrs.PlateCarree(),
                    facecolor='lightgray',
                    edgecolor='black',
                    alpha=0.5,
                )
                self.ax_traj.add_feature(land_feature)
                self.ax_prob.add_feature(land_feature)
        else:
            self.ax_traj = self.fig.add_axes([0.07, 0.35, 0.4, 0.6])
            self.ax_prob = self.fig.add_axes([0.53, 0.35, 0.4, 0.6])
        
        # Set extent and labels
        for ax in (self.ax_traj, self.ax_prob):
            ax.set_xlim(self.bounds['min_lon'], self.bounds['max_lon'])
            ax.set_ylim(self.bounds['min_lat'], self.bounds['max_lat'])
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
    
    def _setup_controls(self):
        """Set up slider and button controls."""
        # Time slider
        ax_slider = self.fig.add_axes([0.2, 0.25, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Time Step', 0, self.max_time_steps - 1, 
                            valinit=0, valstep=1, color='steelblue')
        self.slider.on_changed(self._on_slider_change)
        
        # Resample button
        ax_button = self.fig.add_axes([0.75, 0.15, 0.1, 0.04])
        self.btn_resample = Button(ax_button, 'Resample', color='lightcoral', hovercolor='red')
        self.btn_resample.on_clicked(self._on_resample)
        
        # Info text
        self.info_text = self.fig.text(0.1, 0.05, '', fontsize=10, family='monospace')
    
    def _resample_particles(self):
        """Randomly sample particles at time zero within grid bounds."""
        # Reset drift objects
        self.drift_objects = DriftObjectCollection()

        # Gaussian spherical distribution in meters around specified center
        center_lat = self.center_lat
        center_lon = self.center_lon
        sigma_m = self.init_sigma_m

        bounds = self.bounds
        lon = np.zeros(self.n_particles)
        lat = np.zeros(self.n_particles)

        # Precompute metric conversion at center latitude
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

        remaining = np.ones(self.n_particles, dtype=bool)
        max_iterations = 50
        iterations = 0
        while np.any(remaining) and iterations < max_iterations:
            iterations += 1

            n_remain = remaining.sum()
            # Sample dx, dy ~ N(0, sigma^2) in meters
            dx = np.random.normal(0.0, sigma_m, n_remain)
            dy = np.random.normal(0.0, sigma_m, n_remain)

            dlon = dx / m_per_deg_lon
            dlat = dy / m_per_deg_lat

            lon_candidate = center_lon + dlon
            lat_candidate = center_lat + dlat

            # Check bounds
            in_bounds = (
                (lat_candidate >= bounds['min_lat']) &
                (lat_candidate <= bounds['max_lat']) &
                (lon_candidate >= bounds['min_lon']) &
                (lon_candidate <= bounds['max_lon'])
            )

            # Check land mask
            on_land = self.currents.is_on_land(lon_candidate, lat_candidate)
            valid = in_bounds & (~on_land)

            # Assign valid samples into lon/lat arrays
            idx_remain = np.where(remaining)[0]
            assigned_idx = idx_remain[valid]
            lon[assigned_idx] = lon_candidate[valid]
            lat[assigned_idx] = lat_candidate[valid]

            # Update remaining mask
            remaining[assigned_idx] = False

        # Fallback: if we still have remaining points, sample uniformly in bounds (without land check)
        if np.any(remaining):
            idx_remain = np.where(remaining)[0]
            lon[idx_remain] = np.random.uniform(bounds['min_lon'], bounds['max_lon'], len(idx_remain))
            lat[idx_remain] = np.random.uniform(bounds['min_lat'], bounds['max_lat'], len(idx_remain))

        # Uniformly distribute particle releases across first 3 hours (30 time steps)
        total = self.n_particles
        drop_window_hours = 3.0
        drop_window_steps = int(drop_window_hours * 3600 / self.dt)  # Convert hours to time steps
        n_release_steps = min(drop_window_steps, max(1, self.max_time_steps - 1))
        per_step = total // n_release_steps
        extra = total % n_release_steps

        idx = 0
        # Distribute particles uniformly across first n_release_steps time steps
        for step in range(n_release_steps):
            count_this_step = per_step + (1 if step < extra else 0)
            for _ in range(count_this_step):
                if idx >= total:
                    break
                self.drift_objects.add_object(float(lon[idx]), float(lat[idx]), created_time_idx=step)
                idx += 1

        # Initialize trajectories (time_steps x particles x 2) with NaNs before release
        self.trajectories = np.full((self.max_time_steps, self.n_particles, 2), np.nan)
        positions = self.drift_objects.get_positions()
        created_times = np.array([o.created_time_idx for o in self.drift_objects.objects])
        for i in range(self.n_particles):
            if created_times[i] == 0:
                self.trajectories[0, i, 0] = positions[i, 0]
                self.trajectories[0, i, 1] = positions[i, 1]

        # Advect particles through time steps
        self._advect_particles()
    
    def _advect_particles(self):
        """Advect particles through the vector field."""
        for t in range(1, self.max_time_steps):
            # Advance drift objects by one time step and record positions
            self.drift_objects.step(self.currents, time_idx=t, dt=self.dt)
            positions = self.drift_objects.get_positions()
            created_times = np.array([o.created_time_idx for o in self.drift_objects.objects])
            for i in range(self.n_particles):
                if created_times[i] <= t:
                    self.trajectories[t, i, 0] = positions[i, 0]
                    self.trajectories[t, i, 1] = positions[i, 1]
    
    def _update_display(self):
        """Update the visualization."""
        self.ax_traj.clear()
        self.ax_prob.clear()

        # Redraw maps and land overlay
        for ax in (self.ax_traj, self.ax_prob):
            if HAS_CARTOPY:
                if getattr(self.currents, "land_geometry", None) is not None:
                    land_feature = cfeature.ShapelyFeature(
                        [self.currents.land_geometry],
                        ccrs.PlateCarree(),
                        facecolor='lightgray',
                        edgecolor='black',
                        alpha=0.5,
                    )
                    ax.add_feature(land_feature)
            ax.set_xlim(self.bounds['min_lon'], self.bounds['max_lon'])
            ax.set_ylim(self.bounds['min_lat'], self.bounds['max_lat'])
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        # Probability surface over GridWorld cells from current positions (right panel)
        if self.trajectories is not None:
            current_lon = self.trajectories[self.current_time_step, :, 0]
            current_lat = self.trajectories[self.current_time_step, :, 1]

            hist, lat_edges, lon_edges = np.histogram2d(
                current_lat,
                current_lon,
                bins=[self.lat_edges, self.lon_edges],
            )

            if hist.sum() > 0:
                prob = hist / hist.sum()
                lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
                lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
                lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
                self.ax_prob.pcolormesh(
                    lon_grid,
                    lat_grid,
                    prob,
                    cmap="viridis",
                    shading="auto",
                    alpha=0.8,
                    zorder=1,
                )

            # Plot trajectories up to current time step (left panel)
            for i in range(self.n_particles):
                traj_lon = self.trajectories[:self.current_time_step + 1, i, 0]
                traj_lat = self.trajectories[:self.current_time_step + 1, i, 1]
                
                # Plot trajectory line
                self.ax_traj.plot(traj_lon, traj_lat, 'b-', alpha=0.3, linewidth=1, zorder=2)
                
                # Plot current particle position (small red dot)
                self.ax_traj.scatter(traj_lon[-1], traj_lat[-1], c='red', s=10, 
                                     edgecolors='darkred', linewidth=0.5, zorder=3)


        # Update titles and info
        time_label = self.currents.get_time_label(self.current_time_step)
        self.ax_traj.set_title(f'Drift Objects - Time Step {self.current_time_step}\n{time_label}')
        self.ax_prob.set_title('Probability Surface')
        self.ax_traj.legend(loc='upper right')
        
        # Update info text
        info_str = f"Particles: {self.n_particles} | Time Step: {self.current_time_step}/{self.max_time_steps - 1}"
        self.info_text.set_text(info_str)
        
        self.fig.canvas.draw_idle()
    
    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_time_step = int(self.slider.val)
        self._update_display()
    
    def _on_resample(self, event):
        """Handle resample button click."""
        self._resample_particles()
        self.slider.set_val(0)
        self._update_display()
    
    def show(self):
        """Display the dashboard."""
        plt.show()

    def export_probability_surface(self, filename: str) -> None:
        """Export probability surface over GridWorld cells for all time steps to JSON.

        The JSON structure is:
        {
            "lat_edges": [...],
            "lon_edges": [...],
            "time_steps": [
                {
                    "time_step": int,
                    "time_label": str,
                    "probability": [[...], ...]  # 2D list matching grid shape
                },
                ...
            ]
        }
        """
        # Ensure trajectories exist
        if self.trajectories is None:
            self._resample_particles()

        output = {
            "lat_edges": self.lat_edges.tolist(),
            "lon_edges": self.lon_edges.tolist(),
            "time_steps": [],
        }

        for t in range(self.max_time_steps):
            lon_t = self.trajectories[t, :, 0]
            lat_t = self.trajectories[t, :, 1]

            # Ignore particles not yet released (NaNs)
            mask = ~np.isnan(lon_t) & ~np.isnan(lat_t)
            if not np.any(mask):
                prob_list = None
            else:
                hist, lat_edges, lon_edges = np.histogram2d(
                    lat_t[mask],
                    lon_t[mask],
                    bins=[self.lat_edges, self.lon_edges],
                )
                if hist.sum() > 0:
                    prob = hist / hist.sum()
                    prob_list = prob.tolist()
                else:
                    prob_list = None

            output["time_steps"].append({
                "time_step": int(t),
                "time_label": self.currents.get_time_label(t),
                "probability": prob_list,
            })

        with open(filename, "w") as f:
            json.dump(output, f)

if __name__ == "__main__":
    forecast_file = "data/sfbofs.t15z.20251105.stations.forecast.nc"
    bounds = (37.68, 37.92, -122.75, -122.35)
    
    print("Creating GridWorld...")
    grid = GridWorld(*bounds, cell_size_m=250)
    
    print("Loading currents...")
    currents = Currents(forecast_file)
    
    print("Creating drift dashboard...")
    init_center = (37.781827, -122.541443)
    init_sigma_m = 250.0
    dashboard = DriftDashboard(
        grid,
        currents,
        n_particles=1000,
        max_time_steps=100,
        init_center=init_center,
        init_sigma_m=init_sigma_m,
    )
    dashboard.show()
    dashboard.export_probability_surface("probability_surface.json")
    plt.close('all')
