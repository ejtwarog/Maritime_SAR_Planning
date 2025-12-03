import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from grid_world import GridWorld
from currents import Currents

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def _setup_map(figsize, use_map):
    """Create figure with optional map projection."""
    if use_map and HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines(resolution='10m', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _set_extent(ax, bounds, use_map):
    """Set map extent."""
    if use_map and HAS_CARTOPY:
        ax.set_extent([bounds['min_lon'], bounds['max_lon'], bounds['min_lat'], bounds['max_lat']])
    else:
        ax.set_xlim(bounds['min_lon'], bounds['max_lon'])
        ax.set_ylim(bounds['min_lat'], bounds['max_lat'])


def _scale_uv_to_degrees(u: np.ndarray, v: np.ndarray, lat: np.ndarray) -> tuple:
    """Scale u, v components from m/s to degrees/hour for visualization."""
    lat_mean = np.mean(lat)
    m_per_deg_lon = 111320 * np.cos(np.radians(lat_mean))
    m_per_deg_lat = 111320
    u_deg, v_deg = u / m_per_deg_lon, v / m_per_deg_lat
    return u_deg * 3600, v_deg * 3600


def visualize_station_vectors(grid: GridWorld, currents: Currents, time_step: int = 0, figsize=(12, 8), use_map: bool = True):
    """Visualize current vectors at forecast stations within grid bounds."""
    fig, ax = _setup_map(figsize, use_map)
    
    bounds = grid.get_bounds()
    mask = (
        (currents.station_lat >= bounds['min_lat']) & 
        (currents.station_lat <= bounds['max_lat']) &
        (currents.station_lon >= bounds['min_lon']) & 
        (currents.station_lon <= bounds['max_lon'])
    )
    
    lon, lat = currents.station_lon[mask], currents.station_lat[mask]
    u, v = currents.u[time_step, mask], currents.v[time_step, mask]
    speed = np.sqrt(u**2 + v**2)
    
    # Plot stations
    scatter = ax.scatter(lon, lat, c=speed, s=150, cmap='viridis', 
                        edgecolors='white', linewidth=1.5, alpha=0.95, zorder=3)
    plt.colorbar(scatter, ax=ax, label='Speed (m/s)').ax.tick_params(labelsize=10)
    
    # Scale for 1-hour drift visualization
    u_scaled, v_scaled = _scale_uv_to_degrees(u, v, lat)
    
    # Plot vectors with outline
    ax.quiver(lon, lat, u_scaled, v_scaled, scale=1, scale_units='xy', angles='xy',
             width=0.008, color='black', alpha=0.5, zorder=3.5)
    ax.quiver(lon, lat, u_scaled, v_scaled, scale=1, scale_units='xy', angles='xy',
             width=0.004, color='red', alpha=0.95, zorder=4)
    
    _set_extent(ax, bounds, use_map)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Station Vectors (1-hour drift) - {currents.get_time_label(time_step)}\n({np.sum(mask)} stations)')
    plt.tight_layout()
    return fig, ax


def visualize_vector_field(grid: GridWorld, currents: Currents, time_step: int = 0, figsize=(12, 8), use_map: bool = True):
    """Visualize interpolated current vectors on GridWorld points."""
    fig, ax = _setup_map(figsize, use_map)
    
    grid_lat, grid_lon = grid.get_grid_points()
    grid_shape = grid.get_grid_shape()
    
    # Interpolate station data to grid
    u_grid, v_grid = currents.interpolate_to_grid(grid_lon, grid_lat, time_step, method='linear')
    speed_grid = np.sqrt(u_grid**2 + v_grid**2)
    
    # Reshape for plotting
    u_2d, v_2d = u_grid.reshape(grid_shape), v_grid.reshape(grid_shape)
    speed_2d = speed_grid.reshape(grid_shape)
    lat_2d, lon_2d = grid_lat.reshape(grid_shape), grid_lon.reshape(grid_shape)
    
    # Plot speed background
    im = ax.contourf(lon_2d, lat_2d, speed_2d, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Speed (m/s)')
    
    # Scale for visualization
    u_scaled, v_scaled = _scale_uv_to_degrees(u_2d, v_2d, grid_lat)
    
    # Plot vectors with outline (subsample for clarity)
    subsample = 3
    ax.quiver(lon_2d[::subsample, ::subsample], lat_2d[::subsample, ::subsample],
              u_scaled[::subsample, ::subsample], v_scaled[::subsample, ::subsample],
              scale=1, scale_units='xy', angles='xy', width=0.008, color='black', alpha=0.5, zorder=3.5)
    ax.quiver(lon_2d[::subsample, ::subsample], lat_2d[::subsample, ::subsample],
              u_scaled[::subsample, ::subsample], v_scaled[::subsample, ::subsample],
              scale=1, scale_units='xy', angles='xy', width=0.004, color='red', alpha=0.95, zorder=4)
    
    bounds = grid.get_bounds()
    _set_extent(ax, bounds, use_map)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Interpolated Vector Field (1-hour drift) - {currents.get_time_label(time_step)}\n({grid.n_cells} grid points)')
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    forecast_file = "data/sfbofs.t15z.20251105.stations.forecast.nc"
    bounds = (37.7, 37.9, -122.7, -122.4)
    
    print("Creating GridWorld...")
    grid = GridWorld(*bounds, cell_size_m=500)
    
    print("Loading currents...")
    currents = Currents(forecast_file)
    grid = currents.populate_gridworld(grid)
    
    # Generate visualizations for next 5 time steps
    for t in range(5):
        print(f"\nVisualizing time step {t}...")
        
        fig, ax = visualize_station_vectors(grid, currents, time_step=t)
        plt.savefig(f'stations_t{t}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: stations_t{t}.png")
        plt.close(fig)
        
        fig, ax = visualize_vector_field(grid, currents, time_step=t)
        plt.savefig(f'vector_field_t{t}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: vector_field_t{t}.png")
        plt.close(fig)
