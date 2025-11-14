import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from typing import Tuple

try:
    from .grid_world import GridWorld
except ImportError:
    from grid_world import GridWorld


class Currents:
    """Load forecast currents and interpolate to GridWorld."""
    
    def __init__(self, forecast_file: str):
        """
        Initialize with forecast NetCDF file.
        
        Args:
            forecast_file: Path to forecast.nc file
        """
        self.forecast_file = forecast_file
        self.ds = xr.open_dataset(forecast_file)
        self._extract_forecast_data()
    
    def _extract_forecast_data(self):
        """Extract u, v, lat, lon, and time from forecast file."""
        # Get surface layer (siglay=0 if available)
        u_key, v_key = 'u', 'v'
        if 'siglay' in self.ds.dims:
            self.u = self.ds[u_key].isel(siglay=0).values
            self.v = self.ds[v_key].isel(siglay=0).values
        else:
            self.u = self.ds[u_key].values
            self.v = self.ds[v_key].values
        
        # Get coordinates (stations or fields format)
        lon_key = 'lon' if 'lon' in self.ds else 'lonc'
        lat_key = 'lat' if 'lat' in self.ds else 'latc'
        self.station_lon = np.where(self.ds[lon_key].values > 180, 
                                     self.ds[lon_key].values - 360, 
                                     self.ds[lon_key].values)
        self.station_lat = self.ds[lat_key].values
        
        # Get time
        time_key = 'Times' if 'Times' in self.ds else 'time'
        self.time = self.ds[time_key].values
        self.n_times = self.u.shape[0]
    
    def _calculate_speed_direction(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate speed (m/s) and direction (degrees) from u, v components."""
        speed = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u) * 180 / np.pi  # Convert to degrees
        direction = (direction + 360) % 360  # Normalize to 0-360
        return speed, direction
    
    def populate_gridworld(self, grid: GridWorld) -> GridWorld:
        """Interpolate forecast currents to GridWorld."""
        grid_lat, grid_lon = grid.get_grid_points()
        bounds = grid.get_bounds()
        
        # Filter stations within bounds
        mask = (
            (self.station_lat >= bounds['min_lat']) & 
            (self.station_lat <= bounds['max_lat']) &
            (self.station_lon >= bounds['min_lon']) & 
            (self.station_lon <= bounds['max_lon'])
        )
        if not np.any(mask):
            raise ValueError("No forecast stations within grid bounds")
        
        # Get bounded station data
        lon, lat = self.station_lon[mask], self.station_lat[mask]
        u_bounded, v_bounded = self.u[:, mask], self.v[:, mask]
        
        # Initialize storage
        grid.current_speed = {}
        grid.current_direction = {}
        
        # Interpolate for each time step
        for t in range(self.n_times):
            speed_t, direction_t = self._calculate_speed_direction(u_bounded[t], v_bounded[t])
            grid.current_speed[t] = griddata((lon, lat), speed_t, (grid_lon, grid_lat), method='nearest')
            grid.current_direction[t] = griddata((lon, lat), direction_t, (grid_lon, grid_lat), method='nearest')
        
        return grid
    
    def get_time_label(self, time_idx: int) -> str:
        """Get formatted time label for a time step."""
        time_val = self.time[time_idx]
        if isinstance(time_val, bytes):
            return time_val.decode('utf-8')[:16].replace('T', ' ')
        return str(time_val)[:16].replace('T', ' ')
    
    def get_info(self, grid: GridWorld = None) -> dict:
        """Return info about forecast data."""
        n_stations = len(self.station_lat)
        if grid is not None:
            bounds = grid.get_bounds()
            mask = (
                (self.station_lat >= bounds['min_lat']) & 
                (self.station_lat <= bounds['max_lat']) &
                (self.station_lon >= bounds['min_lon']) & 
                (self.station_lon <= bounds['max_lon'])
            )
            n_stations = np.sum(mask)
        return {
            'file': self.forecast_file,
            'n_times': self.n_times,
            'n_stations': n_stations,
            'time_range': (self.get_time_label(0), self.get_time_label(self.n_times - 1))
        }


if __name__ == "__main__":
    from grid_world import GridWorld
    
    # Example usage
    forecast_file = "data/sfbofs.t15z.20251105.stations.forecast.nc"
    
    # Create GridWorld
    min_lat, max_lat = 37.746386588546684, 37.84169189475321
    min_lon, max_lon = -122.63743707976295, -122.42795905807327
    grid = GridWorld(min_lat, max_lat, min_lon, max_lon, cell_size_m=100)
    
    # Load currents and populate grid
    currents = Currents(forecast_file)
    print("Forecast info (all stations):", currents.get_info())
    print("Forecast info (bounded stations):", currents.get_info(grid))
    
    grid = currents.populate_gridworld(grid)
    
    print(f"\nGrid populated with currents")
    print(f"Current speed at t=0 - Mean: {grid.current_speed[0].mean():.3f} m/s")
    print(f"Current direction at t=0 - Mean: {grid.current_direction[0].mean():.1f}°")
