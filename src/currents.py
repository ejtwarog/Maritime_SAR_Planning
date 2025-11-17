import json

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from typing import Tuple

from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import vectorized

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
        self.land_geometry = None
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
    
    def load_land_geometry(self, geojson_path: str) -> None:
        """Load land geometry from a GeoJSON file and cache as a unary union geometry."""
        if self.land_geometry is not None:
            return

        try:
            with open(geojson_path, "r") as f:
                data = json.load(f)
        except Exception:
            self.land_geometry = None
            return

        features = data.get("features", [])
        if not features:
            self.land_geometry = None
            return

        geometries = []
        for feature in features:
            geom_data = feature.get("geometry")
            if geom_data is None:
                continue
            geometries.append(shape(geom_data))

        if not geometries:
            self.land_geometry = None
            return

        self.land_geometry = unary_union(geometries)

    def mask_on_land(self, lon: np.ndarray, lat: np.ndarray,
                     u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-out u, v where (lon, lat) fall on cached land geometry, if available."""
        if self.land_geometry is None:
            return u, v

        on_land = vectorized.contains(self.land_geometry, lon, lat)
        u = u.copy()
        v = v.copy()
        u[on_land] = 0.0
        v[on_land] = 0.0
        return u, v

    def is_on_land(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating which positions lie on cached land geometry."""
        if self.land_geometry is None:
            return np.zeros_like(lon, dtype=bool)
        return vectorized.contains(self.land_geometry, lon, lat)
    
    def interpolate_to_grid(self, grid_lon: np.ndarray, grid_lat: np.ndarray, 
                           time_step: int, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate u, v components to grid points at a specific time step.
        
        Args:
            grid_lon: Grid longitude points (1D or flattened)
            grid_lat: Grid latitude points (1D or flattened)
            time_step: Time index to interpolate
            method: Interpolation method ('linear', 'nearest', 'cubic')
        
        Returns:
            Tuple of (u_grid, v_grid) interpolated to grid points
        """
        bounds = {'min_lat': grid_lat.min(), 'max_lat': grid_lat.max(),
                  'min_lon': grid_lon.min(), 'max_lon': grid_lon.max()}
        
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
        station_lon, station_lat = self.station_lon[mask], self.station_lat[mask]
        u_station = self.u[time_step, mask]
        v_station = self.v[time_step, mask]
        
        # Interpolate to grid
        u_grid = griddata((station_lon, station_lat), u_station, (grid_lon, grid_lat), method=method)
        v_grid = griddata((station_lon, station_lat), v_station, (grid_lon, grid_lat), method=method)
        
        return u_grid, v_grid
    
    def interpolate_speed_direction_to_grid(self, grid_lon: np.ndarray, grid_lat: np.ndarray,
                                           time_step: int, method: str = 'nearest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate speed and direction to grid points at a specific time step.
        
        Args:
            grid_lon: Grid longitude points (1D or flattened)
            grid_lat: Grid latitude points (1D or flattened)
            time_step: Time index to interpolate
            method: Interpolation method ('linear', 'nearest', 'cubic')
        
        Returns:
            Tuple of (speed_grid, direction_grid) interpolated to grid points
        """
        bounds = {'min_lat': grid_lat.min(), 'max_lat': grid_lat.max(),
                  'min_lon': grid_lon.min(), 'max_lon': grid_lon.max()}
        
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
        station_lon, station_lat = self.station_lon[mask], self.station_lat[mask]
        u_station = self.u[time_step, mask]
        v_station = self.v[time_step, mask]
        
        # Calculate speed and direction at stations
        speed_station, direction_station = self._calculate_speed_direction(u_station, v_station)
        
        # Interpolate to grid
        speed_grid = griddata((station_lon, station_lat), speed_station, (grid_lon, grid_lat), method=method)
        direction_grid = griddata((station_lon, station_lat), direction_station, (grid_lon, grid_lat), method=method)
        
        return speed_grid, direction_grid
    
    def populate_gridworld(self, grid: GridWorld) -> GridWorld:
        """Interpolate forecast currents to GridWorld for all time steps."""
        grid_lat, grid_lon = grid.get_grid_points()
        
        # Initialize storage
        grid.current_speed = {}
        grid.current_direction = {}
        
        # Interpolate for each time step
        for t in range(self.n_times):
            grid.current_speed[t], grid.current_direction[t] = \
                self.interpolate_speed_direction_to_grid(grid_lon, grid_lat, t, method='nearest')
        
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
