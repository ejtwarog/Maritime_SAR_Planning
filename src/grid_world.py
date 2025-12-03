import numpy as np
from typing import Tuple, List


class GridWorld:
    """Geographic grid sampled at regular intervals within lat/lon bounds."""
    
    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        cell_size_m: float = 100
    ):
        """Initialize grid within lat/lon bounds with given cell size (meters)."""
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.cell_size_m = cell_size_m
        
        self.earth_radius_m = 6371000
        self._calculate_grid_dimensions()
        self._generate_grid()
    
    def _calculate_grid_dimensions(self):
        """Calculate grid dimensions in cells."""
        lat_diff_m = self._lat_to_meters(self.max_lat - self.min_lat)
        lon_diff_m = self._lon_to_meters(self.max_lon - self.min_lon, self.min_lat)
        self.n_cells_lat = int(np.ceil(lat_diff_m / self.cell_size_m))
        self.n_cells_lon = int(np.ceil(lon_diff_m / self.cell_size_m))
        self.n_cells = self.n_cells_lat * self.n_cells_lon
    
    def _lat_to_meters(self, lat_diff: float) -> float:
        """Convert lat difference to meters."""
        return lat_diff * (self.earth_radius_m * np.pi / 180)
    
    def _lon_to_meters(self, lon_diff: float, lat: float) -> float:
        """Convert lon difference to meters at given latitude."""
        return lon_diff * (self.earth_radius_m * np.pi / 180) * np.cos(np.radians(lat))
    
    def _meters_to_lat(self, meters: float) -> float:
        """Convert meters to lat difference."""
        return meters / (self.earth_radius_m * np.pi / 180)
    
    def _meters_to_lon(self, meters: float, lat: float) -> float:
        """Convert meters to lon difference at given latitude."""
        return meters / (self.earth_radius_m * np.pi / 180 * np.cos(np.radians(lat)))
    
    def _generate_grid(self):
        """Generate grid points in lat/lon coordinates."""
        lat_spacing = self._meters_to_lat(self.cell_size_m)
        lon_spacing = self._meters_to_lon(self.cell_size_m, (self.min_lat + self.max_lat) / 2)
        lats = np.arange(self.min_lat, self.max_lat, lat_spacing)
        lons = np.arange(self.min_lon, self.max_lon, lon_spacing)
        if lats[-1] < self.max_lat:
            lats = np.append(lats, self.max_lat)
        if lons[-1] < self.max_lon:
            lons = np.append(lons, self.max_lon)
        self.lon_grid, self.lat_grid = np.meshgrid(lons, lats)
        self.points_lat = self.lat_grid.flatten()
        self.points_lon = self.lon_grid.flatten()
        self.n_cells = len(self.points_lat)
    
    def get_grid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) arrays of all grid points."""
        return self.points_lat, self.points_lon
    
    def get_grid_shape(self) -> Tuple[int, int]:
        """Return grid shape (n_lat, n_lon)."""
        return self.lat_grid.shape
    
    def get_cell_at_index(self, i: int, j: int) -> Tuple[float, float]:
        """Get (lat, lon) of grid cell at indices (i, j)."""
        return self.lat_grid[i, j], self.lon_grid[i, j]
    
    def get_cell_at_location(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get (i, j) indices of closest grid cell to (lat, lon)."""
        distances = np.sqrt((self.points_lat - lat)**2 + (self.points_lon - lon)**2)
        closest_idx = np.argmin(distances)
        shape = self.lat_grid.shape
        i = closest_idx // shape[1]
        j = closest_idx % shape[1]
        return i, j
    
    def get_bounds(self) -> dict:
        """Return dict with min/max lat/lon bounds."""
        return {
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
            'min_lon': self.min_lon,
            'max_lon': self.max_lon
        }
    
    def get_info(self) -> dict:
        """Return dict with grid statistics."""
        return {
            'cell_size_m': self.cell_size_m,
            'n_cells': self.n_cells,
            'grid_shape': self.get_grid_shape(),
            'bounds': self.get_bounds(),
            'lat_range': self.max_lat - self.min_lat,
            'lon_range': self.max_lon - self.min_lon
        }
