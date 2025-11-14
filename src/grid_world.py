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
        """
        Initialize grid world.
        
        Args:
            min_lat, max_lat: Latitude bounds (degrees)
            min_lon, max_lon: Longitude bounds (degrees)
            cell_size_m: Grid cell size in meters (default: 100m)
        """
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
        
        # Earth's radius in meters
        self.earth_radius_m = 6371000
        
        # Calculate grid dimensions
        self._calculate_grid_dimensions()
        
        # Generate grid points
        self._generate_grid()
    
    def _calculate_grid_dimensions(self):
        """Calculate grid dimensions in cells."""
        # Convert lat/lon differences to approximate meters
        lat_diff_m = self._lat_to_meters(self.max_lat - self.min_lat)
        lon_diff_m = self._lon_to_meters(self.max_lon - self.min_lon, self.min_lat)
        
        # Calculate number of cells
        self.n_cells_lat = int(np.ceil(lat_diff_m / self.cell_size_m))
        self.n_cells_lon = int(np.ceil(lon_diff_m / self.cell_size_m))
        
        # Total number of grid points
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
        # Calculate actual spacing in lat/lon
        lat_spacing = self._meters_to_lat(self.cell_size_m)
        lon_spacing = self._meters_to_lon(self.cell_size_m, (self.min_lat + self.max_lat) / 2)
        
        # Generate grid points
        lats = np.arange(self.min_lat, self.max_lat, lat_spacing)
        lons = np.arange(self.min_lon, self.max_lon, lon_spacing)
        
        # Ensure we include the boundaries
        if lats[-1] < self.max_lat:
            lats = np.append(lats, self.max_lat)
        if lons[-1] < self.max_lon:
            lons = np.append(lons, self.max_lon)
        
        # Create meshgrid
        self.lon_grid, self.lat_grid = np.meshgrid(lons, lats)
        
        # Flatten to get list of points
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
        # Find closest point
        distances = np.sqrt(
            (self.points_lat - lat)**2 + (self.points_lon - lon)**2
        )
        closest_idx = np.argmin(distances)
        
        # Convert flat index to 2D indices
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


if __name__ == "__main__":
    # Example usage
    # San Francisco Bay Area bounds
    min_lat, max_lat = 37.746386588546684, 37.84169189475321
    min_lon, max_lon = -122.63743707976295, -122.42795905807327
    
    # Create grid world with 10m cells
    grid = GridWorld(min_lat, max_lat, min_lon, max_lon, cell_size_m=100)
    
    # Get grid info
    print("Grid World Info:")
    print(grid.get_info())
    
    # Get grid points
    lats, lons = grid.get_grid_points()
    print(f"\nTotal grid points: {len(lats)}")
    print(f"First 5 points (lat, lon):")
    for i in range(5):
        print(f"  ({lats[i]:.4f}, {lons[i]:.4f})")
    
    # Get cell at specific location
    test_lat, test_lon = 37.5, -122.5
    i, j = grid.get_cell_at_location(test_lat, test_lon)
    cell_lat, cell_lon = grid.get_cell_at_index(i, j)
    print(f"\nClosest cell to ({test_lat}, {test_lon}): ({cell_lat:.4f}, {cell_lon:.4f})")
