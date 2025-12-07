import numpy as np
from dataclasses import dataclass, field
from typing import List

from scipy.interpolate import griddata

try:
    from .currents import Currents
    from .grid_world import GridWorld
except ImportError:
    from currents import Currents
    from grid_world import GridWorld


@dataclass
class DriftObject:
    """A single drift object with position and creation time index."""
    lon: float
    lat: float
    created_time_idx: int = 0
    id: int = field(default=-1)


class DriftObjectCollection:
    """Manage a collection of drift objects and advance them in time."""

    def __init__(self) -> None:
        self.objects: List[DriftObject] = []
        self._next_id: int = 0

    def add_object(self, lon: float, lat: float, created_time_idx: int = 0) -> DriftObject:
        obj = DriftObject(lon=lon, lat=lat, created_time_idx=created_time_idx, id=self._next_id)
        self._next_id += 1
        self.objects.append(obj)
        return obj

    def add_random_objects(self, n: int, grid: GridWorld, time_idx: int = 0) -> None:
        bounds = grid.get_bounds()
        lon = np.random.uniform(bounds["min_lon"], bounds["max_lon"], n)
        lat = np.random.uniform(bounds["min_lat"], bounds["max_lat"], n)
        for i in range(n):
            self.add_object(lon[i], lat[i], created_time_idx=time_idx)

    def get_positions(self) -> np.ndarray:
        if not self.objects:
            return np.empty((0, 2))
        lon = np.array([o.lon for o in self.objects])
        lat = np.array([o.lat for o in self.objects])
        return np.stack([lon, lat], axis=-1)

    def remove_objects_in_cells(self, cells: List[tuple], lat_edges: np.ndarray, 
                                lon_edges: np.ndarray, pod: float) -> int:
        """Remove objects in searched cells based on probability of detection.
        
        For each object in a searched cell, remove it with probability = pod.
        
        Args:
            cells: List of (lat_idx, lon_idx) tuples representing searched cells
            lat_edges: Latitude bin edges
            lon_edges: Longitude bin edges
            pod: Probability of detection (0 to 1)
            
        Returns:
            Number of objects removed
        """
        if not cells or pod <= 0:
            return 0
        
        # Get current positions
        positions = self.get_positions()
        if positions.shape[0] == 0:
            return 0
        
        lon = positions[:, 0]
        lat = positions[:, 1]
        
        # Find which objects are in searched cells
        objects_to_remove = []
        
        for obj_idx, (obj_lon, obj_lat) in enumerate(zip(lon, lat)):
            if np.isnan(obj_lon) or np.isnan(obj_lat):
                continue
            
            # Find which cell this object is in
            lat_idx = np.searchsorted(lat_edges, obj_lat) - 1
            lon_idx = np.searchsorted(lon_edges, obj_lon) - 1
            
            # Check if this cell was searched
            if (lat_idx, lon_idx) in cells:
                # Remove with probability = pod
                if np.random.random() < pod:
                    objects_to_remove.append(obj_idx)
        
        # Remove objects in reverse order to maintain indices
        for obj_idx in sorted(objects_to_remove, reverse=True):
            self.objects.pop(obj_idx)
        
        return len(objects_to_remove)

    def step(self, currents: Currents, time_idx: int, dt: float) -> None:
        """Advance all objects one step. dt in seconds."""
        if not self.objects:
            return

        created_times = np.array([o.created_time_idx for o in self.objects])
        active_mask = created_times <= time_idx
        if not np.any(active_mask):
            return

        lon = np.array([o.lon for o in self.objects])
        lat = np.array([o.lat for o in self.objects])
        lon_active = lon[active_mask]
        lat_active = lat[active_mask]

        u_interp = griddata((currents.station_lon, currents.station_lat),
                            currents.u[time_idx, :],
                            (lon_active, lat_active), method="linear")
        v_interp = griddata((currents.station_lon, currents.station_lat),
                            currents.v[time_idx, :],
                            (lon_active, lat_active), method="linear")

        u_interp = np.nan_to_num(u_interp, nan=0.0)
        v_interp = np.nan_to_num(v_interp, nan=0.0)
        u_interp, v_interp = currents.mask_on_land(lon_active, lat_active, u_interp, v_interp)

        speed = np.sqrt(u_interp ** 2 + v_interp ** 2)
        direction = np.arctan2(v_interp, u_interp)
        angle_noise_deg = np.random.normal(loc=0.0, scale=10.0, size=speed.shape)
        angle_noise_rad = np.deg2rad(angle_noise_deg)
        speed_factor = np.random.normal(loc=1.0, scale=0.1, size=speed.shape)
        speed_factor = np.clip(speed_factor, 0.0, None)

        speed_noisy = speed * speed_factor
        direction_noisy = direction + angle_noise_rad

        u_noisy = speed_noisy * np.cos(direction_noisy)
        v_noisy = speed_noisy * np.sin(direction_noisy)

        m_per_deg_lon = 111320 * np.cos(np.radians(lat_active))
        m_per_deg_lat = 111320
        u_deg = u_noisy / m_per_deg_lon
        v_deg = v_noisy / m_per_deg_lat

        new_lon_active = lon_active + u_deg * dt
        new_lat_active = lat_active + v_deg * dt

        active_indices = np.where(active_mask)[0]
        for idx_array, obj_idx in enumerate(active_indices):
            obj = self.objects[obj_idx]
            obj.lon = float(new_lon_active[idx_array])
            obj.lat = float(new_lat_active[idx_array])
