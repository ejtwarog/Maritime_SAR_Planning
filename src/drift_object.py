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

    def step(self, currents: Currents, time_idx: int, dt: float) -> None:
        """Advance all objects one step using the given Currents and time index.

        dt is in seconds.
        """
        if not self.objects:
            return

        # Only advect objects that have been "released" by this time step
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
        
        # Add stochastic noise: direction (mean 0, std 10 degrees) and speed (+/-10%)
        speed = np.sqrt(u_interp ** 2 + v_interp ** 2)
        direction = np.arctan2(v_interp, u_interp)  # radians

        angle_noise_deg = np.random.normal(loc=0.0, scale=10.0, size=speed.shape)
        angle_noise_rad = np.deg2rad(angle_noise_deg)

        # Speed noise as multiplicative factor around 1.0 with ~10% std
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

        # Write back updated positions only for active objects
        active_indices = np.where(active_mask)[0]
        for idx_array, obj_idx in enumerate(active_indices):
            obj = self.objects[obj_idx]
            obj.lon = float(new_lon_active[idx_array])
            obj.lat = float(new_lat_active[idx_array])
