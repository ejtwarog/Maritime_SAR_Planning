"""
Search Simulation Framework

Orchestrates the full search and rescue simulation cycle:
1. Time step drift objects
2. Generate probability surface
3. Run search algorithm
4. Update search state
5. Repeat

This module is designed for easy algorithm swapping and reproducible experiments.
"""

import json
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

try:
    from .search_algorithms import SearchAlgorithm
    from .drift_object import DriftObjectCollection
    from .currents import Currents
    from .grid_world import GridWorld
except ImportError:
    from search_algorithms import SearchAlgorithm
    from drift_object import DriftObjectCollection
    from currents import Currents
    from grid_world import GridWorld

@dataclass
class SearchMetrics:
    """Metrics for a single search step."""
    time_step: int
    time_label: str
    search_start_lat: float
    search_start_lon: float
    cells_searched: int
    probability_covered: float
    objects_removed: int = 0
    cells_searched_list: List[Tuple[int, int]] = field(default_factory=list)
    # Platform-specific metrics (for dual-platform simulations)
    platform_metrics: dict = field(default_factory=dict)  # {platform_name: {cells, prob_covered, objects_removed, cells_list}}

class SearchSimulation:
    """
    Orchestrates search and rescue simulation with pluggable search algorithms.
    
    Workflow:
    - Initialize with drift objects, currents, and grid
    - At each time step: advance drift, compute probability surface, run search
    - Track metrics and search history
    """

    def __init__(
        self,
        grid: GridWorld,
        currents: Currents,
        drift_objects: DriftObjectCollection,
        search_algorithm: SearchAlgorithm = None,
        search_algorithms: dict = None,
        max_time_steps: int = 100,
        dt: float = 360.0,
        pod: float = 0.8,
    ):
        """Initialize simulation with grid, currents, particles, and search algorithm(s).
        
        Args:
            search_algorithm: Single search algorithm (legacy)
            search_algorithms: Dict of {platform_name: SearchAlgorithm} for multi-platform search
        """
        self.grid = grid
        self.currents = currents
        self.drift_objects = drift_objects
        
        # Support both single and multi-platform configurations
        if search_algorithms is not None:
            self.search_algorithms = search_algorithms
        elif search_algorithm is not None:
            self.search_algorithms = {"default": search_algorithm}
        else:
            raise ValueError("Either search_algorithm or search_algorithms must be provided")
        
        self.max_time_steps = min(max_time_steps, currents.n_times - 1)
        self.dt = dt
        self.pod = pod  # Probability of detection
        self.initial_object_count = len(drift_objects.objects)  # Track initial count

        self._setup_grid_metadata()
        self.current_time_step = 0
        self.probability_surface = None
        self.searched_cells = set()
        self.metrics = []

    def _setup_grid_metadata(self):
        """Precompute grid cell edges for probability surface calculations."""
        grid_lat, grid_lon = self.grid.get_grid_points()
        grid_shape = self.grid.get_grid_shape()

        lat_2d = grid_lat.reshape(grid_shape)
        lon_2d = grid_lon.reshape(grid_shape)
        lat_axis = lat_2d[:, 0]
        lon_axis = lon_2d[0, :]

        # Compute bin edges
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
        self.grid_shape = grid_shape

    def _compute_probability_surface(self) -> np.ndarray:
        """Compute probability surface from current drift object positions."""
        positions = self.drift_objects.get_positions()

        if positions.shape[0] == 0:
            return np.zeros((len(self.lat_edges) - 1, len(self.lon_edges) - 1))

        lon = positions[:, 0]
        lat = positions[:, 1]

        mask = ~np.isnan(lon) & ~np.isnan(lat)
        if not np.any(mask):
            return np.zeros((len(self.lat_edges) - 1, len(self.lon_edges) - 1))

        hist, _, _ = np.histogram2d(
            lat[mask],
            lon[mask],
            bins=[self.lat_edges, self.lon_edges],
        )

        if hist.sum() > 0:
            prob = hist / hist.sum()
        else:
            prob = np.zeros_like(hist)

        return prob

    def _get_argmax_position(self, prob_surface: np.ndarray) -> Tuple[float, float]:
        """Get lat/lon of highest probability cell."""
        if prob_surface.size == 0 or np.max(prob_surface) == 0:
            bounds = self.grid.get_bounds()
            return (
                0.5 * (bounds["min_lat"] + bounds["max_lat"]),
                0.5 * (bounds["min_lon"] + bounds["max_lon"]),
            )

        lat_idx, lon_idx = np.unravel_index(np.argmax(prob_surface), prob_surface.shape)

        lat = 0.5 * (self.lat_edges[lat_idx] + self.lat_edges[lat_idx + 1])
        lon = 0.5 * (self.lon_edges[lon_idx] + self.lon_edges[lon_idx + 1])

        return (lat, lon)

    def step(self, search_depth: int) -> SearchMetrics:
        """Execute one step: advance drift, compute probability, run search from all platforms."""
        self.drift_objects.step(self.currents, time_idx=self.current_time_step, dt=self.dt)
        self.probability_surface = self._compute_probability_surface()
        start_lat, start_lon = self._get_argmax_position(self.probability_surface)
        
        # Collect cells from all platforms
        all_cells_to_search = []
        platform_metrics = {}
        platform_cells_map = {}  # Store cells per platform for visualization
        first_platform_start_cell = None  # Track actual starting cell from first platform
        first_platform_start_lat = None
        first_platform_start_lon = None
        
        for i, (platform_name, algorithm) in enumerate(self.search_algorithms.items()):
            # For subsequent platforms, use the starting cell from the first platform
            if i > 0 and first_platform_start_cell is not None:
                platform_start_lat = first_platform_start_lat
                platform_start_lon = first_platform_start_lon
            else:
                platform_start_lat = start_lat
                platform_start_lon = start_lon
            
            cells_to_search = algorithm.search(
                probability_surface=self.probability_surface,
                lat_edges=self.lat_edges,
                lon_edges=self.lon_edges,
                start_lat=platform_start_lat,
                start_lon=platform_start_lon,
                depth=search_depth,
                searched_cells=self.searched_cells,
            )
            
            # Capture the actual starting cell from the first platform
            if first_platform_start_cell is None and len(cells_to_search) > 0:
                first_platform_start_cell = cells_to_search[0]
                lat_idx, lon_idx = first_platform_start_cell
                # Convert cell indices to lat/lon (cell center)
                first_platform_start_lat = 0.5 * (self.lat_edges[lat_idx] + self.lat_edges[lat_idx + 1])
                first_platform_start_lon = 0.5 * (self.lon_edges[lon_idx] + self.lon_edges[lon_idx + 1])
            
            all_cells_to_search.extend(cells_to_search)
            prob_covered = self._compute_probability_covered(cells_to_search)
            platform_cells_map[platform_name] = cells_to_search
            
            platform_metrics[platform_name] = {
                "cells_searched": len(cells_to_search),
                "probability_covered": float(prob_covered),
                "cells_list": [(int(lat), int(lon)) for lat, lon in cells_to_search],
            }
        
        # Remove duplicates while preserving order
        cells_to_search = list(dict.fromkeys(all_cells_to_search))
        
        for cell in cells_to_search:
            self.searched_cells.add(cell)
        probability_covered = self._compute_probability_covered(cells_to_search)
        
        # Remove objects in searched cells based on probability of detection
        objects_removed = self.drift_objects.remove_objects_in_cells(
            cells=cells_to_search,
            lat_edges=self.lat_edges,
            lon_edges=self.lon_edges,
            pod=self.pod
        )
        
        # Track objects removed per platform (proportional to cells searched)
        for platform_name in platform_metrics:
            if len(cells_to_search) > 0:
                platform_cells = platform_metrics[platform_name]["cells_searched"]
                platform_metrics[platform_name]["objects_removed"] = int(
                    objects_removed * platform_cells / len(cells_to_search)
                )
            else:
                platform_metrics[platform_name]["objects_removed"] = 0
        
        # Recompute probability surface after removing objects
        self.probability_surface = self._compute_probability_surface()
        
        time_label = self.currents.get_time_label(self.current_time_step)

        metrics = SearchMetrics(
            time_step=self.current_time_step,
            time_label=time_label,
            search_start_lat=start_lat,
            search_start_lon=start_lon,
            cells_searched=len(cells_to_search),
            probability_covered=probability_covered,
            objects_removed=objects_removed,
            cells_searched_list=cells_to_search,
            platform_metrics=platform_metrics,
        )

        self.metrics.append(metrics)
        self.current_time_step += 1

        return metrics

    def _compute_probability_covered(self, cells_searched: List[Tuple[int, int]]) -> float:
        """Compute total probability mass in searched cells."""
        if self.probability_surface is None or len(cells_searched) == 0:
            return 0.0

        total_prob = 0.0
        for lat_idx, lon_idx in cells_searched:
            if 0 <= lat_idx < self.probability_surface.shape[0] and \
               0 <= lon_idx < self.probability_surface.shape[1]:
                total_prob += self.probability_surface[lat_idx, lon_idx]

        return float(total_prob)


    def run(self, search_depth: int, num_steps: Optional[int] = None) -> List[SearchMetrics]:
        """Run simulation for multiple steps."""
        if num_steps is None:
            num_steps = self.max_time_steps

        for _ in range(num_steps):
            if self.current_time_step >= self.max_time_steps:
                break
            self.step(search_depth)

        return self.metrics

    def export_results(self, filename: str) -> None:
        """Export simulation results to JSON."""
        output = {
            "search_algorithm": self.search_algorithm.__class__.__name__,
            "max_time_steps": self.max_time_steps,
            "total_steps_run": len(self.metrics),
            "total_cells_searched": len(self.searched_cells),
            "steps": [],
        }

        for metric in self.metrics:
            output["steps"].append({
                "time_step": metric.time_step,
                "time_label": metric.time_label,
                "search_start_lat": float(metric.search_start_lat),
                "search_start_lon": float(metric.search_start_lon),
                "cells_searched": metric.cells_searched,
                "probability_covered": float(metric.probability_covered),
            })

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

    def get_metrics_summary(self) -> dict:
        """Get summary statistics of simulation."""
        if not self.metrics:
            return {}

        prob_covered = [m.probability_covered for m in self.metrics]
        cells_searched = [m.cells_searched for m in self.metrics]
        objects_removed = [m.objects_removed for m in self.metrics]
        total_removed = int(np.sum(objects_removed))

        return {
            "total_steps": len(self.metrics),
            "total_cells_searched": len(self.searched_cells),
            "initial_objects": self.initial_object_count,
            "total_objects_removed": total_removed,
            "removal_rate": float(total_removed / self.initial_object_count) if self.initial_object_count > 0 else 0.0,
            "avg_probability_covered_per_step": float(np.mean(prob_covered)),
            "avg_cells_per_step": float(np.mean(cells_searched)),
            "min_cells_per_step": int(np.min(cells_searched)),
            "max_cells_per_step": int(np.max(cells_searched)),
        }
