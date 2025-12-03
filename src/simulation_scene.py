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
    cells_searched_list: List[Tuple[int, int]] = field(default_factory=list)

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
        search_algorithm: SearchAlgorithm,
        max_time_steps: int = 100,
        dt: float = 360.0,
    ):
        """Initialize simulation with grid, currents, particles, and search algorithm."""
        self.grid = grid
        self.currents = currents
        self.drift_objects = drift_objects
        self.search_algorithm = search_algorithm
        self.max_time_steps = min(max_time_steps, currents.n_times - 1)
        self.dt = dt

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
        """Execute one step: advance drift, compute probability, run search."""
        self.drift_objects.step(self.currents, time_idx=self.current_time_step, dt=self.dt)
        self.probability_surface = self._compute_probability_surface()
        start_lat, start_lon = self._get_argmax_position(self.probability_surface)
        cells_to_search = self.search_algorithm.search(
            probability_surface=self.probability_surface,
            lat_edges=self.lat_edges,
            lon_edges=self.lon_edges,
            start_lat=start_lat,
            start_lon=start_lon,
            depth=search_depth,
            searched_cells=self.searched_cells,
        )
        for cell in cells_to_search:
            self.searched_cells.add(cell)
        probability_covered = self._compute_probability_covered(cells_to_search)
        time_label = self.currents.get_time_label(self.current_time_step)

        metrics = SearchMetrics(
            time_step=self.current_time_step,
            time_label=time_label,
            search_start_lat=start_lat,
            search_start_lon=start_lon,
            cells_searched=len(cells_to_search),
            probability_covered=probability_covered,
            cells_searched_list=cells_to_search,
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

        return {
            "total_steps": len(self.metrics),
            "total_cells_searched": len(self.searched_cells),
            "avg_probability_covered_per_step": float(np.mean(prob_covered)),
            "total_probability_covered": float(np.sum(prob_covered)),
            "avg_cells_per_step": float(np.mean(cells_searched)),
            "min_cells_per_step": int(np.min(cells_searched)),
            "max_cells_per_step": int(np.max(cells_searched)),
        }
