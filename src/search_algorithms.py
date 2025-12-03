"""Search algorithm implementations for SAR simulation."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms."""

    @abstractmethod
    def search(
        self,
        probability_surface: np.ndarray,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
        start_lat: float,
        start_lon: float,
        depth: int,
        searched_cells: set = None,
    ) -> List[Tuple[int, int]]:
        """Generate search pattern from starting position within depth budget.
        
        Returns list of (lat_idx, lon_idx) tuples ordered by priority, max length = depth.
        """
        pass

    def _get_cell_indices(self, lat: float, lon: float, lat_edges: np.ndarray, lon_edges: np.ndarray) -> Tuple[int, int]:
        """Convert lat/lon to grid cell indices. Returns (-1, -1) if out of bounds."""
        lat_idx = np.searchsorted(lat_edges, lat) - 1
        lon_idx = np.searchsorted(lon_edges, lon) - 1

        if lat_idx < 0 or lat_idx >= len(lat_edges) - 1:
            return (-1, -1)
        if lon_idx < 0 or lon_idx >= len(lon_edges) - 1:
            return (-1, -1)

        return (lat_idx, lon_idx)

    def _get_cell_center(self, lat_idx: int, lon_idx: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> Tuple[float, float]:
        """Convert grid cell indices to lat/lon coordinates (cell center)."""
        lat = 0.5 * (lat_edges[lat_idx] + lat_edges[lat_idx + 1])
        lon = 0.5 * (lon_edges[lon_idx] + lon_edges[lon_idx + 1])
        return (lat, lon)


class TrivialGreedySearchAlgorithm(SearchAlgorithm):
    """Trivial Greedy: at each step, choose max of up, down, left, right neighbors.
    
    Maintains state across search steps to continue from last position rather than restarting.
    """

    def __init__(self):
        """Initialize with no current position."""
        self.current_lat_idx = None
        self.current_lon_idx = None
        self.path_visited = set()  # Track all visited cells in current path

    def search(
        self,
        probability_surface: np.ndarray,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
        start_lat: float,
        start_lon: float,
        depth: int,
        searched_cells: set = None,
    ) -> List[Tuple[int, int]]:
        """Search by greedily moving to highest probability neighbor at each step.
        
        Continues from previous position if available, otherwise starts at argmax.
        """
        if searched_cells is None:
            searched_cells = set()

        # Initialize position on first call or if no valid current position
        if self.current_lat_idx is None or self.current_lon_idx is None:
            start_lat_idx, start_lon_idx = self._get_cell_indices(
                start_lat, start_lon, lat_edges, lon_edges
            )
            if start_lat_idx == -1 or start_lon_idx == -1:
                return []
            self.current_lat_idx = start_lat_idx
            self.current_lon_idx = start_lon_idx
            self.path_visited = set()

        cells = []

        for _ in range(depth):
            # Add current cell if not already searched
            if (self.current_lat_idx, self.current_lon_idx) not in searched_cells:
                cells.append((self.current_lat_idx, self.current_lon_idx))

            # Get neighbors: up, down, left, right
            neighbors = [
                (self.current_lat_idx - 1, self.current_lon_idx),  # up (decreasing lat)
                (self.current_lat_idx + 1, self.current_lon_idx),  # down (increasing lat)
                (self.current_lat_idx, self.current_lon_idx - 1),  # left (decreasing lon)
                (self.current_lat_idx, self.current_lon_idx + 1),  # right (increasing lon)
            ]

            # Find valid neighbor with highest probability
            best_neighbor = None
            best_prob = -1

            for lat_idx, lon_idx in neighbors:
                if (
                    0 <= lat_idx < probability_surface.shape[0]
                    and 0 <= lon_idx < probability_surface.shape[1]
                    and (lat_idx, lon_idx) not in self.path_visited
                    and (lat_idx, lon_idx) not in searched_cells
                ):
                    prob = probability_surface[lat_idx, lon_idx]
                    if prob > best_prob:
                        best_prob = prob
                        best_neighbor = (lat_idx, lon_idx)

            if best_neighbor is None:
                break

            self.path_visited.add((self.current_lat_idx, self.current_lon_idx))
            self.current_lat_idx, self.current_lon_idx = best_neighbor

        return cells