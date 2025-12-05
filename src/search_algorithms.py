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

from typing import Callable, Optional, Set, List, Tuple
import numpy as np

class RolloutPolicySearchAlgorithm(SearchAlgorithm):
    """
    Search algorithm that uses a Monte Carlo rollout planner to choose
    the next cell, adapted from the original rollout_action() policy.

    At each step, from the current cell it:
      - considers four moves (N, S, E, W)
      - for each, runs multiple random rollouts of length = horizon
      - reward in each rollout = sum of PoD * belief over newly visited cells
      - returns the neighbor cell corresponding to the action
        with highest average simulated return.
    """

    def __init__(self, horizon: int = 8, n_rollouts: int = 8, pod: float = 0.8):
        # Search state
        self.current_lat_idx = None
        self.current_lon_idx = None
        self.path_visited = set()

        # Rollout hyperparameters
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.pod = pod

    def _rollout_policy( # Added discounting of future rewards
        self,
        current_cell: Tuple[int, int],
        probability_surface: np.ndarray,
        searched_cells: set,
        path_visited: set,
    ) -> Tuple[int, int] | None:
        """
        Monte Carlo rollout planner adapted from rollout_action(), with:
          - avoidance of NaNs / zero-probability cells
          - discounting of future rewards (gamma < 1)
          - mild bias against going into extremely low-probability regions
        """
        lat_idx, lon_idx = current_cell  # row, col
        H, W = probability_surface.shape

        # Base visited mask = everything we've already searched this mission
        base_visited = np.zeros((H, W), dtype=bool)
        for (li, lj) in searched_cells.union(path_visited):
            if 0 <= li < H and 0 <= lj < W:
                base_visited[li, lj] = True

        # Define actions and deltas (N,S,E,W) in row/col space
        ACTIONS = ['N', 'S', 'E', 'W']
        DELTAS = {
            'N': (-1, 0),  # move up: row-1
            'S': (1, 0),   # move down: row+1
            'E': (0, 1),   # move right: col+1
            'W': (0, -1),  # move left:  col-1
        }

        # Global stats for masking/biasing
        # Ignore NaNs when computing max
        finite_probs = probability_surface[np.isfinite(probability_surface)]
        if finite_probs.size == 0:
            return None
        max_prob = float(finite_probs.max())
        # Threshold: don't start a rollout into cells that are *extremely* low compared to the max
        MIN_START_FRAC = 1e-3
        min_start_prob = MIN_START_FRAC * max_prob

        gamma = 0.93  # discount factor for steps into the future

        best_value = -1e9
        best_next_cell = None

        for action in ACTIONS:
            dr, dc = DELTAS[action]

            # First move from current cell
            first_row = int(np.clip(lat_idx + dr, 0, H - 1))
            first_col = int(np.clip(lon_idx + dc, 0, W - 1))

            # Skip obviously bad starts: NaN or way below global max
            p0 = probability_surface[first_row, first_col]
            if not np.isfinite(p0) or p0 < min_start_prob:
                # We'll still pick something if *all* actions are bad,
                # but for now, just mark this as "low priority".
                candidate_penalty = True
            else:
                candidate_penalty = False

            total_return = 0.0

            for _ in range(self.n_rollouts):
                # Belief + visited copy for this rollout
                b = probability_surface.copy()
                visited = base_visited.copy()

                rollout_reward = 0.0

                r = first_row
                c = first_col

                # First step (t = 0)
                if not visited[r, c] and np.isfinite(b[r, c]) and b[r, c] > 0.0:
                    rollout_reward += (self.pod * b[r, c])  # gamma^0 = 1
                    visited[r, c] = True
                    b[r, c] = b[r, c] * (1.0 - self.pod)

                # Simulate the rest of the horizon with random actions
                discount = gamma
                for _ in range(self.horizon - 1):
                    a2 = np.random.choice(ACTIONS)
                    dr2, dc2 = DELTAS[a2]

                    r = int(np.clip(r + dr2, 0, H - 1))
                    c = int(np.clip(c + dc2, 0, W - 1))

                    if (
                        0 <= r < H and 0 <= c < W and
                        not visited[r, c] and
                        np.isfinite(b[r, c]) and b[r, c] > 0.0
                    ):
                        rollout_reward += discount * (self.pod * b[r, c])
                        visited[r, c] = True
                        b[r, c] = b[r, c] * (1.0 - self.pod)

                    discount *= gamma

                total_return += rollout_reward

            avg_return = total_return / float(self.n_rollouts)

            # Soft penalty for starting in a super-low-probability cell
            if candidate_penalty:
                avg_return *= 0.5

            if avg_return > best_value:
                best_value = avg_return
                best_next_cell = (first_row, first_col)

        return best_next_cell

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

        if searched_cells is None:
            searched_cells = set()

        # Initialize position on first call or if invalid
        if self.current_lat_idx is None or self.current_lon_idx is None:
            start_lat_idx, start_lon_idx = self._get_cell_indices(
                start_lat, start_lon, lat_edges, lon_edges
            )
            if start_lat_idx == -1 or start_lon_idx == -1:
                return []
            self.current_lat_idx = start_lat_idx
            self.current_lon_idx = start_lon_idx
            self.path_visited = set()

        cells: List[Tuple[int, int]] = []

        for _ in range(depth):
            current_cell = (self.current_lat_idx, self.current_lon_idx)

            # Record current cell if not already searched
            if current_cell not in searched_cells:
                cells.append(current_cell)

            # Use rollout-based policy to choose next cell
            next_cell = self._rollout_policy(
                current_cell=current_cell,
                probability_surface=probability_surface,
                searched_cells=searched_cells,
                path_visited=self.path_visited,
            )

            if next_cell is None:
                break

            next_lat_idx, next_lon_idx = next_cell

            # Bounds check (should be safe but we keep it defensive)
            if not (
                0 <= next_lat_idx < probability_surface.shape[0]
                and 0 <= next_lon_idx < probability_surface.shape[1]
            ):
                break

            # Update path and current position
            self.path_visited.add(current_cell)
            self.current_lat_idx, self.current_lon_idx = next_lat_idx, next_lon_idx

        return cells
