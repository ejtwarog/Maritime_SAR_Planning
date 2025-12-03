# Search Algorithms Development Guide

## Architecture

### Base Class: `SearchAlgorithm`

All search algorithms inherit from the abstract `SearchAlgorithm` class defined in `src/search_algorithms.py`.

```python
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
        """Generate search pattern from starting position within depth budget."""
        pass
```

### Key Concepts

- **Probability Surface**: A 2D numpy array where each cell contains the probability of finding a lost object in that grid cell. Values range from 0 to 1.
- **Grid Indices**: Cells are referenced by `(lat_idx, lon_idx)` tuples representing row and column indices in the probability surface.
- **Depth Budget**: The maximum number of cells to search in a single time step (e.g., `depth=10` means search up to 10 cells per step).
- **Searched Cells**: A set of `(lat_idx, lon_idx)` tuples that have already been searched in previous steps. Your algorithm should avoid re-searching these cells.
- **Start Position**: The argmax position (highest probability cell) where the search begins.

### Helper Methods

The base class provides utility methods for coordinate conversion:

```python
def _get_cell_indices(self, lat: float, lon: float, lat_edges: np.ndarray, lon_edges: np.ndarray) -> Tuple[int, int]:
    """Convert lat/lon coordinates to grid cell indices. Returns (-1, -1) if out of bounds."""
    pass

def _get_cell_center(self, lat_idx: int, lon_idx: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> Tuple[float, float]:
    """Convert grid cell indices to lat/lon coordinates (cell center)."""
    pass
```

## Implementing a New Algorithm

### Step 1: Create Your Algorithm Class

In `src/search_algorithms.py`, add your algorithm class:

```python
class YourSearchAlgorithm(SearchAlgorithm):
    """Brief description of your algorithm."""
    
    def __init__(self):
        """Initialize any state your algorithm needs."""
        # Example: if your algorithm maintains state across search steps
        self.current_position = None
        self.visited_cells = set()
    
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
        """
        Generate search pattern.
        
        Args:
            probability_surface: 2D array of cell probabilities
            lat_edges: Latitude grid edges
            lon_edges: Longitude grid edges
            start_lat: Starting latitude (argmax position)
            start_lon: Starting longitude (argmax position)
            depth: Maximum cells to search this step
            searched_cells: Set of already-searched cells to avoid
        
        Returns:
            List of (lat_idx, lon_idx) tuples ordered by search priority
        """
        if searched_cells is None:
            searched_cells = set()
        
        # Your algorithm implementation here
        cells_to_search = []
        
        # Example: search cells in order of decreasing probability
        flat_prob = probability_surface.flatten()
        sorted_indices = np.argsort(-flat_prob)
        
        for idx in sorted_indices:
            if len(cells_to_search) >= depth:
                break
            lat_idx, lon_idx = np.unravel_index(idx, probability_surface.shape)
            if (lat_idx, lon_idx) not in searched_cells:
                cells_to_search.append((lat_idx, lon_idx))
        
        return cells_to_search
```

### Step 2: Algorithm Design Patterns

**Example**: Trivial Greedy (continues from last position, follows highest probability neighbors)

```python
class TrivialGreedySearchAlgorithm(SearchAlgorithm):
    def __init__(self):
        self.current_lat_idx = None
        self.current_lon_idx = None
        self.path_visited = set()
    
    def search(self, ...):
        # Initialize on first call
        if self.current_lat_idx is None:
            self.current_lat_idx, self.current_lon_idx = self._get_cell_indices(...)
        
        cells = []
        for _ in range(depth):
            # Add current cell
            if (self.current_lat_idx, self.current_lon_idx) not in searched_cells:
                cells.append((self.current_lat_idx, self.current_lon_idx))
            
            # Find best neighbor and move there
            # ... neighbor selection logic ...
            
            self.path_visited.add((self.current_lat_idx, self.current_lon_idx))
            self.current_lat_idx, self.current_lon_idx = best_neighbor
        
        return cells
```

## Running Simulations

### Basic Simulation Run

To run drift-only for N steps, then begin search:

```bash
python run_experiment.py \
    --algorithm trivial_greedy \
    --depth 10 \
    --particles 500 \
    --steps 60 \
    --start-step 30 \
    --search-duration 30 \
    --output results_delayed.json
```

**Parameters:**
- `--algorithm`: Algorithm name (must match class name in lowercase with underscores)
- `--depth`: Cells to search per time step (default: 20)
- `--particles`: Number of drift particles (default: 100)
- `--steps`: Total simulation time steps (default: 30)
- `--start-step`: Time step when search begins (default: 0) | This is the time step at which the search algorithm begins to search for the lost object.
- `--search-duration`: Number of steps to run search (default: steps - start_step) | This is the number of time steps the search algorithm will run for.
- `--output`: Output JSON file path (default: search_results.json)
- `--bounds`: Search bounds as `min_lat max_lat min_lon max_lon` (default: San Francisco Bay)
- `--forecast`: Forecast data file (default: data/sfbofs.t15z.20251105.stations.forecast.nc)

### Output

The simulation generates a JSON file containing:
- **metadata**: Algorithm name, parameters, bounds, time steps
- **grid**: Latitude and longitude edges for grid cells
- **trajectories**: Particle positions at each time step
- **probability_surfaces**: Probability surface at each time step
- **search_results**: Search metrics for each search step (cells searched, probability covered, etc.)

## Visualizing Results

### Interactive Visualization

```bash
python visualizations/visualize_search.py results.json
```

This opens an interactive matplotlib window with:
- **Left panel**: Particle trajectories (blue lines, red dots for current positions)
- **Right panel**: Probability surface (heatmap) + search visualization
  - Green star: Argmax position (search start, only shown on first search step)
  - Blue rectangles: Searched cells
- **Info panel**: Metrics for current step | This panel displays the metrics for the current step, including the number of cells searched, the probability covered, and the time taken to search the cells.

**Visualization Behavior:**
- Steps 0 to (start_step - 1): Drift only, no search visualization
- Step start_step onwards: Full search visualization with probability surface and searched cells