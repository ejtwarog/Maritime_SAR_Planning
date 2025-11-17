# Maritime SAR Planning

Algorithms for decision-making in maritime search and rescue operations using oceanographic forecasts.

## Overview

This project loads FVCOM ocean current forecasts and interpolates them to a regular geographic grid for SAR planning algorithms.

**Key Components:**
- `GridWorld`: Regular lat/lon grid with configurable cell size
- `Currents`: Loads NetCDF forecasts and interpolates to grid points
- Visualization: Maps current speed and direction over time

## Project Structure

- **`src/`**
  - `grid_world.py` – grid definition and utilities.
  - `currents.py` – forecast loading, interpolation, and land masking.
  - `drift_object.py` – drift object representation and collection dynamics.

- **`visualizations/`**
  - (Historical) `prototype_drift.py` – early drift/field visualization utilities.
  - `scenario_creator.py` – helpers for setting up specific drift scenarios (case studies, experiments).
  - (Recreated) `drift_dashboard.py` – interactive drift dashboard (trajectories + probability surface).

- **`data/`**
  - `California.geojson` – coastline / land geometry for masking and map overlays.
  - `sfbofs...forecast.nc` – FVCOM station forecast file(s) used as current input.
  - `case_study_windsurfer/` – example probability-surface JSONs for case studies.
  - `visualize_probability.py` – standalone probability-surface dashboard reading JSON outputs.

- **Root**
  - `README.md` – overview, architecture notes, and pointers to visualizations.
  - `probability_surface.json` – example exported probability surface from a drift run (for debugging/analysis).

## Quick Start

```python
from grid_world import GridWorld
from currents import Currents

# Create grid (SF Bay area, 500m cells)
grid = GridWorld(37.7, 37.9, -122.7, -122.4, cell_size_m=500)

# Load currents and interpolate
currents = Currents("data/sfbofs.t15z.20251105.stations.forecast.nc")
grid = currents.populate_gridworld(grid)
```

## Visualizations

## Architecture Notes

- **GridWorld (`src/grid_world.py`)**
  - Defines a regular lat/lon grid within configurable bounds and cell size (meters).
  - Provides grid points, grid shape, bounds, and helper methods to map between geographic locations and grid cells.

- **Currents (`src/currents.py`)**
  - Loads FVCOM / NetCDF ocean current forecasts (station-based u/v components).
  - Interpolates currents to arbitrary positions or onto a `GridWorld`.
  - Supports land masking using a cached `California.geojson` geometry, with helpers for masking velocities and testing if points lie on land.

- **DriftObject & DriftObjectCollection (`src/drift_object.py`)**
  - `DriftObject` represents a single drifting asset with lon/lat and a release time index.
  - `DriftObjectCollection` manages many objects, interpolates currents at their locations, and advects them forward in time.
  - Includes simple stochastic perturbations: direction noise (mean 0, ~10° std) and speed noise (~±10%) on the current vectors.
  - Honors per-object `created_time_idx` so staged releases are possible.

- **DriftDashboard (`visualizations/drift_dashboard.py`)**
  - Interactive Matplotlib dashboard for exploring drift simulations over the SF Bay grid.
  - Key features:
    - Gaussian initial spatial distribution around a configurable center with configurable spread (meters).
    - Staged particle releases: initial fraction at t=0, remaining particles released over the first N time steps.
    - Land masking and California coastline overlay from `data/California.geojson`.
    - Time slider to scrub through forecast time steps.
    - Side-by-side panels: trajectories on the left, probability surface over `GridWorld` cells on the right.
  - Can export a per-timestep probability surface to JSON for downstream analysis.

- **Probability Visualization (`data/visualize_probability.py`)**
  - Loads a precomputed probability-surface JSON (e.g. from `data/case_study_windsurfer/ex1_probability_surface.json`).
  - Provides a lightweight dashboard with a time slider that shows the probability heatmap over the same grid.
  - Zero-probability cells are transparent; California coastline is overlaid when Cartopy is available.


Three visualization modes for current analysis:

1. **Station Vectors** - Raw forecast data at station locations
   ```python
   from visualizations.prototype_drift import visualize_station_vectors
   fig, ax = visualize_station_vectors(grid, currents, time_step=0)
   ```

2. **Vector Field** - Interpolated vectors on GridWorld points
   ```python
   from visualizations.prototype_drift import visualize_vector_field
   fig, ax = visualize_vector_field(grid, currents, time_step=0)
   ```

3. **Batch Generation** - Generate visualizations for multiple time steps
   ```bash
   python visualizations/prototype_drift.py
   ```

All vectors show **1-hour drift** for SAR planning.

## Data Format

NetCDF files from FVCOM contain:
- **`u`, `v`**: Eastward/northward velocity (m/s) at surface layer
- **`lon`, `lat`** (or `lonc`, `latc`): Station/element coordinates
- **`Times`** (or `time`): Time steps

## Project Structure

```
src/
  ├── grid_world.py          # Regular geographic grid
  ├── currents.py            # Forecast loading and interpolation
visualizations/
  └── prototype_drift.py     # Station and vector field visualizations
data/
  ├── *.nc                   # NetCDF forecast files
  └── ca_shoreline.geojson   # CA coastline boundaries
```