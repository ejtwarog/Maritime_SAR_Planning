# Maritime SAR Planning

Algorithms for decision-making in maritime search and rescue operations using oceanographic forecasts.

## Overview

This project loads FVCOM ocean current forecasts and interpolates them to a regular geographic grid for SAR planning algorithms.

**Key Components:**
- `GridWorld`: Regular lat/lon grid with configurable cell size
- `Currents`: Loads NetCDF forecasts and interpolates to grid points
- Visualization: Maps current speed and direction over time

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