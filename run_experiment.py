"""
Run a search and rescue simulation experiment.

Outputs a JSON file containing:
- Drift object trajectories at each time step
- Probability surfaces at each time step
- Search results (cells searched, metrics) at each time step
- Experiment metadata

Usage:
    python run_search_experiment.py --output results.json --algorithm greedy --depth 20
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from simulation_scene import SearchSimulation
from search_algorithms import (
    TrivialGreedySearchAlgorithm,
    SearchAlgorithm, RolloutPolicySearchAlgorithm,
)
from drift_object import DriftObjectCollection
from currents import Currents
from grid_world import GridWorld
from typing import List, Tuple


def _drop_particles_gaussian(
    n_particles: int,
    grid: GridWorld,
    currents: Currents,
    max_time_steps: int,
    center_lat: float = None,
    center_lon: float = None,
    init_sigma_m: float = 250.0,
    dt: float = 360.0,
) -> DriftObjectCollection:
    """Drop particles using Gaussian distribution, matching drift_object_visualizer."""
    drift_objects = DriftObjectCollection()

    if center_lat is None or center_lon is None:
        bounds = grid.get_bounds()
        center_lat = 0.5 * (bounds['min_lat'] + bounds['max_lat'])
        center_lon = 0.5 * (bounds['min_lon'] + bounds['max_lon'])

    bounds = grid.get_bounds()
    lon = np.zeros(n_particles)
    lat = np.zeros(n_particles)

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

    remaining = np.ones(n_particles, dtype=bool)
    max_iterations = 50
    iterations = 0
    while np.any(remaining) and iterations < max_iterations:
        iterations += 1
        n_remain = remaining.sum()
        dx = np.random.normal(0.0, init_sigma_m, n_remain)
        dy = np.random.normal(0.0, init_sigma_m, n_remain)
        dlon = dx / m_per_deg_lon
        dlat = dy / m_per_deg_lat
        lon_candidate = center_lon + dlon
        lat_candidate = center_lat + dlat

        in_bounds = (
            (lat_candidate >= bounds['min_lat']) &
            (lat_candidate <= bounds['max_lat']) &
            (lon_candidate >= bounds['min_lon']) &
            (lon_candidate <= bounds['max_lon'])
        )
        on_land = currents.is_on_land(lon_candidate, lat_candidate)
        valid = in_bounds & (~on_land)

        idx_remain = np.where(remaining)[0]
        assigned_idx = idx_remain[valid]
        lon[assigned_idx] = lon_candidate[valid]
        lat[assigned_idx] = lat_candidate[valid]
        remaining[assigned_idx] = False

    if np.any(remaining):
        idx_remain = np.where(remaining)[0]
        lon[idx_remain] = np.random.uniform(bounds['min_lon'], bounds['max_lon'], len(idx_remain))
        lat[idx_remain] = np.random.uniform(bounds['min_lat'], bounds['max_lat'], len(idx_remain))

    drop_window_hours = 3.0
    drop_window_steps = int(drop_window_hours * 3600 / dt)
    n_release_steps = min(drop_window_steps, max(1, max_time_steps - 1))
    per_step = n_particles // n_release_steps
    extra = n_particles % n_release_steps

    idx = 0
    for step in range(n_release_steps):
        count_this_step = per_step + (1 if step < extra else 0)
        for _ in range(count_this_step):
            if idx >= n_particles:
                break
            drift_objects.add_object(float(lon[idx]), float(lat[idx]), created_time_idx=step)
            idx += 1

    return drift_objects


def get_algorithm(name: str) -> SearchAlgorithm:
    """Get search algorithm by name."""
    algorithms = {
        "trivial_greedy": TrivialGreedySearchAlgorithm,
        "rollout_policy": RolloutPolicySearchAlgorithm,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")

    return algorithms[name]()


def run_experiment(
    forecast_file: str,
    bounds: Tuple[float, float, float, float],
    algorithm_name: str,
    search_depth: int,
    n_particles: int,
    max_time_steps: int,
    output_file: str,
    start_step: int = 0,
    search_duration: int = None,
):
    """Run search simulation and export results.
    
    Args:
        start_step: Time step at which to begin search (default: 0)
        search_duration: Number of steps to run search (default: max_time_steps - start_step)
    """
    if search_duration is None:
        search_duration = max_time_steps - start_step
    
    actual_max_steps = start_step + search_duration
    
    print(f"Initializing experiment...")
    print(f"  Algorithm: {algorithm_name}")
    print(f"  Search depth: {search_depth}")
    print(f"  Particles: {n_particles}")
    print(f"  Start step: {start_step}")
    print(f"  Search duration: {search_duration} steps")
    print(f"  Total time steps: {actual_max_steps}")

    # Setup
    grid = GridWorld(*bounds, cell_size_m=250)
    currents = Currents(forecast_file)
    currents.load_land_geometry("data/California.geojson")

    drift_objects = _drop_particles_gaussian(
        n_particles=n_particles,
        grid=grid,
        currents=currents,
        max_time_steps=max_time_steps,
        init_sigma_m=250.0,
        dt=360.0,
    )

    algorithm = get_algorithm(algorithm_name)

    # Create simulation
    sim = SearchSimulation(
        grid=grid,
        currents=currents,
        drift_objects=drift_objects,
        search_algorithm=algorithm,
        max_time_steps=actual_max_steps,
    )

    print(f"\nRunning simulation...")

    # Store trajectory data for visualization
    trajectories = np.full((actual_max_steps, n_particles, 2), np.nan)
    probability_surfaces = []
    search_results = []

    # Advance to start_step without searching
    for step_num in range(start_step):
        sim.drift_objects.step(sim.currents, time_idx=step_num, dt=sim.dt)
        sim.current_time_step += 1
        positions = drift_objects.get_positions()
        for i in range(n_particles):
            trajectories[step_num, i, 0] = positions[i, 0]
            trajectories[step_num, i, 1] = positions[i, 1]
        
        # Store probability surface for pre-search steps
        prob_surface = sim._compute_probability_surface()
        prob_list = prob_surface.tolist() if prob_surface is not None else None
        probability_surfaces.append(prob_list)

    # Initial positions at start_step
    positions = drift_objects.get_positions()
    for i in range(n_particles):
        trajectories[start_step, i, 0] = positions[i, 0]
        trajectories[start_step, i, 1] = positions[i, 1]

    # Run search simulation
    for step_num in range(start_step, actual_max_steps):
        if (step_num - start_step) % 10 == 0:
            print(f"  Step {step_num - start_step}/{search_duration}")

        metrics = sim.step(search_depth=search_depth)

        # Store probability surface
        prob_list = sim.probability_surface.tolist() if sim.probability_surface is not None else None
        probability_surfaces.append(prob_list)

        # Store search results
        search_results.append({
            "time_step": metrics.time_step,
            "time_label": metrics.time_label,
            "search_start_lat": float(metrics.search_start_lat),
            "search_start_lon": float(metrics.search_start_lon),
            "cells_searched": metrics.cells_searched,
            "probability_covered": float(metrics.probability_covered),
            "cells_searched_list": [(int(lat), int(lon)) for lat, lon in metrics.cells_searched_list],
        })

        # Store particle positions
        positions = drift_objects.get_positions()
        for i in range(n_particles):
            trajectories[step_num, i, 0] = positions[i, 0]
            trajectories[step_num, i, 1] = positions[i, 1]

    print(f"\nExporting results to {output_file}...")

    # Prepare output
    output = {
        "metadata": {
            "algorithm": algorithm_name,
            "search_depth": search_depth,
            "n_particles": n_particles,
            "start_step": start_step,
            "search_duration": search_duration,
            "total_time_steps": actual_max_steps,
            "bounds": {
                "min_lat": bounds[0],
                "max_lat": bounds[1],
                "min_lon": bounds[2],
                "max_lon": bounds[3],
            },
        },
        "grid": {
            "lat_edges": sim.lat_edges.tolist(),
            "lon_edges": sim.lon_edges.tolist(),
        },
        "trajectories": trajectories[:actual_max_steps].tolist(),
        "probability_surfaces": probability_surfaces,
        "search_results": search_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f)

    print(f"✓ Results exported to {output_file}")

    # Print summary
    summary = sim.get_metrics_summary()
    print(f"\nExperiment Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total cells searched: {summary['total_cells_searched']}")
    print(f"  Avg probability per step: {summary['avg_probability_covered_per_step']:.6f}")
    print(f"  Total probability covered: {summary['total_probability_covered']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run search and rescue simulation")
    parser.add_argument(
        "--forecast",
        default="data/sfbofs.t15z.20251105.stations.forecast.nc",
        help="Path to forecast file",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=[37.68, 37.92, -122.75, -122.35],
        help="Search bounds: min_lat max_lat min_lon max_lon",
    )
    parser.add_argument(
        "--algorithm",
        default="trivial_greedy",
        choices=["trivial_greedy", "rollout_policy"],
        help="Search algorithm",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Search depth (cells per step)",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=100,
        help="Number of drift particles",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Maximum time steps",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Time step at which to begin search (default: 0)",
    )
    parser.add_argument(
        "--search-duration",
        type=int,
        default=None,
        help="Number of steps to run search (default: steps - start_step)",
    )
    parser.add_argument(
        "--output",
        default="search_results.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    run_experiment(
        forecast_file=args.forecast,
        bounds=tuple(args.bounds),
        algorithm_name=args.algorithm,
        search_depth=args.depth,
        n_particles=args.particles,
        max_time_steps=args.steps,
        output_file=args.output,
        start_step=args.start_step,
        search_duration=args.search_duration,
    )
