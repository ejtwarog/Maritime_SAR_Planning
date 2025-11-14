import numpy as np

def generate_probability_surface(grid_size=50, num_particles=10000, drift_std=5):
    x0, y0 = grid_size // 2, grid_size // 2  # initial center
    positions = np.zeros((num_particles, 2))

    for i in range(num_particles):
        dx = np.random.normal(0, drift_std)
        dy = np.random.normal(0, drift_std)
        positions[i] = [x0 + dx, y0 + dy]

    prob_grid = np.zeros((grid_size, grid_size))
    for (x, y) in positions:
        xi, yi = int(x), int(y)
        if 0 <= xi < grid_size and 0 <= yi < grid_size:
            prob_grid[yi, xi] += 1

    prob_grid /= np.sum(prob_grid)
    return prob_grid
