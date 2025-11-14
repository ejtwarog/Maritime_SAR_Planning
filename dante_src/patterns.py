import numpy as np

def victor_sierra_pattern(grid_size, datum, leg_length=20, sweep_angle=20):
    """
    Generate a full Victor Sierra (sector search) pattern.
    Produces 3 triangular sectors (not just radial lines).
    sweep_angle = half-angle of each wedge in degrees.
    """
    x0, y0 = datum
    base_angles = [0, 120, 240]  # main directions
    cells = []

    for base_angle in base_angles:
        for ang in range(base_angle - sweep_angle, base_angle + sweep_angle + 1):
            theta = np.radians(ang)
            dx = np.cos(theta)
            dy = np.sin(theta)

            # sweep outward along the ray
            for r in range(1, leg_length + 1):
                x = int(round(x0 + dx * r))
                y = int(round(y0 + dy * r))

                if 0 <= x < grid_size and 0 <= y < grid_size:
                    cells.append((x, y))

    return cells


def apply_manned_pattern(surface, swept_cells, reduction=0.9):
    new_surface = surface.copy()
    for (x, y) in swept_cells:
        new_surface[y, x] *= (1 - reduction)
    new_surface /= new_surface.sum()
    return new_surface
