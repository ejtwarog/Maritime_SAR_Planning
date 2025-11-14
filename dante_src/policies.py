import numpy as np

ACTIONS = ['N', 'S', 'E', 'W']

def greedy_action(env):
    x, y = env.position
    best_prob = -1
    best_target = None

    for i in range(env.N):
        for j in range(env.N):
            if env.visited[j, i]:
                continue
            p = env.prob_surface[j, i]
            if p > best_prob:
                best_prob = p
                best_target = (i, j)

    tx, ty = best_target
    dx = np.sign(tx - x)
    dy = np.sign(ty - y)

    if abs(dx) > abs(dy):
        return 'E' if dx > 0 else 'W'
    else:
        return 'S' if dy > 0 else 'N'


def simulate_trajectory(env, position, visited, depth):
    if depth == 0:
        return 0

    x, y = position
    best = 0

    for action in ACTIONS:
        if action == 'N': dx, dy = 0, -1
        elif action == 'S': dx, dy = 0, 1
        elif action == 'E': dx, dy = 1, 0
        else: dx, dy = -1, 0

        nx = int(np.clip(x + dx, 0, env.N - 1))
        ny = int(np.clip(y + dy, 0, env.N - 1))

        r = 0
        if not visited[ny, nx]:
            r = env.prob_surface[ny, nx]

        new_visited = visited.copy()
        new_visited[ny, nx] = True

        total = r + simulate_trajectory(env, (nx, ny), new_visited, depth - 1)
        best = max(best, total)

    return best


def k_step_action(env, K=6):
    x, y = env.position
    visited = env.visited.copy()

    # find the global peak
    peak_y, peak_x = np.unravel_index(np.argmax(env.prob_surface), env.prob_surface.shape)

    lambda_dist = 0.3  # weight for distance heuristic

    best_action = None
    best_value = -1e9

    for action in ACTIONS:
        if action == 'N': dx, dy = 0, -1
        elif action == 'S': dx, dy = 0, 1
        elif action == 'E': dx, dy = 1, 0
        else: dx, dy = -1, 0

        nx = int(np.clip(x + dx, 0, env.N - 1))
        ny = int(np.clip(y + dy, 0, env.N - 1))

        first_reward = env.prob_surface[ny, nx]

        new_visited = visited.copy()
        new_visited[ny, nx] = True

        future = simulate_trajectory(env, (nx, ny), new_visited, K - 1)

        # distance to global peak
        dist = np.sqrt((nx - peak_x)**2 + (ny - peak_y)**2)
        distance_penalty = lambda_dist * dist

        score = first_reward + future - distance_penalty

        if score > best_value:
            best_value = score
            best_action = action

    return best_action

def rollout_action(env, belief, horizon=8, n_rollouts=8):
    """
    Monte Carlo rollout planner.

    For each action in ACTIONS:
      - simulate n_rollouts random trajectories of length = horizon
      - use 'belief' as the probability map (approximate belief)
      - inside the rollout, visiting a cell gives reward = belief[y, x]
        and we set that cell's belief to zero (approximate Bayesian update)
    Returns the action with the highest average simulated return.
    """
    x, y = env.position
    N = env.N

    best_action = None
    best_value = -1e9

    for action in ACTIONS:
        total_return = 0.0

        for _ in range(n_rollouts):
            # starting from current state, apply this first action
            if action == 'N': dx, dy = 0, -1
            elif action == 'S': dx, dy = 0, 1
            elif action == 'E': dx, dy = 1, 0
            else: dx, dy = -1, 0

            px = int(np.clip(x + dx, 0, N - 1))
            py = int(np.clip(y + dy, 0, N - 1))

            # copy belief and visited for this simulated rollout
            b = belief.copy()
            visited = env.visited.copy()

            rollout_reward = 0.0

            # first step reward
            if not visited[py, px]:
                rollout_reward += b[py, px]
                visited[py, px] = True
                b[py, px] = 0.0

            cx, cy = px, py

            # simulate the rest of the horizon with random actions
            for _ in range(horizon - 1):
                a2 = np.random.choice(ACTIONS)
                if a2 == 'N': dx2, dy2 = 0, -1
                elif a2 == 'S': dx2, dy2 = 0, 1
                elif a2 == 'E': dx2, dy2 = 1, 0
                else: dx2, dy2 = -1, 0

                cx = int(np.clip(cx + dx2, 0, N - 1))
                cy = int(np.clip(cy + dy2, 0, N - 1))

                if not visited[cy, cx]:
                    rollout_reward += b[cy, cx]
                    visited[cy, cx] = True
                    b[cy, cx] = 0.0

            total_return += rollout_reward

        avg_return = total_return / n_rollouts

        if avg_return > best_value:
            best_value = avg_return
            best_action = action

    return best_action