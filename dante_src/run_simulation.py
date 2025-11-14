from probability import generate_probability_surface
from gridworld import GridworldSAR
from policies import greedy_action, k_step_action, rollout_action
from patterns import victor_sierra_pattern, apply_manned_pattern
import copy
import matplotlib.pyplot as plt


def probability_mass_covered(path, original_surface):
    """Compute the total probability mass that the path searched."""
    visited = set(path)
    mass = 0.0
    for (x,y) in visited:
        mass += original_surface[y, x]
    return mass

# 1. Generate base probability surface
raw_surface = generate_probability_surface(grid_size=50)

# 2. Victor Sierra manned pattern
datum = (25, 25)
vs_cells = victor_sierra_pattern(
    grid_size=50,
    datum=datum,
    leg_length=20,
    sweep_angle=20
)

# 3. Apply manned pattern (reduce, not erase)
surface = apply_manned_pattern(raw_surface, vs_cells, reduction=0.2)

# Environment uses this surface for reward
env = GridworldSAR(surface, start=(20, 20))


def run_policy(env, policy, T=200, **kwargs):
    env.reset()
    total = 0.0
    path = []
    for _ in range(T):
        action = policy(env, **kwargs) if kwargs else policy(env)
        pos, reward, _ = env.step(action)

        # detect if this cell was not previously visited
        new_cell = pos not in path
        path.append(pos)
        
        coverage_bonus = 0.0002 if new_cell else 0.0
        total += reward + coverage_bonus
    return path, total


def run_rollout_policy(env, surface, T=200, horizon=12, n_rollouts=16):
    """
    Run the rollout-based planner using an evolving belief.
    The environment still uses 'surface' for the actual reward.
    """
    env.reset()
    belief = surface.copy()
    total = 0.0
    path = []

    for _ in range(T):
        action = rollout_action(env, belief, horizon=horizon, n_rollouts=n_rollouts)
        pos, reward, _ = env.step(action)
        path.append(pos)
        total += reward

        x, y = pos
        # belief[y, x] = 0.0  # approximate belief update: searched cell gets probability 0

        # Bayesian update: remove searched cells and renormalize
        removed = belief[y,x]
        belief[y, x] = 0.0

        remaining = belief.sum()
        if remaining > 0:
            belief /= remaining 

    return path, total


# 4. Evaluate greedy and rollout (we can ignore MPC for now)
greedy_path, greedy_reward = run_policy(env, greedy_action)
rollout_path, rollout_reward = run_rollout_policy(env, surface, T=200, horizon=12, n_rollouts=16)

print("Greedy reward:", greedy_reward)
print("Rollout reward:", rollout_reward)

greedy_mass = probability_mass_covered(greedy_path, raw_surface)
rollout_mass = probability_mass_covered(rollout_path, raw_surface)

print("Greedy probability mass covered:", greedy_mass)
print("Rollout probability mass covered:", rollout_mass)


def unpack(path):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    return xs, ys


gx, gy = unpack(greedy_path)
rx, ry = unpack(rollout_path)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Greedy
axs[0].imshow(surface, cmap='hot', origin='lower')
axs[0].plot(gx, gy, color='cyan', linewidth=1)
axs[0].scatter(gx[0], gy[0], color='white', s=25)
for (x, y) in vs_cells:
    axs[0].scatter(x, y, s=4, color='white')
axs[0].set_title(f"Greedy Policy\nReward = {greedy_reward:.4f}")

# Rollout
axs[1].imshow(surface, cmap='hot', origin='lower')
axs[1].plot(rx, ry, color='cyan', linewidth=1)
axs[1].scatter(rx[0], ry[0], color='white', s=25)
for (x, y) in vs_cells:
    axs[1].scatter(x, y, s=4, color='white')
axs[1].set_title(f"Rollout Planner\nReward = {rollout_reward:.4f}")

plt.tight_layout()
plt.show()

# store a copy of initial belief (before rollout search)
initial_belief = surface.copy()

# Now re-run rollout purely for belief evolution (without collecting reward)
env.reset()
belief = surface.copy()
path_temp = []

for _ in range(200):
    action = rollout_action(env, belief, horizon=12, n_rollouts=16)
    pos, reward, _ = env.step(action)
    path_temp.append(pos)

    x, y = pos

    # Bayesian update
    removed = belief[y, x]
    belief[y, x] = 0.0

    rem = belief.sum()
    if rem > 0:
        belief /= rem

# Plot initial vs final belief
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
axs2[0].imshow(initial_belief, cmap='hot', origin='lower')
axs2[0].set_title("Initial Belief Surface")

axs2[1].imshow(belief, cmap='hot', origin='lower')
axs2[1].set_title("Belief After Rollout Search")

plt.tight_layout()
plt.show()