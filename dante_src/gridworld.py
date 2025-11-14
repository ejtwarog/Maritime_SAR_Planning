import numpy as np

class GridworldSAR:
    def __init__(self, prob_surface, start=(0,0)):
        self.prob_surface = prob_surface
        self.N = prob_surface.shape[0]
        self.start = start
        self.reset()

    def reset(self):
        self.position = self.start
        self.visited = np.zeros((self.N, self.N), dtype=bool)
        self.visited[self.position[1], self.position[0]] = True

    def step(self, action):
        x, y = self.position

        if action == 'N': dy = -1; dx = 0
        elif action == 'S': dy = 1; dx = 0
        elif action == 'E': dx = 1; dy = 0
        elif action == 'W': dx = -1; dy = 0
        else: dx = dy = 0

        nx = int(np.clip(x + dx, 0, self.N - 1))
        ny = int(np.clip(y + dy, 0, self.N - 1))

        reward = 0
        if not self.visited[ny, nx]:
            reward = self.prob_surface[ny, nx]

        self.position = (nx, ny)
        self.visited[ny, nx] = True

        return self.position, reward, False
