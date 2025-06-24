import numpy as np
import gym
from gym import spaces

class SimplePlacementEnv(gym.Env):
    """A simple grid based placement environment."""

    def __init__(self, area_size=(10, 10), components=None, connections=None):
        super().__init__()
        self.area_height, self.area_width = area_size
        # components is list of (height, width)
        if components is None:
            components = [(2, 2), (3, 1), (1, 3)]
        self.components = components
        self.num_components = len(components)
        self.connections = connections or []  # list of (idx_a, idx_b)

        # observation is an integer grid representing placed component index+1
        self.observation_space = spaces.Box(low=0, high=self.num_components,
                                            shape=(self.area_height, self.area_width),
                                            dtype=np.int32)
        # action is choosing a grid cell to place current component
        self.action_space = spaces.Discrete(self.area_height * self.area_width)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.area_height, self.area_width), dtype=np.int32)
        self.current = 0  # index of component to place
        return self.grid.copy()

    def _can_place(self, y, x, shape):
        h, w = shape
        if y + h > self.area_height or x + w > self.area_width:
            return False
        sub = self.grid[y:y+h, x:x+w]
        return np.all(sub == 0)

    def _place(self, y, x, shape):
        h, w = shape
        self.grid[y:y+h, x:x+w] = self.current + 1

    def step(self, action):
        y = action // self.area_width
        x = action % self.area_width
        shape = self.components[self.current]
        reward = -1.0
        done = False

        if self._can_place(y, x, shape):
            self._place(y, x, shape)
            reward = 0.0
            self.current += 1
            if self.current == self.num_components:
                reward = -self._total_connection_length()
                done = True
        else:
            # invalid placement
            reward = -5.0

        return self.grid.copy(), reward, done, {}

    def _centroid(self, component_idx):
        loc = np.argwhere(self.grid == component_idx + 1)
        if loc.size == 0:
            return None
        y_min, x_min = loc.min(axis=0)
        y_max, x_max = loc.max(axis=0)
        return np.array([(y_min + y_max) / 2, (x_min + x_max) / 2])

    def _total_connection_length(self):
        total = 0.0
        for a, b in self.connections:
            ca = self._centroid(a)
            cb = self._centroid(b)
            if ca is not None and cb is not None:
                total += np.abs(ca - cb).sum()
        return total
