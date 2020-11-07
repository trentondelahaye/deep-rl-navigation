import numpy as np

from .base import Agent


class RandomAgent(Agent):
    def act(self, state: np.ndarray) -> int:
        return np.random.randint(self.action_size)
