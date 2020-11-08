from configparser import ConfigParser

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    def act(self, state: np.ndarray) -> int:
        return np.random.randint(self.action_size)

    def step(self, *args):
        pass

    @classmethod
    def from_config(cls, config: ConfigParser, state_size: int, action_size: int):
        return cls(state_size, action_size)
