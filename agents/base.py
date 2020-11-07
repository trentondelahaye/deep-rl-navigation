import numpy as np

from abc import abstractmethod


class Agent:
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass
