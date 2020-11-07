from typing import Tuple

import numpy as np
import random
import torch

from abc import abstractmethod
from collections import deque

from typing import NamedTuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

States = Actions = Rewards = NextStates = Dones = torch.Tensor


class Agent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.train = False

    def set_train_mode(self, train_mode: bool):
        self.train = train_mode

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, action_size: int, buffer_size: int, batch_size: int):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    # noinspection PyTypeChecker
    def sample_experiences(self) -> Tuple[States, Actions, Rewards, NextStates, Dones]:
        experiences = random.sample(self.memory, k=self.batch_size)
        return tuple(
            (
                torch.from_numpy(
                    np.vstack([getattr(e, field) for e in experiences if e is not None])
                )
                .float()
                .to(device)
            )
            for field in Experience._fields
        )

    def __len__(self):
        return len(self.memory)
