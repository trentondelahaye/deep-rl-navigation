import configparser
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

States = Actions = Rewards = NextStates = Dones = torch.Tensor
Experiences = Tuple[States, Actions, Rewards, NextStates, Dones]


class Agent(ABC):
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.train = False

    def set_train_mode(self, train_mode: bool):
        self.train = train_mode

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: configparser.SectionProxy, state_size: int, action_size: int
    ):
        pass


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: np.float32
    next_state: np.ndarray
    done: np.float32


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
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
        experience = Experience(
            state.astype(np.float32),
            action,
            np.float32(reward),
            next_state.astype(np.float32),
            np.float32(done),
        )
        self.memory.append(experience)

    # noinspection PyTypeChecker
    def sample_experiences(self) -> Experiences:
        experiences = random.sample(self.memory, k=self.batch_size)
        return tuple(
            (
                torch.from_numpy(
                    np.vstack([getattr(e, field) for e in experiences])
                ).to(device)
            )
            for field in Experience._fields
        )

    def __len__(self):
        return len(self.memory)
