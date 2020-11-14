import configparser
from abc import ABC, abstractmethod
from typing import Tuple

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

    @abstractmethod
    def save(self, *args, filename: str = "", **kwargs):
        pass

    @abstractmethod
    def load(self, *args, filename: str = "", **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: configparser.SectionProxy, state_size: int, action_size: int
    ):
        pass
