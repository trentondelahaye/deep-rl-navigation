import logging
import os
import random
from abc import ABC
from configparser import ConfigParser
from typing import Any, Dict, Iterable

import numpy as np

import torch
from agents.networks.q_network import QNetwork
from torch.nn.functional import mse_loss
from torch.optim import Adam

from .base import Agent, Experiences
from .replay_buffers import PrioritisedReplayBuffer, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = logging.getLogger()


class BaseDeepQAgent(Agent, ABC):
    def __init__(
        self,
        *args,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        buffer_size: int = 8192,
        batch_size: int = 64,
        update_every: int = 4,
        tau: float = 1e-3,
        lr: float = 5e-4,
        fc_layers: Iterable = (64, 64),
    ):
        super().__init__(*args)

        # Q learning params
        self.epsilon = 1.0
        self.t_step = 0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        # replay buffer params
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # neural network update params
        self.update_every = update_every
        self.tau = tau
        self.lr = lr
        self.fc_layers = fc_layers

        self.q_network = QNetwork(self.state_size, self.action_size, self.fc_layers)
        self.target_q_network = QNetwork(
            self.state_size, self.action_size, self.fc_layers
        )
        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)

    def save(self, *args, filename: str = "", **kwargs):
        if not len(filename):
            log.warning("Please provide a filename")
            return

        dir_name = os.path.dirname(__file__)
        path = os.path.join(dir_name, f"checkpoints/{filename}")
        state = {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, *args, filename: str = "", **kwargs):
        if not len(filename):
            log.warning("Please provide a filename")
            return

        dir_name = os.path.dirname(__file__)
        path = os.path.join(dir_name, f"checkpoints/{filename}")

        try:
            state = torch.load(path)
        except FileNotFoundError as e:
            log.warning(f"Unable to load agent state: {e}")
            return

        try:
            self.q_network.load_state_dict(state["q_network"])
            self.target_q_network.load_state_dict(state["target_q_network"])
            self.optimizer.load_state_dict(state["optimizer_state"])
        except RuntimeError as e:
            log.warning(f"Unable to load agent state: {e}")

    # noinspection PyArgumentList
    @classmethod
    def _get_base_config_params(cls, config: ConfigParser) -> Dict[str, Any]:
        return dict(
            epsilon_decay=config.getfloat("epsilon_decay"),
            min_epsilon=config.getfloat("min_epsilon"),
            gamma=config.getfloat("gamma"),
            buffer_size=config.getint("buffer_size"),
            batch_size=config.getint("batch_size"),
            update_every=config.getint("update_every"),
            tau=config.getfloat("tau"),
            lr=config.getfloat("lr"),
            fc_layers=tuple(int(layer) for layer in config.get("fc_layers").split(",")),
        )

    def act(self, state: np.ndarray) -> int:
        if self.train:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            epsilon = self.epsilon
        else:
            epsilon = 0.0

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        # Epsilon-greedy policy
        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.memory.add_experience(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample_experiences()
            self._learn(experiences)

    def _learn(self, experiences: Experiences) -> torch.Tensor:
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = (
            self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        pointwise_loss = mse_loss(q_expected, q_targets, reduce=False)
        loss = torch.mean(pointwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update()
        return pointwise_loss

    def _soft_update(self):
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )


class DeepQAgent(BaseDeepQAgent):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: ConfigParser, state_size: int, action_size: int):
        base_config_params = cls._get_base_config_params(config)
        return cls(state_size, action_size, **base_config_params)


class PrioritisedDeepQAgent(BaseDeepQAgent):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = PrioritisedReplayBuffer(self.buffer_size, self.batch_size)

    @classmethod
    def from_config(cls, config: ConfigParser, state_size: int, action_size: int):
        base_config_params = cls._get_base_config_params(config)
        return cls(state_size, action_size, **base_config_params)

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.memory.add_experience(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample_experiences()
            loss = self._learn(experiences)
            self.memory.update_priorities(loss)
