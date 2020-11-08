import random
from abc import ABC
from configparser import ConfigParser
from typing import Dict, Any

import numpy as np

import torch
from agents.networks.q_network import QNetwork
from torch.nn.functional import mse_loss
from torch.optim import Adam

from .base import Agent, Experiences, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    ):
        super().__init__(*args)

        # Q learning params
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        # replay buffer params
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # neural network update params
        self.update_every = update_every
        self.tau = tau
        self.lr = lr

        self.q_network = QNetwork(self.state_size, self.action_size)

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


class DeepQAgent(BaseDeepQAgent):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.target_q_network = QNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.t_step = 0

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
            self._learn(experiences)

    def _learn(self, experiences: Experiences):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = (
            self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        loss = mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update()

    def _soft_update(self):
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
