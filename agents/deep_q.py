import numpy as np
import random
import torch

from .base import Agent
from agents.networks.q_network import QNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseDeepQAgent(Agent):
    def __init__(self, *args, epsilon_decay: float = 0.995):
        super().__init__(*args)
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.q_network = QNetwork(self.state_size, self.action_size)

    def act(self, state: np.ndarray) -> int:
        if self.train:
            self.epsilon *= self.epsilon_decay
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
    pass
