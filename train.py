import matplotlib.pyplot as plt
import numpy as np

from agents import Agent
from unityagents import UnityEnvironment


class AgentTrainer:
    def __init__(self, score_window_size: int, score_threshold: float):
        self.score_window_size = score_window_size
        self.score_threshold = score_threshold
        self.scores = []

    def train_agent(
        self,
        env: UnityEnvironment,
        agent: Agent,
        verbose: bool = True,
        exit_when_solved: bool = True,
        number_episodes: int = 1000,
        **kwargs,
    ) -> None:
        agent.set_train_mode(True)
        brain_name = env.brain_names[0]
        for _ in range(number_episodes):
            episode_number = len(self.scores) + 1
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            while True:
                action = agent.act(state)
                env_info = env.step(action)[brain_name]
                reward = env_info.rewards[0]
                next_state = env_info.vector_observations[0]
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done:
                    break

            self.scores.append(score)
            average_score_window = np.mean(self.scores[-self.score_window_size :])

            if verbose:
                print(
                    f"\rEpisode {episode_number}\tAverage Score: {average_score_window:.2f}",
                    end="",
                )
                if episode_number % 100 == 0:
                    print(
                        f"\rEpisode {episode_number}\tAverage Score: {average_score_window:.2f}",
                    )

            if (
                exit_when_solved
                and episode_number >= self.score_window_size
                and average_score_window >= self.score_threshold
            ):
                if verbose:
                    print(
                        f"\rEnvironment solved in {len(self.scores)} episodes!          ",
                    )
                break

    def plot_progress(self, *args, **kwargs):
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.title("Agent training progress")
        plt.show()
