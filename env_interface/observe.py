from agents.base import Agent
from unityagents import UnityEnvironment


def watch_episode(env: UnityEnvironment, agent: Agent, **kwargs):
    agent.set_train_mode(False)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    done = env_info.local_done[0]
    while not done:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        state = next_state
