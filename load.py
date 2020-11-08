from configparser import ConfigParser
from typing import Callable, Dict, List, Union

import agents
from agents import Agent
from observe import watch_episode
from train import AgentTrainer
from unityagents import UnityEnvironment


def load_env(env_path: str, no_graphics: bool = False) -> UnityEnvironment:
    return UnityEnvironment(file_name=env_path, no_graphics=no_graphics)


def load_agent(env: UnityEnvironment, agent_cfg: str) -> agents.Agent:
    config = ConfigParser()
    config.read("./agents/configs.cfg")

    try:
        section = config[agent_cfg]
    except KeyError:
        raise Exception("Config section not found")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    agent_name = section.get("agent")

    try:
        agent: Agent = getattr(agents, agent_name)
        return agent.from_config(section, state_size, action_size)
    except AttributeError:
        raise Exception(f"Unrecognised agent {agent_name}")


def build_commands(trainer: AgentTrainer) -> Dict[str, Callable]:
    return {
        "exit": lambda *args: True,
        "watch": watch_episode,
        "train": trainer.train_agent,
        "plot": trainer.plot_progress,
    }


def process_kwarg(key: str, value: str) -> Union[int, float, bool]:
    if value.isdigit():
        return int(value)
    elif value.isnumeric():
        return float(value)
    elif value in {"True", "true", "T", "t"}:
        return True
    elif value in {"False", "false", "F", "f"}:
        return False
    raise ValueError(f"Unknown param type {key}={value}")


def build_kwargs(raw_kwargs: List[str]) -> Dict[str, str]:
    return {
        key: process_kwarg(key, value)
        for kwarg in raw_kwargs
        for key, value in [kwarg.split("=")]
    }
