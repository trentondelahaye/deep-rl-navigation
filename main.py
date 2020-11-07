import agents
import click
import logging

from unityagents import UnityEnvironment
from watch_agent import watch_episode

log = logging.getLogger()

COMMAND_TO_FUNC = {
    "exit": lambda *args: True,
    "watch": watch_episode,
}


def load_env(env_path: str) -> UnityEnvironment:
    return UnityEnvironment(file_name=env_path)


def load_agent(env: UnityEnvironment, agent_name: str) -> agents.Agent:
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    try:
        return getattr(agents, agent_name)(state_size, action_size)
    except AttributeError:
        raise Exception("Unrecognised agent")


@click.command()
@click.option("--unity-env", default="./Banana.app", help="Path to UnityEnvironment")
@click.option("--agent", default="RandomAgent", help="Name of agent to run")
def main(unity_env, agent):
    env = load_env(unity_env)
    agent = load_agent(env, agent)
    to_exit = False

    while not to_exit:
        command = input("Input command: ").lower()
        if command not in COMMAND_TO_FUNC:
            log.info(f"Unrecognised command, select from {set(COMMAND_TO_FUNC.keys())}")
            continue
        func = COMMAND_TO_FUNC[command]
        to_exit = func(env, agent)

    env.close()


if __name__ == "__main__":
    main()
