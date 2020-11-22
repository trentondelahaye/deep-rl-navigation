# Deep Reinforcement Learning - Banana

## Overview

This project is part of the fulfillment of the Udacity nanodegree. DeepQ reinforment learning agents were programmed to solve a unity based banana-collecting game. Specific emphasis was placed on making it easy for others to run and extend the project, removing repetition from defining agents and making it easy to change hyperparameters in a config file. It also serves as a basis for future Deep RL projects as it has been designed to be extensible and generic to other unity environments by inferring the state and action sizes.

A gif of a trained DeepQ agent playing the game is shown below.
![](trained_agent.gif)

## Game description

The game is set in a 2D arena of yellow and blue bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The aim of the game is to collect as much reward before the episode ends.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Four discrete actions are available to the agent, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic with 300 frames per game. In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies

Set up of the Python environment in order to run this project.

1. (Suggested) Create and activate a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drl-navigation python=3.6
	source activate drl-navigation
	```
	- __Windows__: 
	```bash
	conda create --name drl-navigation python=3.6 
	activate drl-navigation
	```
	
2. Clone this repository provided by Udacity, then install dependencies.
  ```bash
  git clone https://github.com/udacity/deep-reinforcement-learning.git
  cd deep-reinforcement-learning/python
  pip install .
  ```
  
3. Install the additional dependency for this project.
  ```bash
  pip install click
  ```

4. Download the environment as per your operating system and unzip to the project directory:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - macOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    
## Training and watching agents

In order to start the program, run `python main.py` in the project directory (using the virtual env). Running `python main.py --help` will show the following options:

```
Options:
  --unity-env TEXT            Path to UnityEnvironment
  --agent-cfg TEXT            Section of config used to load agent
  --no-graphics / --graphics  Load environment without graphics
  --help                      Show this message and exit
```

The `unity-env` needs to point to the directory that you placed the environment from Dependencies Step 4 and the `agent-cfg` can point to any config section specified in `agents/configs.cfg`. You can change hyperparameters and configurations of the DeepQ agents in `agents/configs.cfg`. For example 

```
python main.py --unity-env ./Banana.app --agent-cfg PrioritisedDeepQ
```

Once the program has started, it will display environment information and await a command. The following commands are currently available:

```
{'save-agent', 'exit', 'plot', 'load-agent', 'watch', 'train'}
```

Using the flag `--help` after a command will show the keyword arguments available for the command, if any. For example, the user can train an agent with

```
train number_episodes=20 exit_when_solved=False
```

which will train the agent for 20 more episodes, not quitting when the average over the past 100 is above 13. The command `watch` is unavailable if the unity environment has been started in `no-graphics` mode.
