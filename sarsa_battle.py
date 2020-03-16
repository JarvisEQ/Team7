## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb

import torch

import numpy as np
import random as r

from everglades_server import server

EPISODES = 1

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/sarsa.py'
#agent1_file = 'agents/same_commands.py'
agent1_file = 'agents/random_actions.py'

config_dir = './config/'
map_file = config_dir + 'DemoMap.json'
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

# Debug view
debug = 0

# Game view
view = 1

# Create telemetry
createOut = 0

## Specific Imports
agent0_name, agent0_extension = os.path.splitext(agent0_file)
agent0_mod = importlib.import_module(agent0_name.replace('/','.'))
agent0_class = getattr(agent0_mod, os.path.basename(agent0_name))

agent1_name, agent1_extension = os.path.splitext(agent1_file)
agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))


## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

# Initalize agents 
players[0] = agent0_class()
names[0] = agent0_class.__name__

players[1] = agent1_class(env.num_actions_per_turn)
names[1] = agent1_class.__name__


# Episode loop
for episode in range(EPISODES):
    print(f"-------------- Episode {episode} --------------\n\n")

    #Environment Initialization
    state = env.reset(
            players=players,
            config_dir = config_dir,
            map_file = map_file,
            unit_file = unit_file,
            output_dir = output_dir,
            pnames = names,
            debug = debug,
            view = view,
            out = createOut
    )

    actions  = {}
    Qs = {}
    actions_ = {}

    # Inital actions
    actions[0], Qs[0] = players[0].get_action( state[0] )
    actions[1], Qs[1] = players[1].get_action( state[1] )

    # Game Loop
    done = 0
    while not done:
        if debug:
            env.game.debug_state()

        if view:
            env.game.view_state()


        # Take action
        state_, reward, done, info = env.step(actions)

        # Get new actions
        actions_[0], Qs[0] = players[0].get_action( state_[0] )
        actions_[1], Qs[1] = players[1].get_action( state_[1] )
        
        print(f'state_ \n{state_[0]}')
        print(f'state \n{state[0]}')
        print(f'actions \n{actions[0]}')

        # create experiences tuple
        experiences = (torch.tensor([state[0], actions[0], reward, state_[0], done]))

        # learn 
        players[0].learn(experiences)

        # Reward / Observations for the action
        print(f"----- Reward -----\n{reward}\n")

    print(f"Reward = {reward}")
