## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb

import numpy as np
import random as r
from Stats import Stats

from everglades_server import server

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/random_actions.py'
#agent1_file = 'agents/same_commands.py'
agent1_file = 'agents/random_actions.py'

config_dir = './config/'
map_file = config_dir + 'DemoMap.json'
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

# 0 for no debug /  1 for debug
debug = 0

view = 0

createOut = 0

numberOfGames = 100

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


players[0] = agent0_class(env.num_actions_per_turn)
names[0] = agent0_class.__name__

players[1] = agent1_class(env.num_actions_per_turn)
names[1] = agent1_class.__name__

# load model
#players[0].load_model()
#player[1].load_model()

s = Stats()
for game in range(numberOfGames):
    
    # get inital state
    current_state = env.reset(
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
    new_state = current_state 

    actions = {}
    Qs = {}

    # Game Loop
    # assuming only training player 0
    done = False
    
    while not done:
        if debug:
            env.game.debug_state()

        if view:
            env.game.view_state()
        
        # last new_state becomes current_state
        current_state = new_state
        
        # get actions
        for pid in players:
            actions[pid], Qs[pid] = players[pid].get_action( current_state[pid] )
        
        # TODO, find out what info is?
        new_state, reward, done, info = env.step(actions)

    s.updateStats(reward, game+1)
    print(f'Game {game+1}')
    s.showAbsWinRate()
    s.showRelWinRate()


