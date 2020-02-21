## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb

import numpy as np
import random as r

from everglades_server import server

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/rainbow.py'
#agent1_file = 'agents/same_commands.py'
agent1_file = 'agents/random_actions.py'

config_dir = './config/'
map_file = config_dir + 'DemoMap.json'
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

debug = 1

view = 0

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

# parameters for rainbow agent
num_frames = 2000
memory_size = 1000
batch_size = 32
target_update = 100

players[0] = agent0_class(env, memory_size, batch_size, target_update)
names[0] = agent0_class.__name__

players[1] = agent1_class(env.num_actions_per_turn, 1)
names[1] = agent1_class.__name__

observations = env.reset(
        players = players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = output_dir,
        pnames = names,
        debug = debug,
        view = view,
        out = createOut
)

## TRAIN AGENT IN RAINBOW TEST SCRIPT

"""Train the agent."""
players[0].is_test = False

state = players[0].env.reset(
        players = players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = output_dir,
        pnames = names,
        debug = debug,
        view = view,
        out = createOut
)

update_cnt = 0
losses = []
scores = []
score = 0

actions = {}

for frame_idx in range(1, num_frames + 1):
    
    for pid in players:
        print(pid)
        actions[pid] = players[pid].get_action( state[pid] )
    
    next_state, reward, done = players[0].step(actions)

    state = next_state
    score += reward[0]
    
    # NoisyNet: removed decrease of epsilon
    
    # PER: increase beta
    fraction = min(frame_idx / num_frames, 1.0)
    players[0].beta = players[0].beta + fraction * (1.0 - players[0].beta)

    # if episode ends
    if done:
        state = players[0].env.reset(
            players = players,
            config_dir = config_dir,
            map_file = map_file,
            unit_file = unit_file,
            output_dir = output_dir,
            pnames = names,
            debug = debug,
            view = view,
            out = createOut
        )
        scores.append(score)
        score = 0

    # if training is ready
    if len(players[0].memory) >= players[0].batch_size:
        loss = players[0].update_model()
        losses.append(loss)
        update_cnt += 1
        
        # if hard update is needed
        if update_cnt % players[0].target_update == 0:
            players[0]._target_hard_update()

        
players[0].env.close()

actions = {}

## Game Loop
done = 0
while not done:
    if debug:
        env.game.debug_state()

    if view:
        env.game.view_state()

    for pid in players:
        actions[pid] = players[pid].get_action( observations[pid] )

    observations, reward, done, info = env.step(actions)

print(f"reward = {reward}")