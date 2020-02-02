## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
from tqdm import tqdm 

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random as r

from everglades_server import server
from helpers.getActions import getActions

# Hyperparameters
LEARNING_RATE = 0.00069
epsilon = 0.99
EPSILON_DECAY = 0.00025
NUM_EPISODES = 5
TARGET_UPDATE = 10
ACTION_SPACE = 72
STATE_SPACE = 105


## Input Variables
config_dir = './config/'
map_file = config_dir + 'DemoMap.json'
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

# 0 for no debug /  1 for debug
debug = 0

view = 0

createOut = 0

## Main Script
env = gym.make('everglades-v0')

# defining policy
class Policy(nn.Module):
    
    def __init__(self):
        super(Policy, self).__init__()
        
        # fc layers to outputs
        # TODO get the proper observations space and flatten it
        self.fc0 = nn.Linear(STATE_SPACE, 128)
        self.fc1 = nn.Linear(128, 64)
		# 60 possible options for actions for our network
		# 12 units
		# 5 connections we can move to, 1 do nothing 
        self.fc2 = nn.Linear(64, ACTION_SPACE)
        
    def forward(self, x):
        
        x = torch.flatten(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# make the networks
policy = Policy()
target_policy = Policy()
target_policy.load_state_dict(policy.state_dict())

# loss and optimiser
loss = nn.MSELoss()
opti = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# training loop
for episode in tqdm(range(NUM_EPISODES), ascii=True, unit="episode"):
    
    observations = env.reset(
            players= ["dqn", "random"],        # TODO find out what to put here
            config_dir = config_dir,
            map_file = map_file,
            unit_file = unit_file,
            output_dir = output_dir,
            pnames = ["dqn", "random"],
            debug = debug,
            view = view,
            out = createOut
    )
    

    actions = {}

    # Game Loop
    # TODO, finish this
    done = 0
    while not done:

        # get Q from policy
        Q = policy(Variable(torch.from_numpy(observations[0]).type(torch.FloatTensor)))
        
        # epsilon-greedy action
        if np.random(1) < epsilon:
            
            # random actions 
            actionDQN = np.zeros(self.shape)
            actionDQN[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            actionDQN[:, 1] = np.random.choice(np.arange(1, self.num_nodes + 1), self.num_actions, replace=False)
        else:
            continue
            actionDQN = getActions(Q)
        
        # get random actions as our random agent
        actionRAN = np.zeros(self.shape)
        actionRAN[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
        actionRAN[:, 1] = np.random.choice(np.arange(1, self.num_nodes + 1), self.num_actions, replace=False)
        
        # wrap the actions up
        actions = [actionDQN, actionRAN]
        
        # do one step
        observations, reward, done, info = env.step(actions)
        
        # get next set of Qs
        Q_new = policy(Variable(torch.from_numpy(observations[0]).type(torch.FloatTensor)))
        
        # create target Qs
        
        # update policy 
        
        # end of episode 
        if done:
            
            # print reward for this episode
            print(f"reward = {reward}")
            
            # decay epsilon
            epsilon *= EPSILON_DECAY
            
            # copy weights to target if applicable
            if episode % TARGET_UPDATE:
                target_policy.load_state_dict(policy.state_dict())
            
            # end episode
            break

# TODO, test this to make sure that it works properly
# might be good to break this out into a different file as well 
# expects a numpy array of size 72 and state of size 105
def Q_to_Actions(Qs, state):
    
    actions_for_env = []
    
    # get 7 actions
    for index in range(7):
        
        # get the max Q
        action = np.argmax(Qs)
        
        # set that Q low so we don't choose it again
        Qs[action] = -420.0
        
        # get the unit we want to map to 
        unit = action/6
        to_node = action%6
        
        # get the node we want to look at
        state_index = (unit * 5) + 45
        node = state[state_index]
        
        # compare to map
        move_to = NODE_CONNECTIONS[(node*6) + to_node]
        
        # append it to the array 
        actions_for_env.append([unit, move_to])

    return actions_for_env
    
    
