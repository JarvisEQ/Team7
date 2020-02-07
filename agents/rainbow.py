# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch import optim

# other imports
import os
import numpy as np
import math
from __future__ import division

# Hyperparameters
LEARNING_RATE = 0.000_69
epsilon = 0.99
EPSILON_DECAY = 0.000_25
NUM_EPISODES = 5
TARGET_UPDATE = 10
ACTION_SPACE = 132
STATE_SPACE = 105

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# this probably should probably be in a constants file
NODE_CONNECTIONS = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 4, 5, 6, 7],
    4: [1, 3, 7],
    5: [2, 3, 8, 9],
    6: [3, 9],
    7: [3, 4, 9, 10],
    8: [5, 9, 11],
    9: [5, 6, 7, 8, 10],
    10: [7, 9, 11],
    11: [8, 10]
}

class RainbowAgent:
    def __init__(self):
        self.policy = policy_network()
        self.name = "Rainbow Network"
		
    def step(self, state, action, reward, next_state, done):
		# code for step
		
    def get_action(self, state, eps=0.):
		# get action for state
		with torch.no_grad():
			return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).items()
	
	def learn(self, experiences, gamma):
		# teaches the agent - WORK ON THIS PORTION
		
		# experience replay buffer must be refactored to work with
		idxs, states, actions, returns, next_states, nonterminals, weights = experiences.sample(self.batch_size)
		
		# calculate current state probabilities
		log_state_probabilities = self.online_net(states, log = True)
		log_state_probabilities_a = log_ps[range(self.batch_size), actions]
		
		# get nth next state probabilities 
		with torch.no_grad():
			# calculate probabilities
			prob_ns = self.online_net(next_states)
			
			# calculate distribution net
			dist_ns = self.support.expand_as(prob_ns) * prob_ns
			
			# argmax action selection by online network
			argmax_indices_ns = dist_ns.sum(2).argmax(1)
			
			# sample new target net noise
			self.target_net.reset_noise()
			
			# probabilities
			prob_target_ns = self.target_net(next_states)
			
			# double Q probabilities
			prob_target_ns_a = prob_target_ns[range(self.batch_size), argmax_indices_ns]
			
			# compute Tz
			Tz = returns.unsqueeze(1) + (nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0))
			Tz = Tz.clamp(min = self.Vmin, max = self.Vmax)
			
			# compute L2 projection of Tz onto fixed support z
			b = (Tz - self.Vmin) / self.delta_z
			l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
			
			# fix disappearing probability mass when l = b = u
			l[(u > 0) * (l == u)] -= 1
			u[(l < (self.atoms - 1)) * (l == u)] += 1
			
			# distribute probability of Tz
			m = states.new_zeros(self.batch_size, self.atoms)
			offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
			m.view(-1).index_add_(0, (l + offset).view(-1), (
			
			
	def soft_update(self, local_model, target_model, tau):
		# adsfasd
        
# Factorized Noisy Linear w/ bias
class NoisyLinear(nn.Module):
	def __init__(self, in_f, out_f):
	
	def reset_parameters(self):
	
	def _scale_noise(self, size):
	
	def reset_noise(self):
	
	def forward(self, input):
	

# Deep Q Network
class DQN(nn.Module):
	def __init__(self, args, action_space)::
		
	def forward(self, x, log = False):
		
	def reset_noise(self):
		