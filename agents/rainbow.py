# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Hyperparameters
LEARNING_RATE = 0.000_69
epsilon = 0.99
EPSILON_DECAY = 0.000_25
NUM_EPISODES = 5
TARGET_UPDATE = 10
ACTION_SPACE = 132
STATE_SPACE = 105


class rainbow:
    def __init__(self):
        self.policy = policy_network()
        self.name = "Rainbow Network"
        

# our magical network 
class policy_network(nn.module):
    
    def __init__(self):
        super(Policy, self).__init__()
        
        # fc layers to outputs
        # TODO get the proper observations space and flatten it
        self.fc0 = nn.Linear(STATE_SPACE, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, ACTION_SPACE)
        
    def forward(self, x):
        
        x = torch.flatten(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  
    
