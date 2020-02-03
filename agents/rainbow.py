# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Other imports
from collections import deque

# Hyperparameters
LEARNING_RATE = 0.000_69
epsilon = 0.99
EPSILON_DECAY = 0.000_25
NUM_EPISODES = 5
TARGET_UPDATE = 10
ACTION_SPACE = 132
STATE_SPACE = 105
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 50
DISCOUNT = 0.99

class rainbow:
    def __init__(self):
        
        # make main model
        self.model = policy_network()
        
        # make target model
        self.target_model = policy_network()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # loss and optimiser
        self.opti = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = torch.nn.MSELoss()
        
        # replay memory buffer
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        # some variables for later
        self.name = "Rainbow Network"
        self.target_update_counter = 0
        
    # terminal_state is a bool, state is just the Qs
    def train(self, terminal_state, state):
        
        # test to make sure we have enought replay memory to do the training
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_state = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.forward(current_state)
        
        new_current_state = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.forward(new_current_state)
        
        # X and Y for optimising
        x = []
        y = []
        
        # go throw the mini-batch 
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward 
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            x.append(current_state)
            y.append(current_qs)
        
        # do optimise on the minibatch
        self.model.zerograd()
        pred_y = self.model.forward(x)
        loss = self.loss(pred_y, y)
        self.opti.backward()
        self.opti.step()
        
        # check to see if we need to update the target_model
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
    def getAction(self, state):
        
        # get Qs
        Qs = self.policy(state)
        
        # epsilon-greedy policy 
        if np.random(1) < epsilon:
            
            # random actions 
            actions = np.zeros(self.shape)
            actions[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            actions[:, 1] = np.random.choice(np.arange(1, self.num_nodes + 1), self.num_actions, replace=False)
        else:
            
            # get actions 
            actions = getActions(Q)
        
        return actions
    
    def update_replay_memory(transition):
        self.replay_memory.append(transition)
        
    
        
        
# our magical network 
class policy_network(nn.module):
    
    def __init__(self):
        super(Policy, self).__init__()
        
        # fc layers to outputs
        self.fc0 = nn.Linear(STATE_SPACE, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, ACTION_SPACE)
        
    def forward(self, x):
        
        x = torch.flatten(x)
        x = F.relu(self.fc0(x))
        # TODO, impliment noisy nets
        
        x = F.relu(self.fc1(x))
        
        # softmax for outputs
        x = F.softmax(self.fc2(x))
        return x  
    
