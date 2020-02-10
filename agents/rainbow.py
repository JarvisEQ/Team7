# Don't @ me, it's halfway there
# done - Double-Q Learning
# done - Pioritized Replay
# TODO - Dueling Networks
# TODO - Multi-step Learning (add more extra steps onto transition list)
# done - Distributional RL
# TODO - Nosiy Nets

# with only half

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Other imports
from collections import deque
import numpy as np
import random 

# Hyperparameters
LEARNING_RATE = 0.000_69
EPSILON = 0.99
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
        
        # initalise device 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # make main model
        self.model = policy_network()
        
        # make target model
        self.target_model = policy_network()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # send to device
        self.model.to(self.device)
        self.target_model.to(self.device)
        
        # loss and optimiser
        self.opti = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = torch.nn.MSELoss()
        
        # replay memory buffer
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        # some variables for later
        self.name = "Rainbow Network"
        self.target_update_counter = 0
        self.epsilon = EPSILON
        
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
        
    def get_action(self, state):
        
        # move state from np to Tensor and to cuda
        state = torch.Tensor(state)
        state = state.to(self.device)
        
        # get Qs and turn into np array
        Qs = self.model(state)
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        # epsilon-greedy policy 
        if random.random() < self.epsilon:
            
            # random actions
            # hard-coding these in, epsilon-greedy is going to be undone in future rainbow version
            actions = np.zeros((7, 2))
            actions[:, 0] = np.random.choice(12, 7, replace=False)
            actions[:, 1] = np.random.choice(11, 7, replace=False)
        else:
            
            # translate the Qs to action pairs
            actions = self.translateQs(Qs)
        
        # make transition
        
        return actions
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # expects a numpy array of size 72 and state of size 105
    def translateQs(self, Qs):
    
        actionArray = []
        units = []	
        
        # reshaping the array makes life easier
        Qs = np.reshape(Qs, (12,11))

        while len(actionArray) < 7 :
        
            # get the max Q
            action = np.unravel_index(np.argmax(Qs, axis=None), Qs.shape)
            
            # set action low so that we do not chose it again
            Qs[action] = float('-inf')
            
            # convert from a tuple to a np array
            action = np.array(action)
            action[0] += 1
            action[1] += 1		

            # check to see if the unit is already being moved
            if action[0] in units:
                continue	
            
            # add the unit to the unit chosen array
            units.append(action[0])		

            # append it to the action pair
            actionArray.append(action)
        
        # return the array
        return actionArray
        
        
# our magical network 
# TODO, probably need to make deeper and thiccer for betting results!
class policy_network(nn.Module):
    
    def __init__(self):
        super(policy_network, self).__init__()
        
        # fc layers to outputs
        self.fc0 = nn.Linear(STATE_SPACE, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, ACTION_SPACE)
        
    def forward(self, x):
        
        x = torch.flatten(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        
        # softmax for outputs, need
        x = F.softmax(self.fc2(x))
        return x  
    
