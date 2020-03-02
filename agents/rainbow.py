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
from .common.ReplayBuffer import ReplayBuffer
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
SAVE_MODEL_EVERY = 50
PATH = "./agents/savedModels/rainbow/rainbow_v1.weights"


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
        self.replay_memory = ReplayBuffer(STATE_SPACE, 
                                          ACTION_SPACE, 
                                          REPLAY_MEMORY_SIZE, 
                                          MINIBATCH_SIZE)
        
        # some variables for later
        # https://www.youtube.com/watch?v=a_Aej8hAVE4
        self.name = "Rainbow"
        self.target_update_counter = 0
        self.save_model_counter = 0
        self.epsilon = EPSILON
        
    # terminal_state is a bool, state is just the Qs
    def train(self):
        
        # test to make sure we have enought replay memory to do the training
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        sample = self.replay_memory.get_sample()
        
        # move sample to device for increase speed
        state = torch.FloatTensor(sample["init_state"]).to(self.device)
        next_state = torch.FloatTensor(sample["next_state"]).to(self.device)
        action = torch.LongTensor(sample["action"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(sample["reward"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(sample["done"].reshape(-1, 1)).to(self.device)

        # will be generated from our sample
        current_qs = torch.zeros([50,132]).to(self.device)
        expected_qs = torch.zeros([50,132]).to(self.device)
        
        # find the Qs and send to our device
        for i in range(MINIBATCH_SIZE):
            
            current_qs[i] = self.model.forward(state[i])
            
            expected_qs[i] = self.target_model(next_state[i])
        
        mask = 1 - done
        target = (reward + DISCOUNT * expected_qs * mask).to(self.device)
        
        
        # do optimise on the minibatch
        self.opti.zero_grad()
        loss = self.loss(current_qs, expected_qs)
        loss.backward()
        self.opti.step()
        
        # check to see if we need to update the target_model
        self.target_update_counter += 1
        self.save_model_counter += 1
        
        # this is for switching target and main model
        if self.target_update_counter > TARGET_UPDATE:
            tmp_model = self.target_model
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.load_state_dict(tmp_model.state_dict())
            
            self.target_update_counter = 0
            
        # this is saving the model 
        if self.save_model_counter > SAVE_MODEL_EVERY:
            self.saveModel()
            self.save_model_counter = 0
            
            
        # decay that epsilon
        self.epsilon -= EPSILON_DECAY
        
        
    def get_action(self, state):
        
        # move state from np to Tensor and to cuda
        state = torch.Tensor(state)
        state = state.to(self.device)
        
        # get Qs and turn into np array
        Qs = self.model(state)
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        # epsilon-greedy policy 
		# TODO, remove when noisy is implemented
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
        
        return actions, Qs
    
    def update_replay_memory(self, 
                            init_state,
                            next_state,
                            action,
                            reward,
                            done):
        
        self.replay_memory.store_transition(init_state,
                                            next_state,
                                            action,
                                            reward,
                                            done)
    
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
            

            # check to see if the unit is already being moved
            if action[0] in units:
                continue	
            
            # add the unit to the unit chosen array
            units.append(action[0])		

            # append it to the action pair
            actionArray.append(action)
        
        # return the array
        return np.array(actionArray)
        
    # pretty self explanitory 
    def saveModel(self):
        torch.save(self.model.state_dict(), PATH)
        
    def load_model(self):
        
        
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
    
