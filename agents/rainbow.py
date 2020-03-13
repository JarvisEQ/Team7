# Don't @ me, it's halfway there
# done - Double-Q Learning
# TODO - Pioritized Replay
# done - Dueling Networks
# TODO - Multi-step Learning (add more extra steps onto transition list)
# done - Distributional RL
# donish - Nosiy Nets

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
import math as maths

# Hyperparameters
LEARNING_RATE = 0.000_065
TARGET_UPDATE = 10
MINIBATCH_SIZE = 50
DISCOUNT = 0.99


# Environment Specifics
STATE_SPACE = 105
ACTION_SPACE = 132


# Replay Memory parameters
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000


# this is were the model will be saved
PATH = "./agents/savedModels/rainbow/rainbow_v1.weights"


# initalise device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class rainbow:
    def __init__(self):
        
        # make main model
        self.model = policy_network()
        
        # make target model
        self.target_model = policy_network()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # send to device
        self.model.to(device)
        self.target_model.to(device)
        
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
        self.win_rate = 0 
        
    # terminal_state is a bool, state is just the Qs
    def train(self, win_rate):
        
        # test to make sure we have enought replay memory to do the training
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        sample = self.replay_memory.get_sample()
        
        # move sample to device for increase speed
        state = torch.FloatTensor(sample["init_state"]).to(device)
        next_state = torch.FloatTensor(sample["next_state"]).to(device)
        action = torch.LongTensor(sample["action"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(sample["reward"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(sample["done"].reshape(-1, 1)).to(device)

        # will be generated from our sample
        current_qs = torch.zeros([50,132]).to(device)
        expected_qs = torch.zeros([50,132]).to(device)
        
        # find the Qs and send to our device
        for i in range(MINIBATCH_SIZE):
            
            current_qs[i] = self.model.forward(state[i])
            
            expected_qs[i] = self.target_model(next_state[i])
        
        mask = 1 - done
        target = (reward + DISCOUNT * expected_qs * mask).to(device)
        
        
        # do optimise on the minibatch
        self.opti.zero_grad()
        loss = self.loss(current_qs, target)
        loss.backward()
        self.opti.step()
        
        # update noise
        self.model.reset_noise()
        self.target_model.reset_noise()
        
        # check to see if we need to update the target_model
        self.target_update_counter += 1
        
        # updating the target model if it's time
        if self.target_update_counter > TARGET_UPDATE:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
            
        # only save the model if it's the best model
        if win_rate > self.win_rate:
            self.saveModel()
            self.win_rate = win_rate
        
        
    def get_action(self, state):
        
        # move state from np to Tensor and to cuda
        state = torch.Tensor(state)
        state = state.to(device)
        
        # get Qs and turn into np array
        Qs = self.model(state)
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        # take the Qs and make them game-understandable
        actions = self.translateQs(Qs)
 
        print(Qs)
    
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
        torch.save({"model_weights": self.model.state_dict(), 
                   "win_rate": self.win_rate},
                   PATH)
    
    # grab a checkout weights, used mostly in testing 
    def load_model(self):
        return
        
        checkpoint = torch.load(PATH)
        
        self.model.load_state_dict(checkpoint["model_weights"])
        self.target_model.load_state_dict(checkpoint["model_weights"])
        self.epsilon = checkpoint["epsilon"]
        self.win_rate = checkpoint["win_rate"]
        
        print(f"Win rate from train is {self.win_rate}")
        
        return
    
    # this prints out helpful information for training
    def get_debug(self):
        print(f"best win rate = {self.win_rate}")
        return
    
    
# our magical network 
class policy_network(nn.Module):
    
    def __init__(self):
        super(policy_network, self).__init__()
       
        # feature layers, shared by advantage and value
        self.feat_0 = nn.Linear(STATE_SPACE, 1024)
        self.feat_1 = nn.Linear(1024, 1024)
        
        # advantage layers
        self.adv_0 = Noisy(1024, 1024)
        self.adv_1 = Noisy(1024, ACTION_SPACE)
        
        # value layers
        self.val_0 = Noisy(1024, 1024)
        self.val_1 = Noisy(1024, ACTION_SPACE)
        
    
        
    def forward(self, x):
        
        # forward on features 
        x = torch.flatten(x)
        x = F.relu(self.feat_0(x))
        x = F.relu(self.feat_1(x))
        
        # forward for advantage
        adv = F.relu(self.adv_0(x))
        adv = F.relu(self.adv_1(adv))
        
        # forward for value
        val = F.relu(self.val_0(x))
        val = F.relu(self.val_1(val))

        # combine advantage and value to find Qs
        x = val * adv - adv.mean(dim = -1, keepdim=True)
        x = torch.flatten(x)

        # softmax for outputs, distributional part
        x = F.softmax(x, dim = -1)
        
        return x
    
    # unique to NOISY NETS,
    # necessary so that model tries new things! 
    def reset_noise(self):
        
        self.adv_0.reset_noise()
        self.adv_1.reset_noise()
        self.val_0.reset_noise()
        self.val_1.reset_noise()
        
# trainable noisy layer
class Noisy(nn.Module):
    
        def __init__(self, in_features, out_features, std = 0.2):
            super(Noisy, self).__init__()
            
            # standard things 
            self.in_features = in_features 
            self.out_features = out_features
            self.std = std
            
            # weighted parameters
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
            self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
            
            # bias parameters
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer("bias_epsilon", torch.Tensor(out_features))
            
            
            # inital resets
            self.reset_params()
            self.reset_noise()
        
        
        # reset the trainable params
        def reset_params(self):
            
            mu_range = 1 / maths.sqrt(self.in_features)
            
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(
                self.std / maths.sqrt(self.in_features)
            )
            
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(
                self.std / maths.sqrt(self.out_features)
            )
            
        # the reset the noise that the network has 
        def reset_noise(self):
            
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            
            # outer product
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
                
        # calculate weights and biases
        # do linear operation
        def forward(self, x):
            
            return F.linear(
            x,
            self.weight_mu + (self.weight_sigma * self.weight_epsilon),
            self.bias_mu)
    
            
        # how we get our noise
        def scale_noise(self, size):
            x = torch.FloatTensor(np.random.normal(loc=0.0, scale=0.5, size=size))
            # x = x.sign().mul(x.abs().sqrt())
            return x
