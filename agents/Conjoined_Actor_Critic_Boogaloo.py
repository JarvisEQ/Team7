# Conjoined Actor Critic Boogaloo
# thought about the idea, try to find it, don't think it's out There?
# It takes the ideas behind Dueling DQN with having different output-streams, one for advantage and value
# However, they are not conbined, and the training is done like a Actor-Critic model
# No idea how this is going to turn out

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Other imports
from .common.ReplayBuffer_AC import ReplayBuffer
import numpy as np
import random 

# Hyperparameters
LEARNING_RATE = 0.000_05
EPSILON = 0.99
EPSILON_DECAY = 0.000_005
EPSILON_MIN = 0.2
TARGET_UPDATE = 10
ACTION_SPACE = 132
STATE_SPACE = 105
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
PATH = "./agents/savedModels/rainbow/CAC_v1.weights"


class Conjoined_Actor_Critic_Boogaloo:
    def __init__(self):
        
        # initalise device 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # make model
        self.model = policy_network()
        
        # make target model
        self.target_model = policy_network()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # send to device
        self.model.to(self.device)
        self.target_model.to(self.device)
        
        # loss and optimiser
        self.opti = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_actor = torch.nn.MSELoss()
        self.loss_critic = torch.nn.MSELoss()
        
        # replay memory buffer
        self.replay_memory = ReplayBuffer(STATE_SPACE, 
                                          ACTION_SPACE, 
                                          REPLAY_MEMORY_SIZE, 
                                          MINIBATCH_SIZE)
        
        # some variables for later
        # It is a terrible name, I made this when I was high 
        self.name = "Conjoined_Actor_Critic_Boogaloo"
        self.epsilon = EPSILON
        self.win_rate = 0 
        self.target_update_counter = 0
        
    # terminal_state is a bool, state is just the Qs
    def train(self, current_win_rate, game_number):
        
        # test to make sure we have enought replay memory to do the training
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        sample = self.replay_memory.get_sample()
        
        # move sample to device for increase speed
        state = torch.FloatTensor(sample["init_state"]).to(self.device)
        next_state = torch.FloatTensor(sample["next_state"]).to(self.device)
        # action = torch.LongTensor(sample["action"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(sample["reward"].reshape(-1, 1)).to(self.device)
        reward_critic = torch.FloatTensor(sample["reward_critic"].reshape(-1, 1)).to(self.device)
        # done = torch.FloatTensor(sample["done"].reshape(-1, 1)).to(self.device)

        # will be generated from our sample
        current_qs = torch.zeros([MINIBATCH_SIZE, 132]).to(self.device)
        expected_qs = torch.zeros([MINIBATCH_SIZE, 132]).to(self.device)
        
        # find the Qs and send to our device
        for i in range(MINIBATCH_SIZE):
            
            current_qs[i] , reward_critic[i] = self.model(state[i])
            expected_qs[i] , _ = self.target_model(next_state[i])
            
        
        target = (reward_critic + DISCOUNT * expected_qs).to(self.device)
        
        
        # 
        self.opti.zero_grad()
        loss_actor = self.loss_actor(current_qs, target)
        loss_critic = self.loss_critic(reward_critic, reward)
        loss = loss_actor + loss_critic
        
        # do optimise on the minibatch
        loss.backward()
        self.opti.step()
        
        # check to see if we need to update the target_model
        self.target_update_counter += 1
        
        # updating the target model if it's time
        if self.target_update_counter > TARGET_UPDATE:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
            
        # only save the model if it's the best model
        if current_win_rate > self.win_rate:
            self.saveModel()
            self.win_rate = current_win_rate
            
        # decay that epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        
        
    def get_action(self, state):
        
        # move state from np to Tensor and to cuda
        state = torch.Tensor(state)
        state = state.to(self.device)
        
        # get Qs and advantage and turn into np array
        Qs, advantage = self.model(state)
        
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        advantage = advantage.cpu()
        advantage = advantage.data.numpy()
        
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
        
        return actions, Qs, advantage
    
    def update_replay_memory(self, 
                            init_state,
                            next_state,
                            action,
                            reward,
                            reward_critic,
                            done):
        
        self.replay_memory.store_transition(init_state,
                                            next_state,
                                            action,
                                            reward,
                                            reward_critic,
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
                   "win_rate": self.win_rate,
                   "epsilon": self.epsilon},
                   PATH)
    
    # grab a checkout weights, used mostly in testing 
    def load_model(self):
        
        checkpoint = torch.load(PATH)
        
        self.model.load_state_dict(checkpoint["model_weights"])
        self.target_model.load_state_dict(checkpoint["model_weights"])
        self.epsilon = checkpoint["epsilon"]
        self.win_rate = checkpoint["win_rate"]
		
	# get the model summary
        print(self.model) 
        self.get_debug()
        
        return
    
    # this prints out helpful information for training
    def get_debug(self):
        print(f"epsilon = {self.epsilon}")
        print(f"best win rate = {self.win_rate}")
        return
        
# our magical network 
class policy_network(nn.Module):
    
    def __init__(self):
        super(policy_network, self).__init__()
        
        # Conjoined part
        self.fc0 = nn.Linear(STATE_SPACE, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        
        # Actor Part
        self.Actor0 = nn.Linear(4096, 4096)
        self.Actor1 = nn.Linear(4096, ACTION_SPACE)
        
        # Critic Part
        self.Critic0 = nn.Linear(4096, 4096)
        self.Critic1 = nn.Linear(4096, 1)
        
    def forward(self, x):
        
        x = torch.flatten(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        
        action = F.relu(self.Actor0(x))
        action = self.Actor1(action)
        
        reward =  F.relu(self.Critic0(x))
        reward =  self.Critic1(reward)

        # softmax for action
        action = F.softmax(action, dim=-1)
        
        # Sigmoid for the Reward 
        reward = torch.sigmoid(reward)
        
        return action, reward
    
