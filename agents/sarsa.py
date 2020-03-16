# Temporal Difference Learning
# Kyle and Leehe
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# other imports
import numpy as np
import random

# Hyperparameters
LEARNING_RATE = 0.000_3
EPSILON = 0.99
EPSILON_DECAY = 0.000_01
TARGET_UPDATE = 10
ACTION_SPACE = 132
STATE_SPACE = 105
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 50
DISCOUNT = 0.99
TAU = 1
GAMMA = 0.99
PATH = "./agents/savedModels/td/sarsa_v1.weights"
SEED = 123

class sarsa:
    """Actor Critic with temporal difference learning SARSA implementation"""

    def __init__(self):
        """
        Initialize the agent
        """

        # initalise device 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # main model
        self.Q_main = QNetwork(STATE_SPACE, ACTION_SPACE, SEED)

        # Target model
        self.Q_target = QNetwork(STATE_SPACE, ACTION_SPACE, SEED)
        # self.Q_target.load_state_dict(self.Q_main.state_dict()) figure out how this load is working??

        # send to device
        self.Q_main.to(self.device)
        self.Q_target.to(self.device)

        # loss and optimiser
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=LEARNING_RATE)
        self.loss = torch.nn.MSELoss()

        # other variables
        self.name = "sarsa"
        self.target_update_counter = 0
        self.epsilon = EPSILON
        self.win_rate = 0 


    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            where:
                s    = current state
                a    = action
                r    = reward
                s'   = new state
                done = 1 or 0 depending on if the game is done

            gamma (float): discount factor
        """
        print(f'learning...')
        states, actions, rewards, next_states, done = experiences

        # Get max predicted Q values (for next states) from target model
        # look up detach i think it means copy
        Q_target_next = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.Q_main(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        print(f'loss = {loss}')

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.Q_main, self.Q_target, TAU)

        self.epsilon -= EPSILON_DECAY


    def get_action(self, state):
        """
        Using an Epsilon Greedy strategy the action values are returned
        """
        # What the agent sees
        #print(f"State\n{state}")

        # move state from np to Tensor and to cuda
        state = torch.Tensor(state)
        state = state.to(self.device)
        
        # get Qs and turn into np array
        # why not a call to forward on line 73??
        Qs = self.Q_main(state)
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        # epsilon-greedy policy 
        if random.random() < self.epsilon:
            # random actions
            actions = np.zeros((7, 2))
            actions[:, 0] = np.random.choice(12, 7, replace=False)
            actions[:, 1] = np.random.choice(11, 7, replace=False)

        else:
            # translate the Qs to action pairs
            actions = self.translateQs(Qs)
        
        # make transition
        #print(f'actions in getaction\n {actions}')
        
        return actions, Qs


    # expects a numpy array of size 72 and state of size 105
    # have Liam explain this code
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
        torch.save({"model_weights": self.Q_main.state_dict(), 
                   "win_rate": self.win_rate,
                   "epsilon": self.epsilon},
                   PATH)
    
    # grab a checkout weights, used mostly in testing 
    def load_model(self):
        
        checkpoint = torch.load(PATH)
        
        self.Q_main.load_state_dict(checkpoint["model_weights"])
        self.Q_target.load_state_dict(checkpoint["model_weights"])
        self.epsilon = checkpoint["epsilon"]
        self.win_rate = checkpoint["win_rate"]
		
	    # get the model summary
        print(self.Q_main) 
        self.get_debug()
        
        return


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.

        Params:
            state_size  (int): Dimension of each state
            action_size (int): Dimension of each action
            seed        (int): Random seed
            fc1_units   (int): Number of nodes in first hidden layer
            fc2_units   (int): Number of nodes in second hidden layer

        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)