
## Following this guide
## https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb

# Utility imports
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

# Main imports
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

# Segment tree (needs segment_tree.py!)
from segment_tree import MinSegmentTree, SumSegmentTree

# standard replay buffer
class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99
    ):
    
        self.obs_buf = np.zeros([size, obs_dim], dtype = np.float)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype = np.float)
        self.acts_buf = np.zeros([size], dtype = np.float)
        self.rews_buf = np.zeros([size], dtype = np.float)
        self.done_buf = np.zeros(size, dtype = np.float)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # N-step learning
        self.n_step_buffer = deque(maxlen = n_step)
        self.n_step = n_step
        self.gamma = gamma
        
    # Stores observations in the buffer
    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)
        
        # Check if single step transition is ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
            
        # Make an N-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs.cpu()
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]
        
    # Sample a batch of experiences for learning
    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size = self.batch_size, replace = False)

        return dict(
            obs = self.obs_buf[idxs],
            next_obs = self.next_obs_buf[idxs],
            acts = self.acts_buf[idxs],
            rews = self.rews_buf[idxs],
            done = self.done_buf[idxs],
            
            # For N-step learning
            indices = indices,
        )
    
    # Sample batch from indexes
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
    
        # For N-step learning
        return dict(
            obs = self.obs_buf[idxs],
            next_obs = self.next_obs_buf[idxs],
            acts = self.acts_buf[idxs],
            rews = self.rews_buf[idxs],
            done = self.done_buf[idxs],
        )
    
    # Get N-step information
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
    
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size
        
# Prioritized replay buffer 
class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
    
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # Segment tree (for priority sorting)
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    # Store an experience in the replay buffer
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    # Sample a batch of experiences for learning
    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs = obs,
            next_obs = next_obs,
            acts = acts,
            rews = rews,
            done = done,
            weights = weights,
            indices = indices,
        )
    
    # Update the priorites of sampled experiences
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    
    # Sample indices based on proportions 
    def _sample_proportional(self) -> List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    # Calculate the weight of the experience at the index
    def _calculate_weight(self, idx: int, beta: float):
        # Get the max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # Calculate the weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

# Noisy net
class NoisyLinear(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
    
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    # Reset the trainable network parameters
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )
    
    # Reset the noise
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.FloatTensor(np.random.normal(loc = 0.0, scale = 1.0, size = size))

        return x.sign().mul(x.abs().sqrt())

# DQN
class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
    
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # Advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # Value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim = 2)
        
        return q
    
    # Get distribution for atoms
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim = 1, keepdim = True)
        
        dist = F.softmax(q_atoms, dim = -1)
        dist = dist.clamp(min = 1e-3)
        
        return dist
    
    # Reset the noisy layers
    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# Rainbow agent
class rainbow:

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        
        # PER Parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        
        # Categorical DQN Parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        
        # N-Step Learning
        n_step: int = 3,
    ):
    
        ##obs_dim = env.observation_space.shape[0]
        
        ##action_dim = env.action_space.n
        
        ##action_dim = env.num_actions_per_turn
        
        obs_dim = 105
        
        action_dim = 132
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        
        # Determine device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha = alpha
        )
        
        # Memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step = n_step, gamma = gamma
            )
            
        # Categorical DQN Parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # Initialize networks

        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.dqn.parameters())

        # Transition memory
        self.transition = list()

    # Get action
    def get_action(self, state):
    
        
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        
        """Qs = self.dqn(
            torch.Tensor(state).to(self.device)
        )
        Qs = Qs.cpu().data.numpy()"""
        
        state = torch.Tensor(state)
        state = state.to(self.device)
        
        
        # get Qs and turn into np array
        Qs = self.dqn(state)
        
        Qs = Qs.cpu()
        Qs = Qs.data.numpy()
        
        actions = self.translateQs(Qs)
        
        if not self.is_test:
            self.transition = [state, actions]
        
        return actions

    # Take action and return env response
    def step(self, actions):
    
        next_state, reward, done, _ = self.env.step(actions)

        self.transition += [reward[0], next_state[0], done]
        
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
        # 1-step transition
        else:
            one_step_transition = self.transition

        # Add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    # Update model by gradient descent
    def update_model(self) -> torch.Tensor:
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # Importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-Step Learning loss
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: Importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # Reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
                
    # Return the categorical DQN loss
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        device = self.device 
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN Algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double Deep Q-network
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min = self.v_min, max = self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device = self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    # Hard update
    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
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
            
            # doesn't need to be incremented, causes errors
            # action[0] += 1
            # action[1] += 1		

            # check to see if the unit is already being moved
            if action[0] in units:
                continue	
            
            # add the unit to the unit chosen array
            units.append(action[0])		

            # append it to the action pair
            actionArray.append(action)
        
        # return the array
        return np.array(actionArray)



""" CURRENT AGENT ONLY COMPATIBLE WITH ACTION SPACE CONTAINING 1 ACTION, MUST BE COMPATIBLE WITH "MANY" ACTIONS """
