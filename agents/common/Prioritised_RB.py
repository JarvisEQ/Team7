from.Dank_Ass_Trees import MinSegmentTree, SumSegmentTree
import numpy as np
from typing import Dict, List, Tuple


class ReplayBuffer:
    
    def __init__(self, 
                 observation_Space, 
                 action_Space, 
                 size, 
                 batch):
         
         # State buffers
         self.init_state_buffy = np.zeros([size, observation_Space], dtype=np.float32)
         self.next_state_buffy = np.zeros([size, observation_Space], dtype=np.float32)
         
         # Action buffer
         self.action_buffy = np.zeros([size, action_Space], dtype=np.float32)
         
         # reward and done buffer
         self.reward_buffy = np.zeros([size], dtype=np.float32)
         self.done_buffy = np.zeros([size], dtype=np.float32)
         
         # boiler plate varibles
         self.size_max = size
         self.batch = batch
         self.pointer = 0
         self.size = 0
    
    # expects the data as np arrays
    def store_transition(self,
                         init_state,
                         next_state,
                         action,
                         reward,
                         done):
        
        # save the transition  
        self.init_state_buffy[self.pointer] = init_state
        self.next_state_buffy[self.pointer] = next_state
        self.action_buffy[self.pointer] = action
        self.reward_buffy[self.pointer] = reward
        self.done_buffy[self.pointer] = done 
        
        # move the pointer and increase size
        self.pointer = (self.pointer + 1) % self.size_max
        self.size = min(self.size + 1, self.size_max)
    
    # returns dictionary based on batch size
    def get_sample(self):
        
        indexes = np.random.choice(self.size, 
                                  size = self.batch,
                                  replace = False)
        
        sample = dict(init_state = self.init_state_buffy[indexes],
                     next_state = self.next_state_buffy[indexes],
                     action = self.action_buffy[indexes],
                     reward = self.reward_buffy[indexes],
                     done = self.done_buffy[indexes])
        
        return sample
    
    def __len__(self) -> int:
        return self.size

class Prioritised_RB(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
            self, 
            observation_Space, 
            action_Space, 
            size_max, 
            batch_size: int = 32, 
            alpha: float = 0.6):
        
        # negative alpha will not work in Prioritised_RB
        assert alpha >= 0
        
        # init the inheritance 
        super(Prioritised_RB, self).__init__(observation_Space, 
                                             action_Space, 
                                             size_max, 
                                             batch_size)
        
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.size_max:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store_transition(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store_transition(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.size_max

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
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
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
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
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
