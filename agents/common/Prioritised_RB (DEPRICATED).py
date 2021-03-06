import numpy as np
import random

# first part is the sum Tree data structure 
# the tree expects objects, so we can put whatever in it
class Sum_Tree:
    
    # where it all begins
    def __init__(self, capacity, observation_Space, action_Space):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        # State buffers
        self.init_state_buffy = np.zeros([size, observation_Space], dtype=np.float32)
        self.next_state_buffy = np.zeros([size, observation_Space], dtype=np.float32)
        
        # Action buffer
         self.action_buffy = np.zeros([size, action_Space], dtype=np.float32)
         
        # reward and done buffer
        self.reward_buffy = np.zeros([size], dtype=np.float32)
        self.done_buffy = np.zeros([size], dtype=np.float32)
        
        
        self.num_entries = 0
        self.pointer = 0
        
    # returns the value at the root node 
    def total(self):
        return self.tree[0]
    
    # Stores data in the tree 
    def add(self, 
            priority, 
            init_state,
            next_state,
            action,
            reward,
            done):
        
        # get index
        index = self.write + self.capacity - 1
        
        # Stores transition 
        self.init_state_buffy[self.pointer = init_state
        self.next_state_buffy[self.pointer] = next_state
        self.action_buffy[self.pointer] = action
        self.reward_buffy[self.pointer] = reward
        self.done_buffy[self.pointer] = done 
        
        # update the priority 
        self.update_priority(index, priority)
        
        self.pointer += 1 
        self.pointer = (self.pointer % self.capacity)
        
        # capacity is the max num of entries we can have
        if self.num_entries < self.capacity:
            self.num_entries += 1 
        
    # update the priority of our network 
    def update_priority(self, index, priority):
        
        delta = priority -self.tree[index]
        
        self.tree[index] = priority
        self.propagate(index, delta)
        
    
    # returns a sample from the tree 
    def get(self, sample):
        index = self.retrieve(0, sample)
        data_index = index - self.capacity + 1
        
        return (index, self.tree[index], self.data[data_index])
    
    
    # propagate the priority change through the tree 
    def propagate(self, i
    ndex, delta):
        
        parent = (index - 1) // 2 
        
        self.tree[parent] += delta
        
        # if we aren't at the root, then propagate again!
        if parent != 0:
            self.propagate(parent, delta)
        
    # grab them samples
    def retrieve(self, index, sample):
        
        left_node = 2 * index + 1
        right_node = left_node + 1
        
        if left_node >= len(self.tree):
            return index
        
        if sample <= self.tree[left_node]:
            return self.retrieve(left_node, sample)
        else:
            return self.retrieve(right_node, sample - self.tree[left_node])

# implementation of RB using the Sum_Tree            
class Prioritised_RB:
    
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment = 0.001
    
    def __init__(self, capacity):
        
        self.tree = Sum_Tree(capacity, observation_Space, action_Space)
        self.capacity = capacity
        
    # returns transitions from our RB
    def sample(self, num_transitions):
        
        batch = []
        indexes = []
        priorities = []
        
        segment = self.tree.total() / num_transitions
        
        for i in range(num_transitions):
            
            alpha = segment * i
            bravo = segment * (i + 1)
            
            sample = random.uniform(alpha, bravo)
            
            index, priority, data = self.tree.get(sample)
            
            batch.append(data)
            indexes.append(index)
            priorities.append(priority)
            
        sampling_probs = priorities / self.tree.total()
        weight = np.power(self.tree.num_entries * sampling_probs, -self.beta)
        weight /= weight.max()
            
        return batch, indexes, weight
    
    # add a transitions into our RB
    def add(self, error, sample):
        
        # convert to numpy
        error = error.cpu().detach().numpy()
        
        priority = self.get_priority(error)
        self.tree.add(priority, sample)
        
    # get the priority given some error
    def get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a 
    
    # update the priorities in our tree 
    def update(self, index, error):
        priority = self.get_priority(error)
        self.tree.update(index, priority)
        
    def get_len(self):
        return self.tree.num_entries
    

            
