import numpy as np
 
# either this or use queue for the replay buffer
# np is more efficient, thus will speed up training by a bit

class ReplayMemory:
    
    def __init__(self, observation_Space, action_Space, size, batch):
         
         # State buffers
         self.init_state_buffy = np.zeros([Size, observation_Space], dtype=np.float32)
         self.next_state_buffy = np.zeros([Size, observation_Space], dtype=np.float32)
         
         # Action buffer
         self.action_buffy = np.zeros([Size, action_Space], dtype=np.float32)
         
         # reward and done buffer
         self.reward_buffy = np.zeros([Size], dtype=np.float32)
         self.done_buffy = np.zeros([Size], dtype=np.float32)
         
         # boiler plate varibles
         self.size_max = size
         self.batch = batch
         self.pointer = 0
         self.size = 0
    
    # expects the data as np arrays
    def store_transition(self
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
    def get_sample():
        
        indexes = np.random.choice(self.size, 
                                  size = self.batch
                                  replacement = False)
        
        sample = dict(init_state = self.init_state_buffy[indexes],
                     next_state = self.next_state_buffy[indexes],
                     action = self.action_buffy[indexes],
                     reward = self.reward_buffy[indexes],
                     done = self.done_buffy[indexes])
        
        return sample
