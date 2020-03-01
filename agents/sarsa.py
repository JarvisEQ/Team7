# Temporal Difference Learning
# Kyle and Leehe

class sarsa:
    """Actor Critic with temporal difference learning SARSA implementation"""

    def __init__(self, state_size, n_actions, seed):
        """
        Initialize the agent

        Params:
            state_size  (int): Input state size. In the default case that is 105
            n_actions   (int): Output actions. In the default case that is a
                               7 x 2 array where 7 units can each be moved to
                               a node
            seed        (int): random seed

        """

        self.state_size = state_size
        self.n_actions = n_actions
        self.seed = random.seed(seed) # Look this up

        # Q-Network
        self.Q = QNetwork(state_size, n_actions, seed).to(device)


    def get_action(self, obs):
        # What the agent sees
        print(f"Observations\n{obs}")

        # Use policy to choose action (epsilon greedy)
        actions = epsilon_greedy(EPSILON, self.n_actions, obs)

        return actions

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            where:
                s = current state
                a = action
                r = reward
                s' = new state
                done = ?

            gamma (float): discount factor
        """
        print(f'learning...')
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # look up detach
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        print(f'loss = {loss}')

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    

    def epsilon_greedy(epsilon, state):
        """
        Epsilon greedy policy: this implimentation was found
                               here https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e

        Params:
        epsilon (float): for exploration
        n_actions (int): number of actions
        state     (int): state

        """

        # Exploit
        if np.random.rand() < epsilon:
            actions = self.Q.forward(state)
        # Explore
        else:
            actions = np.random.randint(0, self.n_actions) # might have to update for output space 60

        return actions

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

