"""
Solving Taxi by A3C

Algorithm: (From Paper)
```
Initialise thread step counter t = 1
repeat
    Reset gradients: dtheta = 0; dtheta_v = 0
    Synchronise thread-specific parameters theta' = theta, theta_v' = theta_v
    t_start = t
    Get state s_

    repeat
        Perform a_t according to policy pi(a_t|s_t; theta')
        Receive reward r_t and new_state s_t+1
        t = t + 1
        T = T + 1
    until terminal s_t or t - t_start == t_max

    R = V(s_t, theta'_v) for non-terminal state s_t else 0

    for i in {t-1, ..., t_start}
        R = r_i + gamma * R
        Backprop for theta'
        Backgrop for theta'_v
    
    Perform asynchronous update for theta using dtheta and of theta_v using dtheta_v

until T > T_max
```
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

GLOBAL_STEP = 1000


# Setup the network
class PolicyNetwork(nn.Module):
    """Policy Network"""

    def __init__(self, state_size, action_size, fc1_units=128):
        """
        Args:
            state_size (int)
            action_size (int)
            fc1_units (int)
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        """Forward Pass"""
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class ValueNetwork(nn.Module):
    """Value Network"""

    def __init__(self, state_size, fc1_units=128):
        """
        Args:
            state_size (int)
            fc1_units (int)
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 1)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def roll_out(agent, env):
    """Roll out the steps"""

    batches = []
    rewards = []

    for b in range(1):

        experiences = []
        total_reward = 0
        state = torch.FloatTensor(env.reset())

        for i in range(GLOBAL_STEP):
            # For the first n steps, only generate and save the experience

            log_prob, action, entropy


class Worker:
    """Worker for A3C Agent"""

    def __init__(self, policy_network, value_network):
        
        # 1. Reset the environment
        state = env.reset()

        # 


if __name__ == "__main__":

    # Make environment
    env = gym.make("Taxi-v2")

    # Observation space and action space
    observation_space = env.observation_space.n
    action_space = env.action_space.n

    print("Observation space: %d" % observation_space)
    print("Action space: %d" % action_space)

    # Initialise networks
    global_policy_network = PolicyNetwork(observation_space, action_space, fc1_units=128)
    global_value_network = ValueNetwork(observation_space, fc1_units=128)

    target_policy_network = PolicyNetwork(observation_space, action_space, fc1_units=128)
    target_policy_network.load_state_dict(global_policy_network.state_dict)

    target_value_network = ValueNetwork(observation_space, fc1_units=128)
    target_value_network.load_state_dict(global_value_network.state_dict)

    # Set multiprocessing
    num_processes = 4
    processes = []

    # Share the global parameters in multiprocessing
