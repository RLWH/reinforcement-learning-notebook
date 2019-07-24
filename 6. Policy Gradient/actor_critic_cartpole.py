import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as pyplot

from collections import deque
from torch.distributions import Categorical

GLOBAL_STEP = 500
SCORE_REQUIREMENT = 198
NUM_EPISODES = 10000 

class PolicyNetwork(nn.Module):
    """
    Policy Network -> Update the policy gradient
    """
    
    def __init__(self, state_size, action_size, fc1_units=16):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        
    def forward(self, x):
        """
        Forward pass
        Essentially, the forward pass return the Q value
        """
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        
        return x


class ValueNetwork(nn.Module):
    """
    Policy Network
    """
    
    def __init__(self, state_size, fc1_units=16):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 1)
        
    def forward(self, x):
        """
        Forward pass
        Essentially, the forward pass return the Q value
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class A2CAgent:
    """Actor Critic Agent"""
    
    def __init__(self, n_state, n_action, policy_network):
        
        self.env = env
        
        self.n_state = n_state
        self.n_action = n_action
        
        # Initialise the model
        self.policy_network = policy_network
    
    def act(self, state):
#         state = state.float()
        probs = self.policy_network(state)
#         value = self.value_network(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        entropy = m.entropy()
#         policy.saved_log_probs.append(log_prob)

        return log_probs, action.item(), entropy

if __name__ == "__main__":
    pass
    
    