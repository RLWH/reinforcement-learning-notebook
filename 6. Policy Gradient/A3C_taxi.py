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
import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from collections import deque, namedtuple
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

GLOBAL_STEP = 1000
NUM_PROCESSES = 4
ENTROPY_BETA = 1e-3

Experience = namedtuple("Experience",
                        ["state", "log_probs", "entropy", "actions", "rewards",
                         "dones", "next_states"])


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

# Some helper functions
def calculate_rewards(rewards, gamma=0.99):
    """Calculate the discounted rewards"""
    G = 0
    for r in reversed(rewards):
        G = gamma * G + r

    return G

def make_env():
    """
    Make environment
    """
    
    # Make environment
    env = gym.make("Taxi-v2")

    # Observation space and action space
    observation_space = env.observation_space.n
    action_space = env.action_space.n

    print("Observation space: %d" % observation_space)
    print("Action space: %d" % action_space)

    return env, observation_space, action_space

def copy_grad(from_net, to_net):
    """
    Copy the gradient from original network to target network
    https://github.com/muupan/async-rl/blob/master/copy_param.py
    Args:
        from_net (torch.nn.Module)
        to_net (torch.nn.Module)
    """
    # Extract the parameter names from the target net
    to_net_params = dict(to_net.named_parameters())

    # For each of the parameters from the from_net, extract the gradient
    # Note that the gradient exists only when the backward() operation is called
    for param_name, param in from_net.named_parameters():
        to_net_params[param_name].grad[:] = param.grad
    

class Worker:
    """
    Worker for A3C Agent
    A worker contains a copy of the networks, an environment class, and the logic for 
    interacting with the environment, and updating the global network

    The worker contains the following method. Mostly this is similar to A2C
    """

    def __init__(self, global_policy_network, global_value_network, env
                 global_counter, lock, learning_rate=1e-4):
        
        # Obtain global networks
        self.global_policy_network = global_policy_network
        self.global_value_network = global_value_network

        # Initialise thread-specific model
        self.thread_policy_network = PolicyNetwork(observation_space, action_space, fc1_units=128)
        self.thread_value_network = ValueNetwork(observation_space, fc1_units=128)

        # Get environment
        self.env = env

        # Copy the same parameters from the global network
        self.thread_policy_network.load_state_dict(global_policy_network.state_dict)
        self.thread_value_network.load_state_dict(global_value_network.state_dict)

        # Reset the thread-specific gradients
        self.optimiser = torch.optim.Adam(list(self.global_policy_network.parameters())
                                      + list(thread_value_network.parameters()), lr=learning_rate)


        # Some metrics to track
        self.total_reward = 0

        # Create class variable to store batches of experiences
        self.batches = []

    def act(self, state):
        """Act once base on the current state"""
        probs = self.thread_policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        return log_probs, action.item(), entropy

    def run(self)):
        """
        Roll out the steps
        
        Return:
            A batch (list) of experiences (list of Experience),
            Mean rewards (float)
        """

        rewards = []

        for b in range(1):

            experiences = []
            total_reward = 0
            state = torch.FloatTensor(env.reset())

            for i in range(GLOBAL_STEP):
                # For the first n steps, only generate and save the experience

                log_prob, action, entropy = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = torch.FloatTensor(next_state)

                exp = Experience(state, log_prob, entropy,
                                action, reward, next_state, done)
                
                experiences.append(exp)
                total_reward += reward

                if done:
                    rewards.append(total_reward)
                    break

                state = next_state
            
            self.batches.append(experiences)

        return np.mean(rewards)

    def learn(self):
        """
        Learning function, by backprop
        """

        value_losses = []
        actor_losses = []
        entropy_losses = []

        for b in self.batches:
            states, log_probs, entropys, actions, rewards, next_states, dones = zip(*b)

            # Calculate the discounted return
            returns = []

            # Make sure the n_step < len(rewards)
            n_steps = min(n_steps, len(rewards))

            # Convert rewards to discounted n_step returns
            for i in range(len(rewards - n_steps + 1)):
                returns.append(calculate_rewards(rewards[i:i + n_steps]))

            for i in range(len(states) - n_steps + 1):
                current_state_value = self.value_network(states[i])
                next_state_value = self.value_network(states[i + 1])

                target = returns[i] + gamma * next_state_value * (1 - dones[i])
                error = target - current_state_value.detach()

                value_loss = error ** 2
                actor_loss = -log_probs[i] * error.detach()

                value_losses.append(value_loss)
                actor_losses.append(actor_loss)
                entropy_losses.append(entropys[i])

        # Optimise
        sum_value_losses = torch.stack(value_losses).sum()
        sum_actor_losses = torch.stack(actor_losses).sum()
        sum_entropy = torch.stack(entropys).sum()

        # Run optimiser and obtain gradient
        self.optimiser.zero_grad()
        total_loss = sum_value_losses + sum_actor_losses + ENTROPY_BETA * sum_entropy

        total_loss.backward()


            


def train(global_policy_network, global_value_network, global_counter, lock):
    """
    The training process
    This is basically the wrapper only. It follows these steps:
    1. Instantiate the worker
    2. Instruct the worker to run once and generate experience
    3. Train the worker by backpropagation
    4. Update the global variable?
    """

    # Define an optimiser
    optimiser = torch.optim.Adam(list(thread_policy_network.parameters())
                                 + list(thread_value_network.parameters()), lr=learning_rate))

    # Make environment
    env, _, _ = make_env()

    # Instantiate a worker 
    worker = Worker(policy_network, value_network)

    # Run the worker to collect the experiences
    batches, mean_rewards = worker.run()


if __name__ == "__main__":

    # Make an environment, but just for getting observation space and action space
    _, observation_space, action_space = make_env()

    # Initialise globally shared model
    global_policy_network = PolicyNetwork(observation_space, action_space, fc1_units=128)
    global_value_network = ValueNetwork(observation_space, fc1_units=128)

    # Set multiprocessing
    num_processes = 4
    processes = []

    # Share the global parameters in multiprocessing
    global_policy_network.share_memory()
    global_value_network.share_memory()

    # Setup optimisers
    actor_optimiser = torch.optim.Adam(global_policy_network.parameters(), lr=1e-5)
    critic_optimiser = torch.optim.Adam(global_value_network.parameters(), lr=1e-3)

    # Start the processes

    global_counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []
    mp.set_start_method('spawn')
    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=train, 
                       args=(global_policy_network, global_value_network,
                             global_counter, lock))
        # First start the model across `NUM_PROCESSES` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Once the training is complete, do something here
    pass