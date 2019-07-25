"""
Actor Critic Algorithm - Sutton Book
```
Input: a differentiable policy parameterization pi(a|s, theta)                   [Policy Network]
Input: a differentiable state-value function parameterization Q_w(s, a, w)       [Value Network]
Parameters: step sizes alpha_theta > 0; alpha_w > 0
​
Loop forever for each episode:
​
        Initialise S, theta
        Sample a from policy network
        
        Loop while S is not terminal for each time step:
                A = pi(.|S, theta) [policy(state)]
                Take action A, observe S', R
                delta = R + gamma * A(S', A', w) - A(S, A, w)  [TD(0) error, or advantage]
                theta = theta + alpha_theta * grad_pi log pi_theta(s,a) A(S,A)     [policy gradient update]
                w = w + alpha_w * delta * x(s, a)    [TD(0)]
                A = A', S = S'
```
"""

import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import deque, namedtuple
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

GLOBAL_STEP = 200
BATCH_SIZE = 1
SCORE_REQUIREMENT = 198
NUM_EPISODES = 20000 
LEARNING_RATE = 5e-4
ENTROPY_BETA = 1e-3


Experience = namedtuple("Experience", 
                        ["state", "log_probs", "entropy", "actions", "rewards", "dones", "next_states"])

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
    """
    Actor Critic Agent
    Only cares getting the action from the current state
    """
    
    def __init__(self, env, n_state, n_action, policy_network):
        
        self.env = env
        self.policy_network = policy_network
    
    def act(self, state):
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        return log_probs, action.item(), entropy

def calculate_rewards(rewards, gamma=0.99):
    """Calculate the discounted rewards"""
    G = 0
    for r in reversed(rewards):
        G = gamma * G + r
    
    return G

def roll_out(agent, env):
    
    batches = []
    state = torch.FloatTensor(env.reset())
    rewards = []

    for b in range(BATCH_SIZE):

        experiences = []
        total_reward = 0
        state = torch.FloatTensor(env.reset())

        for i in range(GLOBAL_STEP):
            # For the first n steps,
            # only generate and save the experience

            log_prob, action, entropy = agent.act(state)
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

        batches.append(experiences)


    return batches, np.mean(rewards)

def learn(batches, value_network, optimiser, gamma=0.99, n_steps=10):
    """
    Learn from the experiences, do the backprob here
    """

    value_losses = []
    actor_losses = []
    entropy_losses = []

    for b in batches:
        states, log_probs, entropys, actions, rewards, next_states, dones = zip(*b)

        # Calculate the discounted return
        returns = []

        # Make sure the n_step < len(rewards)
        n_steps = min(n_steps, len(rewards))

        for i in range(len(rewards) - n_steps + 1):
            returns.append(calculate_rewards(rewards[i:i + n_steps]))

        for i in range(len(states) - n_steps + 1):
            current_state_value = value_network(states[i])
            next_state_value = value_network(states[i + 1])

            target = returns[i] + gamma * next_state_value * (1 - dones[i])
            error = target - current_state_value.detach()

            value_loss = error ** 2
            actor_loss = -log_probs[i] * error.detach()

            value_losses.append(value_loss)
            actor_losses.append(actor_loss)
            entropy_losses.append(entropys[i])
    
    # Optimize
    sum_value_losses = torch.stack(value_losses).sum()
    sum_actor_losses = torch.stack(actor_losses).sum()
    sum_entropy = torch.stack(entropys).sum()
    
    optimiser.zero_grad()
    total_loss = sum_value_losses + sum_actor_losses + ENTROPY_BETA * sum_entropy

    total_loss.backward()
    optimiser.step()

    return sum_value_losses, sum_actor_losses

def main(n_steps=10, writer=None):
    """
    Main running script
    """

    # Setup the environment
    env = gym.make("CartPole-v0")

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Initialise the networks
    policy_network = PolicyNetwork(observation_space, action_space, fc1_units=256)
    value_network = ValueNetwork(observation_space, fc1_units=256)
    optimiser = torch.optim.Adam(list(value_network.parameters())
                                 + list(policy_network.parameters()), lr=LEARNING_RATE)
    
    agent = A2CAgent(env, observation_space, action_space, policy_network)

    # Initialise the reward list
    ep_reward_list = []

    for ep in range(NUM_EPISODES):

        # Roll out and generate a batch of experiences
        batches, total_rewards = roll_out(agent, env)
        ep_reward_list.append(total_rewards)
        # Learn the experience
        value_losses, actor_losses = learn(batches, value_network, optimiser, n_steps=n_steps)

        # Write to tensorboard
        if writer is not None:
            writer.add_scalar('value_losses', value_losses, ep)
            writer.add_scalar('actor_losses', actor_losses, ep)
            writer.add_scalar('rewards', total_rewards, ep)

        # Print average of last 10 episodes if true
        if ep % 100 == 0 and ep != 0:
            avg_rewards = np.mean(ep_reward_list[-100:])
            print("\r{:d}-step A2C at Episode: {:d}, Avg Reward: {:.2f}".format(n_steps, ep, avg_rewards), end="")

            if avg_rewards >= 198:
                print("\nProblem solved @ Episode %d" % ep, end="")
                
                # Save Actor and critic weights
                torch.save(policy_network.state_dict(), "policy_network_weights_%s.pth" % n_steps)
                torch.save(value_network.state_dict(), "value_network_weights_%s.pth" % n_steps)
                break

    print(".....End Run")


    return ep_reward_list


if __name__ == "__main__":

    n_steps = [15, 20, 30, 40, 50]
    batch_size = [1, 5, 10, 20]
    reward_lists = []

    for n_step in n_steps:

        # Setup TensorBoard Summarywriter
        writer = SummaryWriter('runs/actor_critic/exp_nstep_%s' % n_step)

        reward_lists.append(main(n_steps=n_step, writer=writer))
