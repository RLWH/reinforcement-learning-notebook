"""
The Mountain Car Environment Wrapper
"""

import sys

import gym
import matplotlib
import matplotlib.pyplot as pyplot
import io
import base64

from gym import logger as gymlogger
from gym.wrappers import Monitor

gymlogger.set_level(20)

class Environment:

    def __init__(self, env, global_step=200, num_episodes=10000):
        """
        Args:
            env: The environment
        """

        self.env =  env
        self.global_step = global_step
        self.num_episodes = num_episodes

        print("Environment activated. ")
        print("Observation space: %s" % env.observation_space)
        print("Action space: %s" % env.action_space)

        print("Environment spec: ", env.env.spec)

    def play_game(self, policy=None):
        """
        Args:
            policy: The policy
        """

        if policy is None:
            # Play random game if policy is none
            observation = self.env.reset()

            for i in range(self.global_step):
                env.render()

                action = self.env.action_space.sample()
                observation, reward, done, info = env.step(action)

                if done:
                    break

            env.close()
