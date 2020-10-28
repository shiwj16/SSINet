import os
import gym
import numpy as np
from gym import spaces
from scipy.misc import imresize

"""
the wrapper is taken from the openai baselines

"""

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(3, 84, 84)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        return imresize(observation[35:195], self.shape[1:]).astype(np.float32).reshape(self.shape)/255.

def make_atari(env_id):
    env = gym.make(env_id)
    env = ResizeWrapper(env)
    return env

