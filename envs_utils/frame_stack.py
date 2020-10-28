import numpy as np
from envs_utils import VecEnvWrapper
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        print(venv.num_envs)
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=1)
        for (i, done) in enumerate(dones):
            if done:
                self.stackedobs[i] = 0
        self.stackedobs[:, -obs.shape[1]:, ...] = obs
        return self.stackedobs, rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[:, -obs.shape[1]:, ...] = obs
        return self.stackedobs
        