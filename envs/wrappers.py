import gym
import cv2
import numpy as np

from gym import spaces
from collections import deque


class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(CropWrapper, self).__init__(env)

    def observation(self, observation):
        return observation[int(observation.shape[0]//3):,:,:]


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k 
        self.gray_scale = True
        self.frames = deque([], maxlen=k)
        obsp = self.observation_space
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(obsp.shape[:-1] + (obsp.shape[-1] * k,)), dtype=obsp.dtype)

    def reset(self):
        ob = self.env.reset()
        if self.gray_scale:
            ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        assert len(ob.shape) == 2       #(height, width)
        for _ in range(self.k):
            self.frames.append(ob)
        ob = np.array(list(self.frames)).transpose(1, 2, 0) 
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self.gray_scale:
            ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        assert len(ob.shape) == 2       #(height, width)
        self.frames.append(ob)
        ob = np.array(list(self.frames)).transpose(1, 2, 0)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(list(self.frames)).transpose(1, 2, 0)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


class StableRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StableRewardWrapper, self).__init__(env)

    def reset(self):
        self.action_old = None
        return self.env.reset()

    def step(self, action):
        action = np.array(action)
        if self.action_old is None:
            self.action_old = action
        ob, rew, done, misc = self.env.step(action)
        # add penalty for the change of action
        action_penalty = np.linalg.norm(action - self.action_old)
        new_rew = rew - 0.01 * action_penalty
        print("test for action change in rew:{}".format(action_penalty))
        return ob, new_rew, done, misc


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)


    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

class SoftActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SoftActionWrapper, self).__init__(env)
        self.tau = 0.9

    def reset(self):
        self.soft_action = None
        return self.env.reset()
    
    def step(self, action):
        if self.soft_action is None:
            self.soft_action = action
        else:
            self.soft_action = [a_old*self.tau + (1-self.tau)*a_new for a_old, a_new in zip(self.soft_action, action)]
        self.soft_action = np.clip(self.soft_action, self.action_space.low, self.action_space.high)
        return self.env.step(self.soft_action) 


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self,
                 env,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102
                 ):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist
        self.action_space = spaces.Box(low = np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float) 

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels


class DTPytorchWrapper():
    def __init__(self, shape=(120, 160, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])

    def preprocess(self, obs):
        from scipy.misc import imresize
        new_obs = imresize(obs, self.shape).transpose(2, 0, 1)
        new_obs = new_obs.astype(float) / 255.0
        return new_obs


class SteeringToWheelVelWrapperForTest():
    def __init__(self,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102):
        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist 
        
    def convert(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels
