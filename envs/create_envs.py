import gym

from .env import launch_env
from .wrappers import *
from gym import wrappers


def create_test_env(seed, map_name):
    # only support for duckietown now
    env = launch_env(seed=seed, map_name=map_name)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
    env = SteeringToWheelVelWrapper(env)
    env = DtRewardWrapper(env)
    
    return env


def create_single_env(args, video_dir, rank=0):
    env = launch_env(seed=args.seed+rank, map_name=args.map_name)
    # Wrappers
    if video_dir:
        env = wrappers.Monitor(env, video_dir, force=True)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
    env = SteeringToWheelVelWrapper(env)
    env = DtRewardWrapper(env)
    
    return env


def make_env(rank, args):
    def _thunk():
        env = launch_env(seed=args.seed+rank, map_name=args.map_name)
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env)
        env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
        env = SteeringToWheelVelWrapper(env)
        env = DtRewardWrapper(env)
        return env
    
    return _thunk


# create multiple environments
def create_multiple_envs(args):
    # put into sub processing
    envs = [make_env(i, args)() for i in range(args.num_workers)]
    
    return envs

