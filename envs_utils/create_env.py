import os
import gym
from envs_utils.atari_wrapper import make_atari, wrap_deepmind
from envs_utils.multi_envs_wrapper import SubprocVecEnv
from envs_utils.frame_stack import VecFrameStack

"""
this functions is to create the environments

"""

def create_single_env(args, video_dir, rank=0):
    # start to create environment
    env = make_atari(args.env_name)
    # the monitor
    if video_dir:
        env = gym.wrappers.Monitor(env, video_dir, force=True)
    # use the deepmind environment wrapper
    env = wrap_deepmind(env, frame_stack=True)
    
    # set seeds to the environment to make sure the reproducebility
    env.seed(args.seed + rank)
    return env

# create multiple environments - for multiple
def create_multiple_envs(args):
    def make_env(rank):
        def _thunk():
            env = make_atari(args.env_name)
            # set the seed for the environment
            env.seed(args.seed + rank)
            # use the deepmind environment wrapper
            env = wrap_deepmind(env)
            return env
        return _thunk
        # put into sub processing 
    envs = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])
    # then, frame stack
    envs = VecFrameStack(envs, 4)
    return envs

