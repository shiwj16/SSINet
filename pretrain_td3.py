import os
import gym
import torch
import random
import numpy as np
import gym_duckietown

from hyperdash import Experiment
from envs.env import launch_env
from envs.wrappers import *
from src.args import get_td3_args
from src.td3 import TD3
from src.replaybuffer import ReplayBuffer
from src.utils import seed, evaluate_policy, get_dirs, write_arguments
from gym import wrappers

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


exp = Experiment("[duckietown] - td3")

args = get_td3_args()
model_dir, data_dir = get_dirs(args.dir_name)
write_arguments(args, os.path.dirname(data_dir) + '/arguments.txt')

# Launch the env with our helper function
env = launch_env(map_name=args.map_name)

# Wrappers
# env = wrappers.Monitor(env, data_dir + '/video/train/', force=True)
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
env = SteeringToWheelVelWrapper(env)
env = DtRewardWrapper(env)

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = TD3(state_dim, action_dim, max_action, args=args)

# Initialize replay buffer
replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
evaluations= [evaluate_policy(env, policy)]
exp.metric("rewards", evaluations[0])

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
best_eval_rew = -np.float("Inf")
while total_timesteps < args.max_timesteps:
    if done:
        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, 
                         args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
        
        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))
            exp.metric("rewards", evaluations[-1])
            np.savez(data_dir+'/rewards.npz', evaluations)
            
            if args.save_models:
                policy.save("final", directory=model_dir)
                if evaluations[-1] > best_eval_rew:
                    best_eval_rew = evaluations[-1]
                    policy.save("best", directory=model_dir)
        
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs), is_training=True)
        if args.expl_noise != 0:
            action = (action + np.random.normal(
                0,
                args.expl_noise,
                size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)
    
    # Perform action
    new_obs, reward, done, _ = env.step(action)
    if episode_timesteps+1 >= args.env_timesteps:
        done = True

    episode_reward += reward
    
    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, float(done))
    
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))
exp.metric("rewards", evaluations[-1])
np.savez(data_dir+'/rewards.npz', evaluations)

if args.save_models:
    policy.save("final", directory=model_dir)

exp.end()
