import os
import gym
import torch
import random
import threading
import numpy as np
import gym_duckietown

from hyperdash import Experiment
from envs.create_envs import create_multiple_envs, create_single_env
from src.args import get_ppo_args
from src.ppo import PPO
from src.utils import seed, evaluate_policy, get_dirs, write_arguments

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def envs_step(i, envs, in_actions, obs, masks, rewards):
    ob, reward, done, _ = envs[i].step(in_actions[i])
    obs[i] = ob
    masks[i] = 1.0-float(done)
    rewards[i] = reward


args = get_ppo_args()
exp = Experiment("[{}] - ppo".format(args.env_name))
model_dir, data_dir = get_dirs(args.dir_name)
write_arguments(args, os.path.dirname(data_dir) + '/arguments.txt')
# video_dir = data_dir + '/video/train/'

# building envs
envs = create_multiple_envs(args)
if args.env_name == "duckietown":
    eval_env = envs[0]
else:
    eval_env = create_single_env(args, None)

# Set seeds
seed(args.seed)

# create trainer
state_dim = eval_env.observation_space.shape
act_space = eval_env.action_space
agent = PPO(state_dim, act_space, args)

# Evaluate untrained policy
evaluations = [evaluate_policy(eval_env, agent, max_timesteps=args.env_timesteps, ppo_agent=True)]
exp.metric("rewards", evaluations[0])

# start to train the network...
best_eval_rew = -np.float("Inf")
num_updates = args.total_frames // (args.nsteps * args.num_workers)
for update in range(num_updates):
    # adjust the learning rate
    if args.lr_decay:
        lr_frac = 1 - (update / num_updates)
        adjust_lr = args.lr * lr_frac
        for param_group in agent.optimizer.param_groups:
             param_group['lr'] = adjust_lr
    
    # rollout
    mb_obs, mb_rewards, mb_actions, mb_masks, mb_values = [], [], [], [], []
    epi_rewards = []
    # initial the observation
    if args.env_name == "duckietown":
        obs = [envs[ind].reset() for ind in range(args.num_workers)]
    else:
        obs = envs.reset()
    masks = np.ones(args.num_workers, dtype=np.float32)
    for step in range(args.nsteps):
        # select actions
        values, actions, in_actions = agent.predict(obs, is_training=True)
        
        mb_obs.append(obs)
        mb_masks.append(masks)
        mb_values.append(values)
        mb_actions.append(actions)
        
        # start to excute the actions in the environment
        if args.env_name == "duckietown":
            # multi threading
            obs = [0] * args.num_workers
            masks = [0] * args.num_workers
            rewards = [0] * args.num_workers
            threads = []
            for i in range(args.num_workers):
                thread = threading.Thread(target=envs_step,
                        args=(i, envs, in_actions, obs, masks, rewards))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            
            # iteration
            # obs, masks, rewards = [], [], []
            # for ind in range(args.num_workers):
                # ob, reward, done, _ = envs[ind].step(in_actions[ind])
                # obs.append(ob)
                # masks.append(1.0-float(done))
                # rewards.append(reward)
        else:
            obs, rewards, dones, _ = envs.step(in_actions)
            masks = [1.0-float(done) for done in dones]
        
        mb_rewards.append(rewards)
        # process for displaying the rewards on the screen
        epi_rewards = np.sum(np.asarray(mb_rewards), axis=0)
    
    # start to learn
    agent.learn(update, mb_obs, mb_rewards, mb_actions, mb_masks, mb_values, obs, masks)
    
    # display the training information
    if update % args.display_interval == 0:
        print('Update: {} / {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}'.format(
              update, num_updates, epi_rewards.mean(), epi_rewards.min(), epi_rewards.max()))
    
    # Evaluate episode
    if update % args.eval_interval == 0:
        evaluations.append(evaluate_policy(eval_env, agent, max_timesteps=args.env_timesteps, ppo_agent=True))
        exp.metric("rewards", evaluations[-1])
        np.savez(data_dir+'/rewards.npz', evaluations)
        
        if args.save_models:
            agent.save("final", directory=model_dir)
            if evaluations[-1] > best_eval_rew:
                best_eval_rew = evaluations[-1]
                agent.save("best", directory=model_dir)

# Final evaluation
evaluations.append(evaluate_policy(eval_env, agent, max_timesteps=args.env_timesteps, ppo_agent=True))
exp.metric("rewards", evaluations[-1])
np.savez(data_dir+'/rewards.npz', evaluations)

if args.save_models:
    agent.save("final", directory=model_dir)

exp.end()
