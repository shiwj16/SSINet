import os
import gym
import cv2
import torch
import random
import numpy as np

from hyperdash import Experiment
from torch.utils.data import DataLoader
from envs.env import launch_env
from envs.wrappers import *
from mask.dataset import MyDataset
from mask.trainer import MaskAgent
from src.args import get_ppo_args, get_sac_args, get_td3_args
from src.ppo import PPO
from src.sac import SAC
from src.td3 import TD3
from src.utils import seed, evaluate_policy, show_mask

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


args = get_ppo_args()

exp = Experiment("[duckietown] - {}".format(args.algo))
model_dir = args.model_dir
data_dir = './logs/{}/data'.format(args.algo)
mask_dir = './logs/{}/mask'.format(args.algo)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# Launch the env with our helper function
env = launch_env(seed=args.seed, map_name=args.map_name)

# Wrappers
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
env = SteeringToWheelVelWrapper(env)
env = DtRewardWrapper(env)

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
act_space = env.action_space

# Initialize pretrained policy
if args.algo == "td3":
    pretrained_agent = TD3(state_dim, act_space.shape[0], float(act_space.high[0]), args)
    ppo_agent = False
elif args.algo == "sac":
    pretrained_agent = SAC(state_dim, act_space, args)
    ppo_agent = False
elif args.algo == "ppo":
    pretrained_agent = PPO(state_dim, act_space, args)
    ppo_agent = True

# load pretrained model
print('Load pretrained models')
pretrained_agent.load("best", model_dir)

# Evaluate pretrained policy
evaluations= [evaluate_policy(env, pretrained_agent, ppo_agent=ppo_agent)]
exp.metric("rewards", evaluations[0])

# Initialize policy with attention
agent = MaskAgent(pretrained_agent.actor, act_space, args)

# Evaluate initial policy with attention
evaluations.append(evaluate_policy(env, agent))
exp.metric("rewards", evaluations[-1])
np.savez(data_dir+'/mask_rewards_{}.npz'.format(args.reg_scale), evaluations)

# sampling state-action pairs with the pretrained policy
stateset, actionset = [], []
print('Rollout with the pretrained policy')
num_samples = 0
while num_samples < args.dataset_size:
    epi_timestep = 0
    obs = env.reset()
    while epi_timestep < args.env_timesteps:
        if args.algo == "ppo":
            _, _, act = pretrained_agent.predict(obs, is_training=False)
        else:
            act = pretrained_agent.predict(obs, is_training=False)
        # store the transition
        stateset.append(obs)
        actionset.append(act)
        # env step
        obs, rew, done, misc = env.step(act)
        epi_timestep += 1
    num_samples += epi_timestep
    print('Number of state-action pairs: %d' % num_samples)

# build the trainset for training the decode net
train_data = MyDataset(stateset, actionset)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)

# train the decode net
best_eval_rew = -np.float("Inf")
losses = []
for epoch in range(30):
    agent.maskactor.train()
    if args.actor_net_type == "fcdensenet":
        agent.maskactor.module.actor.eval()
    else:
        agent.maskactor.actor.eval()
    
    for i, batch in enumerate(train_loader):
        loss = agent.train(batch)
        
        if (i+1) % 500 == 0:
            losses.append(loss)
            print(("Epoch: %d, Batch: %d, Loss: %f") % (epoch, i+1, loss))
            
            for j, obs in enumerate(stateset[50:451:50]):
                
                if epoch == 0 and (i+1) == 500:
                    original_obs = np.array(obs*255, dtype=np.uint8).transpose(1, 2, 0)
                    original_obs = original_obs[:, :, ::-1]
                    cv2.imwrite(mask_dir+'/{}_0_000_obs.png'.format(j+1), original_obs)
                
                _, mask = agent.predict(np.array(obs), is_training=False, mask_state=True)
                heatmap, masked_obs = show_mask(obs, mask)
                cv2.imwrite(mask_dir+'/{}_{}_{}_heatmap.png'.format(j+1, epoch, i+1), heatmap)
                cv2.imwrite(mask_dir+'/{}_{}_{}_masked_obs.png'.format(j+1, epoch, i+1), masked_obs)
    
    # save the loss
    np.savez(data_dir+'/mask_loss_{}.npz'.format(args.reg_scale), losses)
    
    # evaluation with trained decode
    evaluations.append(evaluate_policy(env, agent))
    exp.metric("rewards", evaluations[-1])
    np.savez(data_dir+'/mask_rewards_{}.npz'.format(args.reg_scale), evaluations)
    
    # save the model
    agent.save("final_mask_{}".format(args.reg_scale), directory=model_dir)
    if evaluations[-1] > best_eval_rew:
        best_eval_rew = evaluations[-1]
        agent.save("best_mask_{}".format(args.reg_scale), directory=model_dir)

