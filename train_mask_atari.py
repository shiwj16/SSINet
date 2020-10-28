import os
import gym
import cv2
import torch
import random
import numpy as np

from hyperdash import Experiment
from torch.utils.data import DataLoader
from envs_utils.create_env import create_single_env
from mask.dataset import MyDataset
from mask.trainer import MaskAgent
from src.ppo_atari import PPO
from src.args import get_ppo_args
from src.utils import seed, evaluate_policy, show_mask

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


args = get_ppo_args()
exp = Experiment("[Atari] - {}".format(args.algo))
pretrain_dir = './logs/{}'.format(args.dir_name)
if not os.path.exists(pretrain_dir):
    raise ValueError('Pretrain path <%s> does not exist!' % pretrain_dir)
else:
    model_dir = pretrain_dir+"/model"
    data_dir = pretrain_dir+"/data"
    mask_dir = pretrain_dir+"/mask"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

# set seeds
seed(args.seed)

# start to create the environment
env = create_single_env(args, None)

# Initialize pretrained policy
pretrained_agent = PPO(env.action_space.n, args)

# load pretrained model
print('Load pretrained models')
pretrained_agent.load("best", model_dir)

# Evaluate pretrained policy
evaluations= [evaluate_policy(env, pretrained_agent, ppo_agent=True)]
exp.metric("rewards", evaluations[0])

# Initialize policy with attention
agent = MaskAgent(pretrained_agent.actor, env.action_space, args)

# Evaluate initial policy with attention
evaluations.append(evaluate_policy(env, agent))
exp.metric("rewards", evaluations[-1])
np.savez(data_dir+'/mask_rewards.npz', evaluations)

# sampling state-action pairs with the pretrained policy
stateset, actionset, rgbobs_set = [], [], []
print('Rollout with the pretrained policy')
num_samples = 0
while num_samples < 50000:
    epi_timestep = 0
    obs = env.reset()
    rgb_obs = None
    done =False
    while not done:
        # prepocessing the state for the task where there are some black features.
        obs += 1*np.ones(env.observation_space.shape, dtype=env.observation_space.dtype.name)
        # predict
        act_softmax, act = pretrained_agent.predict(obs, is_training=False, training_mask=True)
        # store the transition
        stateset.append(obs)
        actionset.append(act_softmax)
        rgbobs_set.append(rgb_obs)
        # env step
        obs, _, done, rgb_obs = env.step(act)
        epi_timestep += 1
        if epi_timestep >= 50000:
            break
    num_samples += epi_timestep
    print('Number of state-action pairs: %d' % num_samples)

# build the trainset for training the decode net
train_data = MyDataset(stateset, actionset)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)

# train the decode net
best_eval_rew = -np.float("Inf")
losses = []
for epoch in range(3):
    for i, batch in enumerate(train_loader):
        agent.maskactor.train()
        agent.maskactor.actor.eval()

        loss = agent.train(batch)
        
        if (i+1) % 50 == 0 and (i+1)<1000:
            losses.append(loss)
            print(("Epoch: %d, Batch: %d, Loss: %f") % (epoch, i+1, loss))
            
            for j, (obs, rgb_obs) in enumerate(zip(stateset[50:951:50], rgbobs_set[50:951:50])):
                
                # if epoch == 0 and (i+1) == 50 and j == 0:
                    # original_obs = np.array(obs*255, dtype=np.uint8).transpose(1, 2, 0)
                    # cv2.imwrite(mask_dir+'/{}_0_000_obs.png'.format(j+1), original_obs[:, :, -1])
                rgb_obs = np.array(rgb_obs*255, dtype=np.uint8)
                cv2.imwrite(mask_dir+'/{}_0_000_obs.png'.format(j+1), rgb_obs[:, :, ::-1])
                
                _, mask = agent.predict(np.array(obs), is_training=False, mask_state=True)
                # heatmap, masked_obs = show_mask(obs[0], mask[0])
                heatmap, masked_obs = show_mask(rgb_obs, mask[0])
                cv2.imwrite(mask_dir+'/{}_{}_{}_heatmap.png'.format(j+1, epoch, i+1), heatmap)
                cv2.imwrite(mask_dir+'/{}_{}_{}_masked_obs.png'.format(j+1, epoch, i+1), masked_obs)
    
    # save the loss
    np.savez(data_dir+'/mask_loss.npz', losses)
    
    # evaluation with trained decode
    evaluations.append(evaluate_policy(env, agent))
    exp.metric("rewards", evaluations[-1])
    np.savez(data_dir+'/mask_rewards.npz', evaluations)
    
    # save the model
    agent.save("final_{}_mask".format(args.reg_scale), directory=model_dir)
    if evaluations[-1] > best_eval_rew:
        best_eval_rew = evaluations[-1]
        agent.save("best_{}_mask".format(args.reg_scale), directory=model_dir)

