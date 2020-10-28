import os
import gym
import cv2
import torch
import numpy as np

from envs.create_envs import create_single_env
from mask.trainer import MaskAgent
from src.args import get_ppo_args
from src.ppo import PPO
from src.sac import SAC
from src.td3 import TD3
from src.utils import show_mask

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


args = get_ppo_args()

model_dir = args.model_dir
mask_dir = './logs/{}/mask'.format(args.algo)
video_dir = './logs/{}/video/pretrain'.format(args.algo)
video_dir_mask = './logs/{}/video/mask'.format(args.algo)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(video_dir_mask):
    os.makedirs(video_dir_mask)

# Launch the env with our helper function
env = create_single_env(args, None)
# Initialize pretrained policy
state_dim = env.observation_space.shape
act_space = env.action_space
if args.algo == "td3":
    policy = TD3(state_dim, act_space.shape[0], float(act_space.high[0]), args)
elif args.algo == "sac":
    policy = SAC(state_dim, act_space, args)
elif args.algo == "ppo":
    policy = PPO(state_dim, act_space, args)
# load pretrained model
policy.load("best", directory=model_dir)


# render with pretrained policy
with torch.no_grad():
    i = 0
    obs = env.reset()
    done =False
    # env.render()
    while i < 1000:
        i += 1
        if args.algo == "ppo":
            _, _, in_action = policy.predict(obs, is_training=False)
        else:
            in_action = policy.predict(obs, is_training=False)
        obs, rew, done, misc = env.step(in_action)
        # env.render()
        if done:
            break


# render with masked policy
env = create_single_env(args, None)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(mask_dir+'/heatmap_{}.mp4'.format(args.map_name), fourcc, 40, (160, 120))
# Initialize mask policies
policy = MaskAgent(policy.actor, env.action_space, args)
policy.load("best_mask", directory=model_dir)

# show the mask
with torch.no_grad():
    i = 0
    obs = env.reset()
    while i < 1000:
        i += 1
        action, mask = policy.predict(np.array(obs), is_training=False, mask_state=True)
        heatmap, masked_obs = show_mask(obs, mask)
        writer.write(heatmap)
        if i % 25 == 0:
            original_obs = np.uint8(255*obs).transpose(1, 2, 0)
            cv2.imwrite(mask_dir+'/{}_obs.png'.format(i), original_obs[:, :, ::-1])
            cv2.imwrite(mask_dir+'/{}_mask.png'.format(i), np.uint8(255*mask))
            cv2.imwrite(mask_dir+'/{}_masked_obs.png'.format(i), masked_obs)
            cv2.imwrite(mask_dir+'/{}_heatmap.png'.format(i), heatmap)
        # step
        obs, rew, done, misc = env.step(action)
        if done:
            break
    writer.release()
    # cv2.destroyAllWindows()
