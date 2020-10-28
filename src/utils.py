import os
import cv2
import gym
import torch
import random
import numpy as np
from six import iteritems
from datetime import datetime


def seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500, ppo_agent=False):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            if ppo_agent:
                _, _, action = policy.predict(np.array(obs), is_training=False)
            else:
                action = policy.predict(np.array(obs), is_training=False)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward


def evaluate_atari_policy(env, policy, eval_episodes=10, ppo_agent=False):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if ppo_agent:
                _, action = policy.predict(np.array(obs), is_training=False)
            else:
                action = policy.predict(np.array(obs), is_training=False)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


# show the mask
def show_mask(obs, mask, env_name="duckietown"):
    if env_name == "duckietown":
        obs = np.uint8(255*obs).transpose(1, 2, 0)
        obs = obs[:, :, ::-1]
    elif env_name == "atari":
        # for gray image
        # obs = cv2.cvtColor(np.uint8(255*obs), cv2.COLOR_GRAY2BGR)
        # for rgb image
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        obs = np.array(obs*255, dtype=np.uint8)
    else:
        raise NotImplementedError

    mask = np.tile(np.expand_dims(mask, axis=-1), (1,1,3))
    masked_obs = np.uint8(np.multiply(obs, mask))
    
    heatmap = cv2.applyColorMap(masked_obs, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(obs, 0.5, heatmap, 0.5, 0)
    
    return heatmap, masked_obs


def get_dirs(dir_type):
    # current_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = './logs/{}/model'.format(dir_type)
    data_dir = './logs/{}/data'.format(dir_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('Model directory: %s' % model_dir)
    print('Data directory: %s' % data_dir)
    
    return model_dir, data_dir


def write_arguments(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
