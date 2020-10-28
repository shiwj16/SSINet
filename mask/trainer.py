import functools
import operator
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from .networks import MaskUnet, MaskResnetLW, MaskMobilenetLW, MaskFCDensenet, MaskDeeplabv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskAgent(nn.Module):
    def __init__(self, pretrained_Actor, action_space, args):
        super(MaskAgent, self).__init__()
        self.reg_scale = args.reg_scale
        self.algo = args.algo
        self.actor_net_type = args.actor_net_type
        self.env_name = args.env_name

        if args.actor_net_type == "unet":
            self.maskactor = MaskUnet(pretrained_Actor, args.algo, self.env_name).to(device)
        elif args.actor_net_type == "ResnetLW":
            self.maskactor = MaskResnetLW(pretrained_Actor, args.algo).to(device)
        elif args.actor_net_type == "MobilenetLW":
            self.maskactor = MaskMobilenetLW(pretrained_Actor, args.algo).to(device)
        elif args.actor_net_type == "Deeplabv3":
            self.maskactor = MaskDeeplabv3(pretrained_Actor, args.algo).to(device)
        elif args.actor_net_type == "fcdensenet":
            maskactor = MaskFCDensenet(pretrained_Actor.module, args.algo)
            self.maskactor = nn.DataParallel(maskactor).to(device)
        else:
            raise NotImplementedError
        
        if self.env_name == "duckietown" and self.algo in ["sac", "ppo"]:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)
        
        # train the decode layers
        self.criterion = nn.MSELoss()
        self.regularizator = nn.L1Loss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.maskactor.parameters()), 
                                         lr=args.mask_lr, momentum=0.9, weight_decay=0.0005)
    
    def predict(self, state, is_training=False, mask_state=False):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        if self.env_name == "duckietown":
            assert state.shape[0] == 3
        if is_training:
            self.maskactor.train()
            if self.actor_net_type == "fcdensenet":
                self.maskactor.module.actor.eval()
            else:
                self.maskactor.actor.eval()
        else:
            self.maskactor.eval()
        
        state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        if self.algo == "td3":
            action, mask = self.maskactor(state)
        elif self.algo in ["sac", "ppo"]:
            if self.env_name == "duckietown":
                mean, mask = self.maskactor(state)
                temp_action = torch.tanh(mean)
                action = temp_action * self.action_scale + self.action_bias
            elif self.env_name == "atari":
                act_softmax, mask = self.maskactor(state)
                action = Categorical(act_softmax).sample()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        if mask_state:
            return action.detach().cpu().numpy().squeeze(), mask.cpu().data.numpy().squeeze()
        else:
            return action.detach().cpu().numpy().squeeze()
    
    def train(self, batch):
        states, label_actions = batch
        states = states.float().to(device)
        label_actions = label_actions.float().to(device)
        
        if self.algo == "td3":
            actions, masks = self.maskactor(states)
        elif self.algo in ["sac", "ppo"]:
            if self.env_name == "duckietown":
                means, masks = self.maskactor(states)
                temp_actions = torch.tanh(means)
                actions = temp_actions * self.action_scale + self.action_bias
            elif self.env_name == "atari":
                actions, masks = self.maskactor(states)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        label_masks = torch.zeros_like(masks).to(device)
        loss = self.criterion(actions, label_actions) + \
                self.reg_scale * self.regularizator(masks, label_masks)
        
        # Optimize the actor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def save(self, file_name, directory):
        torch.save(self.maskactor.state_dict(), directory+'/{}_actor.pth'.format(file_name))
    
    def load(self, file_name, directory):
        self.maskactor.load_state_dict(torch.load(directory+'/{}_actor.pth'.format(file_name), map_location=device))

