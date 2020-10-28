import os
import math
import torch
import numpy as np

from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from .networks import ACUnet, ACResnetLW, ACMobilenetLW, ACFCDensenet, ACDeeplabv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, state_dim, action_space, args):
        # args
        self.tau = args.tau
        self.clip = args.clip
        self.epoch = args.epoch
        self.gamma = args.gamma
        self.nsteps = args.nsteps
        self.ent_coef = args.ent_coef
        self.vloss_coef = args.vloss_coef
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm
        self.is_discrete = args.is_discrete
        
        if args.is_discrete:
            action_dim = action_space.n
            max_action = None
        else:
            action_dim = action_space.shape[0]
            max_action = float(action_space.high[0])
            # action rescaling
            if action_space is None:
                self.action_scale = torch.tensor(1.).to(device)
                self.action_bias = torch.tensor(0.).to(device)
            else:
                self.action_scale = torch.FloatTensor(
                    (action_space.high - action_space.low) / 2.).to(device)
                self.action_bias = torch.FloatTensor(
                    (action_space.high + action_space.low) / 2.).to(device)
        
        # start to build the network.
        if args.actor_net_type == "unet":
            self.actor = ACUnet(action_dim, max_action, args).to(device)
            self.old_actor = ACUnet(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "ResnetLW":
            self.actor = ACResnetLW(action_dim, max_action, args).to(device)
            self.old_actor = ACResnetLW(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "MobilenetLW":
            self.actor = ACMobilenetLW(action_dim, max_action, args).to(device)
            self.old_actor = ACMobilenetLW(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "Deeplabv3":
            self.actor = ACDeeplabv3(action_dim, max_action, args).to(device)
            self.old_actor = ACDeeplabv3(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "fcdensenet":
            actor = ACFCDensenet(action_dim, max_action, args)
            self.actor = torch.nn.DataParallel(actor).to(device)
            old_actor = ACFCDensenet(action_dim, max_action, args)
            self.old_actor = torch.nn.DataParallel(old_actor).to(device)
        else:
            raise NotImplementedError
        self.old_actor.load_state_dict(self.actor.state_dict())
        # define the optimizer...
        self.optimizer = optim.Adam(self.actor.parameters(), args.lr, eps=args.eps)
        # get batch_ob_shape
        self.batch_ob_shape = (args.num_workers * args.nsteps, ) + state_dim
    
    def predict(self, obs, is_training=False):
        # get tensors
        if is_training:
            self.actor.train()
        else:
            self.actor.eval()
            obs = np.expand_dims(obs, axis=0)
        tensor_obs = torch.FloatTensor(obs).to(device)
        value, pi = self.actor(tensor_obs)
        
        # sampling
        if self.is_discrete:
            action = Categorical(pi).sample()
            in_action = action
        else:
            mean, log_std = pi
            std = torch.exp(log_std)
            if is_training:
                action = Normal(mean, std).sample()
            else:
                action = mean
            
            temp_action = torch.tanh(action)
            in_action = temp_action * self.action_scale + self.action_bias
        
        return value.detach().cpu().numpy().squeeze(), \
               action.detach().cpu().numpy().squeeze(), \
               in_action.detach().cpu().numpy().squeeze()
    
    # start to train the network...
    def learn(self, update, mb_obs, mb_rewards, mb_actions, mb_masks, mb_values, last_obs, last_masks):
        # process the rollouts
        mb_obs = torch.FloatTensor(mb_obs).to(device)
        mb_rewards = torch.FloatTensor(mb_rewards).to(device)
        mb_actions = torch.FloatTensor(mb_actions).to(device)
        mb_values = torch.FloatTensor(mb_values).to(device)
        mb_masks = torch.FloatTensor(mb_masks).to(device)
        last_masks = torch.FloatTensor(last_masks).to(device)
        last_obs = torch.FloatTensor(last_obs).to(device)
        
        # compute the last state value
        with torch.no_grad():
            last_values, _ = self.actor(last_obs)
            last_values = last_values.detach().squeeze()
        
        # start to compute advantages...
        mb_returns = torch.zeros_like(mb_rewards).to(device)
        mb_advs = torch.zeros_like(mb_rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = last_masks
                nextvalues = last_values
            else:
                nextnonterminal = mb_masks[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        
        # after compute the returns, let's process the rollouts
        mb_obs = mb_obs.transpose(0, 1).reshape(self.batch_ob_shape)
        mb_actions = mb_actions.transpose(0, 1).reshape(self.batch_ob_shape[0], -1)
        mb_returns = mb_returns.transpose(0, 1).flatten()
        mb_advs = mb_advs.transpose(0, 1).flatten()
        
        # before update the network, the old network will try to load the weights
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # start to update the network
        self._update_network(update, mb_obs, mb_actions, mb_returns, mb_advs)
    
    # update the network
    def _update_network(self, update, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.batch_size
        for epoch in range(self.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                
                mb_returns = mb_returns.unsqueeze(1)
                mb_advs = mb_advs.unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                # start to get values
                mb_values, pis = self.actor(mb_obs)
                
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.old_actor(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = self.evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = self.evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # final total loss
                total_loss = policy_loss + self.vloss_coef * value_loss - ent_loss * self.ent_coef
                
                # clear the grad buffer
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # update
                self.optimizer.step()
    
    def evaluate_actions(self, pi, actions):
        if self.is_discrete:
            cate_dist = Categorical(pi)
            log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
            entropy = cate_dist.entropy().mean()
        else:
            mean, log_std = pi
            std = torch.exp(log_std)
            normal_dist = Normal(mean, std)
            log_prob = normal_dist.log_prob(actions)
            temp_actions = torch.tanh(actions)
            in_actions = temp_actions * self.action_scale + self.action_bias
            log_prob -= torch.log(self.action_scale * (1 - in_actions.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            entropy = normal_dist.entropy().mean()
        
        return log_prob, entropy
    
    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), directory+'/{}_ACNet.pth'.format(filename))
    
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(directory+'/{}_ACNet.pth'.format(filename), map_location=device))


