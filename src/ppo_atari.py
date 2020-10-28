import os
import copy
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from .networks import ACUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, action_dims, args):
        self.clip = args.clip
        self.epoch = args.epoch
        self.ent_coef = args.ent_coef
        self.batch_size = args.batch_size
        self.vloss_coef = args.vloss_coef
        self.max_grad_norm = args.max_grad_norm
        # start to build the network.
        if args.actor_net_type == 'unet':
            self.actor = ACUnet(action_dims, None, args).to(device)
        else:
            raise NotImplementedError
        self.old_actor = copy.deepcopy(self.actor).to(device)
        # define the optimizer...
        self.optimizer = optim.Adam(self.actor.parameters(), args.lr, eps=args.eps)
    
    def predict(self, obs, is_training=False, training_mask=False):
        if is_training:
            self.actor.train()
        else:
            self.actor.eval()
            obs = np.expand_dims(obs, axis=0)
        with torch.no_grad():
            # get tensors
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            values, acts_logit = self.actor(obs_tensor)
            acts_softmax = F.softmax(acts_logit, dim=1)
        
        # select actions
        actions = Categorical(acts_softmax).sample()
        if training_mask:
            return acts_softmax.detach().cpu().numpy().squeeze(), actions.detach().cpu().numpy().squeeze()
        else:
            return values.detach().cpu().numpy().squeeze(), actions.detach().cpu().numpy().squeeze()

    # update the network
    def _update_network(self, obs, actions, returns, advantages):
        # before update the network, the old network will try to load the weights
        self.old_actor.load_state_dict(self.actor.state_dict())

        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.batch_size
        for _ in range(self.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                # convert minibatches to tensor
                mb_obs = torch.tensor(mb_obs, dtype=torch.float32).to(device)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32).to(device)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).to(device).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).to(device).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                # start to get values
                mb_values, logits = self.actor(mb_obs)
                pis = F.softmax(logits, dim=1)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_logits = self.old_actor(mb_obs)
                    old_pis = F.softmax(old_logits, dim=1)
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

    # convert the numpy array to tensors
    # def _get_tensors(self, obs):
    #     obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32).to(device)
    #     return obs_tensor

    def evaluate_actions(self, pi, actions):
        cate_dist = Categorical(pi)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        entropy = cate_dist.entropy().mean()
        return log_prob, entropy

    # adjust the learning rate
    def _adjust_learning_rate(self, init_lr, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = init_lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), directory+'/{}_ACNet.pth'.format(filename))
    
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(directory+'/{}_ACNet.pth'.format(filename), map_location=device))

 