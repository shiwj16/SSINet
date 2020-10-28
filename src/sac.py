import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.distributions import Normal
from .networks import ActorGauCNN, CriticCNN, ValueCNN, ActorGauUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Soft Actor-Critic (SAC)


class SAC(object):
    def __init__(self, state_dim, action_space, args):
        super(SAC, self).__init__()
        assert args.actor_net_type in ["cnn", "unet"]
        assert args.critic_net_type in ["cnn"]
        
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])
        self.tau = args.tau
        self.alpha = args.alpha
        self.discount = args.discount
        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)
        
        if args.actor_net_type == "cnn":
            print("use cnn for actor")
            self.flat = False
            self.actor = ActorGauCNN(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "unet":
            print("use unet for actor")
            self.flat = False
            self.actor = ActorGauUnet(action_dim, max_action, args).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        
        if args.critic_net_type == "cnn":
            self.critic = CriticCNN(action_dim, args).to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        self.value = ValueCNN(action_dim, args).to(device)
        self.value_target = ValueCNN(action_dim, args).to(device)
        
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=args.critic_lr)
    
    def gau2act(self, mean_std):
        # for reparameterization trick (mean + std * N(0,1))
        mean, log_std = mean_std
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        
        # Enforcing Action Bound
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_act = torch.tanh(mean)
        mean_act = mean_act * self.action_scale + self.action_bias
        
        return action, log_prob, mean_act, log_std
    
    def predict(self, state, is_training=False):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3
        
        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        
        if is_training:
            self.actor.train()
            action, _, _, _ = self.gau2act(self.actor(state))
        else:
            self.actor.eval()
            _, _, action, _ = self.gau2act(self.actor(state))
        
        return action.detach().cpu().numpy().flatten()
    
    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer
        sample = replay_buffer.sample(batch_size, flat=self.flat)
        state = torch.FloatTensor(sample["state"]).to(device)
        action = torch.FloatTensor(sample["action"]).to(device)
        next_state = torch.FloatTensor(sample["next_state"]).to(device)
        done = torch.FloatTensor(1 - sample["done"]).to(device)
        reward = torch.FloatTensor(sample["reward"]).to(device)
        
        """
        Soft Q-function parameters can be trained to minimize the soft Bellman residual
        """
        target_value = self.value_target(next_state)
        next_q_value = reward + done * self.discount * target_value.detach()
        
        new_action, log_prob, mean, log_std = self.gau2act(self.actor(state))
        expected_q1, expected_q2 = self.critic(state, action)
        q1_value_loss = F.mse_loss(expected_q1, next_q_value)
        q2_value_loss = F.mse_loss(expected_q2, next_q_value)
        
        """
        Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
        """
        q1_new, q2_new = self.critic(state, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)
        next_value = expected_new_q_value - (self.alpha * log_prob)
        
        expected_value = self.value(state)
        value_loss = F.mse_loss(expected_value, next_value.detach())
        
        """
        Reparameterization trick is used to get a low variance estimator
        """
        policy_loss = ((self.alpha * log_prob) - expected_new_q_value).mean()
        # Regularization Loss
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()
        
        policy_loss += mean_loss + std_loss
        
        """
        Optimizer
        """
        self.critic_optim.zero_grad()
        q1_value_loss.backward()
        self.critic_optim.step()
        
        self.critic_optim.zero_grad()
        q2_value_loss.backward()
        self.critic_optim.step()
        
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        # Update the frozen target models
        for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), directory+'/{}_actor.pth'.format(filename))
        torch.save(self.critic.state_dict(), directory+'/{}_critic.pth'.format(filename))
        torch.save(self.value.state_dict(), directory+'/{}_value.pth'.format(filename))
    
    def load(self, filename, directory, for_inference=False):
        self.actor.load_state_dict(torch.load(directory+'/{}_actor.pth'.format(filename), map_location=device))
        self.critic.load_state_dict(torch.load(directory+'/{}_critic.pth'.format(filename), map_location=device))
        self.value.load_state_dict(torch.load(directory+'/{}_value.pth'.format(filename), map_location=device))

