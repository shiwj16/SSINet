import os
import torch
import numpy as np
import torch.nn.functional as F

from .networks import ActorDense, CriticDense, ActorCNN, CriticCNN, \
                      ActorUnet, CriticUnet, ActorResnetLW, ActorFCDensenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(TD3, self).__init__()
        assert args.actor_net_type in ["dense", "cnn", "unet", "resnetLW", "fcdensenet"]
        assert args.critic_net_type in ["dense", "cnn", "unet"]
        
        self.args = args
        self.state_dim = state_dim
        self.max_action = max_action
        
        if args.actor_net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        elif args.actor_net_type == "cnn":
            print("use cnn for actor")
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action, args).to(device)
            self.actor_target = ActorCNN(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "unet":
            print("use unet for actor")
            self.flat = False
            self.actor = ActorUnet(action_dim, max_action, args).to(device)
            self.actor_target = ActorUnet(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "resnetLW":
            print("use refinenet for actor")
            self.flat = False
            self.actor = ActorResnetLW(action_dim, max_action, args).to(device)
            self.actor_target = ActorResnetLW(action_dim, max_action, args).to(device)
        elif args.actor_net_type == "fcdensenet":
            print("use FCDensenet for actor")
            self.flat = False
            self.actor = ActorFCDensenet(action_dim, max_action, args).to(device)
            self.actor_target = ActorFCDensenet(action_dim, max_action, args).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        
        if args.critic_net_type == "dense":
            self.critic = CriticDense(state_dim, action_dim).to(device)
            self.critic_target = CriticDense(state_dim, action_dim).to(device)
        elif args.critic_net_type == "cnn":
            self.critic = CriticCNN(action_dim, args).to(device)
            self.critic_target = CriticCNN(action_dim, args).to(device)
        elif args.critic_net_type == "unet":
            self.critic = CriticUnet(action_dim, args).to(device)
            self.critic_target = CriticUnet(action_dim, args).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def predict(self, state, is_training=False):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3
        
        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            if is_training:
                self.actor.train()
            else:
                self.actor.eval()
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, 
              tau=0.001, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):
            
            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)
            
            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(sample["action"]).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()
            
            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), directory+'/{}_actor.pth'.format(filename))
        torch.save(self.critic.state_dict(), directory+'/{}_critic.pth'.format(filename))

    def load(self, filename, directory, for_inference=False):
        self.actor.load_state_dict(torch.load(directory+'/{}_actor.pth'.format(filename), map_location=device))
        self.critic.load_state_dict(torch.load(directory+'/{}_critic.pth'.format(filename), map_location=device))
        if for_inference:
            self.actor.eval()
            self.critic.eval()

