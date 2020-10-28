import math
import functools
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_parts.unet_parts import *
from net_parts.fcdensenet_parts import *
from net_parts.deeplabv3_parts import ResNet18_OS8, ResNet18_OS16, ResNet34_OS8, ResNet34_OS16
from net_parts.refinenetLW_parts import conv1x1, conv3x3, CRPBlock, Bottleneck, convbnrelu, InvertedResidualBlock

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class ActorDense(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDense, self).__init__()
        
        state_dim = functools.reduce(operator.mul, state_dim, 1)
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        
        self.max_action = max_action
        
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * self.tanh(self.l3(x))
        return x


class CriticDense(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticDense, self).__init__()
        
        state_dim = functools.reduce(operator.mul, state_dim, 1)
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64 + action_dim, 64)
        self.l3 = nn.Linear(64, 1)
        
        self.l4 = nn.Linear(state_dim, 64)
        self.l5 = nn.Linear(64 + action_dim, 64)
        self.l6 = nn.Linear(64, 1)
    
    def forward(self, x, u):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(torch.cat([x1, u], 1)))
        x1 = self.l3(x1)
        
        x2 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(torch.cat([x2, u], 1)))
        x2 = self.l3(x2)
        
        return x1, x2
    
    def Q1(self, x, u):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(torch.cat([x1, u], 1)))
        x1 = self.l3(x1)
        
        return x1


class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorCNN, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 10 * 15
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
        self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
        
        if args.spec_init:
            print("use special initializer for actor")
            self.apply(weights_init)
            self.lin2.weight.data = normalized_columns_initializer(
                self.lin2.weight.data, 0.01)
            self.lin2.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.pool2(self.lr(self.conv2(x))))
        x = self.bn3(self.pool3(self.lr(self.conv3(x))))
        x = self.bn4(self.lr(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        
        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])
        x[:, 1] = self.tanh(x[:, 1])
        
        return x


class ActorGauCNN(ActorCNN):
    def __init__(self, action_dim, max_action, args):
        super(ActorGauCNN, self).__init__(action_dim, max_action, args)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        
        if args.spec_init:
            print("use special initializer for actor")
            self.apply(weights_init)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
            self.log_std_lin.weight.data = normalized_columns_initializer(
                self.log_std_lin.weight.data, 0.01)
            self.log_std_lin.bias.data.fill_(0)
    
    def forward(self, state):
        x = self.bn1(self.lr(self.conv1(state)))
        x = self.bn2(self.pool2(self.lr(self.conv2(x))))
        x = self.bn3(self.pool3(self.lr(self.conv3(x))))
        x = self.bn4(self.lr(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


class CriticCNN(nn.Module):
    def __init__(self, action_dim, args):
        super(CriticCNN, self).__init__()
        
        flat_size = 32 * 10 * 15
        self.lr = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv5 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv6 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv8 = nn.Conv2d(32, 32, 3, stride=1)
        
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)
        self.pool7 = nn.MaxPool2d(2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)
        
        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)
        self.lin4 = nn.Linear(flat_size, 256)
        self.lin5 = nn.Linear(256 + action_dim, 128)
        self.lin6 = nn.Linear(128, 1)
        
        if args.spec_init:
            print("use special initializer for critic")
            self.apply(weights_init)
            
            self.lin3.weight.data = normalized_columns_initializer(
                self.lin3.weight.data, 1.0)
            self.lin3.bias.data.fill_(0)
            
            self.lin6.weight.data = normalized_columns_initializer(
                self.lin6.weight.data, 1.0)
            self.lin6.bias.data.fill_(0)
    
    def forward(self, states, actions):
        x1 = self.Q1(states, actions)
        
        x2 = self.bn5(self.lr(self.conv5(states)))
        x2 = self.bn6(self.pool6(self.lr(self.conv6(x2))))
        x2 = self.bn7(self.pool7(self.lr(self.conv7(x2))))
        x2 = self.bn8(self.lr(self.conv8(x2)))
        
        x2 = x2.view(x2.size(0), -1)  # flatten
        x2 = self.lr(self.lin4(x2))
        x2 = self.lr(self.lin5(torch.cat([x2, actions], 1)))  # c
        x2 = self.lin6(x2)
        
        return x1, x2

    def Q1(self, states, actions):
        x1 = self.bn1(self.lr(self.conv1(states)))
        x1 = self.bn2(self.pool2(self.lr(self.conv2(x1))))
        x1 = self.bn3(self.pool3(self.lr(self.conv3(x1))))
        x1 = self.bn4(self.lr(self.conv4(x1)))
        
        x1 = x1.view(x1.size(0), -1)  # flatten
        x1 = self.lr(self.lin1(x1))
        x1 = self.lr(self.lin2(torch.cat([x1, actions], 1)))  # c
        x1 = self.lin3(x1)
        
        return x1


class ValueCNN(CriticCNN):
    def __init__(self, action_dim, args):
        super(ValueCNN, self).__init__(action_dim, args)
        
        self.lin2 = nn.Linear(256, 128)
    
    def forward(self, states):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.pool2(self.lr(self.conv2(x))))
        x = self.bn3(self.pool3(self.lr(self.conv3(x))))
        x = self.bn4(self.lr(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x))
        x = self.lin3(x)
        
        return x


class ActorUnet(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorUnet, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        self.inc = inconv(args.num_channel, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 256)
        self.down2_1 = down(64, 64)
        self.down3_1 = down(128, 128)
        self.down4_1 = down(256, 256)
        
        self.net_layers = args.net_layers
        if args.net_layers == 3:
            print("3 layers for unet")
            if args.env_name == "duckietown":
                flat_size = 64 * 15 * 20
            else:
                flat_size = 64 * 10 * 10
        elif args.net_layers == 4:
            print("4 layers for unet")
            if args.env_name == "duckietown":
                flat_size = 128 * 7 * 10
            else:
                flat_size = 128 * 5 * 5
        elif args.net_layers == 5:
            print("5 layers for unet")
            if args.env_name == "duckietown":
                flat_size = 256 * 3 * 5
            else:
                flat_size = 256 * 2 * 2
        else:
            raise ValueError("the number of unet layers: {} is not supported!".format(args.unet_layers))
        self.flat_size = flat_size
        
        self.dropoutlin = args.dropoutlin
        self.dropoutconv = args.dropoutconv
        self.drop = nn.Dropout(args.dropout_scale)
        
        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
            
            print("use noisy linear for actor")
            self.lin1 = NoisyLinear(flat_size, args.fc_hid_size, args.noisy_std)
            self.lin2 = NoisyLinear(args.fc_hid_size, action_dim, args.noisy_std)
        else:
            self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
            self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
            
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
                self.lin2.weight.data = normalized_columns_initializer(
                    self.lin2.weight.data, 0.01)
                self.lin2.bias.data.fill_(0)
    
    def forward(self, x):
        # encode
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)  # flatten
        
        if self.dropoutlin:
            x = self.drop(x)
        
        x = self.lr(self.lin1(x))
        act = self.lin2(x)
        
        act[:, 0] = self.max_action * self.sigm(act[:, 0])
        act[:, 1] = self.tanh(act[:, 1])
        
        return act
    
    def feature_extraction(self, state):
        # encode
        x = self.inc(state)
        x = self.down1(x)
        x = self.down2(x)
        if self.net_layers == 3:
            if self.dropoutconv:
                x = self.drop(x)
            x = self.down2_1(x)
            return x
        else:
            x = self.down3(x)
            if self.net_layers == 4:
                if self.dropoutconv:
                    x = self.drop(x)
                x = self.down3_1(x)
                return x
            elif self.net_layers == 5:
                x = self.down4(x)
                if self.dropoutconv:
                    x = self.drop(x)
                x = self.down4_1(x)
                return x


class ActorGauUnet(ActorUnet):
    def __init__(self, action_dim, max_action, args):
        super(ActorGauUnet, self).__init__(action_dim, max_action, args)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        
        if args.spec_init:
            print("use special initializer for Gaussian")
            self.apply(weights_init)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
            self.log_std_lin.weight.data = normalized_columns_initializer(
                self.log_std_lin.weight.data, 0.01)
            self.log_std_lin.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        
        x = x.view(x.size(0), -1)  # flatten
        if self.dropoutlin:
            x = self.drop(x)
        
        x = self.lr(self.lin1(x))
        mean = self.mean_lin(x)
        log_std = self.log_std_lin(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std


# for ppo only
class ACUnet(ActorGauUnet):
    def __init__(self, action_dim, max_action, args):
        super(ACUnet, self).__init__(action_dim, max_action, args)
        
        self.is_discrete = args.is_discrete

        self.value_lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
        self.value_lin2 = nn.Linear(args.fc_hid_size, 1)
        if not self.is_discrete:
            # self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
            self.log_std_lin = nn.Parameter(torch.zeros(1, action_dim))
        
        if args.spec_init:
            print("use special initializer for Value_Net")
            self.apply(weights_init)
            self.value_lin2.weight.data = normalized_columns_initializer(
                self.value_lin2.weight.data, 1.0)
            self.value_lin2.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        x = x.view(x.size(0), -1)  # flatten
        if self.dropoutlin:
            x = self.drop(x)
        
        # output the state value
        x1 = self.lr(self.value_lin1(x))
        state_value = self.value_lin2(x1)
        
        # output the policy...
        x2 = self.lr(self.lin1(x))
        if self.is_discrete:
            act_logit = self.lin2(x2)
            pi = act_logit
        else:
            mean = self.mean_lin(x2)
            # log_std = self.log_std_lin(x2)
            log_std = self.log_std_lin.expand_as(mean)
            # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            pi = (mean, log_std)
        
        return state_value, pi


class ActorResnetLW(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorResnetLW, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        self.flat_size = flat_size = 1024 * 4 * 5
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        layers = [3, 4, 6, 3] # [3, 4, 23, 3], [3, 8, 36, 3]
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 32, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, layers[3], stride=2)
        
        self.dropoutconv = args.dropoutconv
        self.do = nn.Dropout(args.dropout_scale)
        
        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
            
            print("use noisy linear for actor")
            self.lin1 = NoisyLinear(flat_size, args.fc_hid_size, args.noisy_std)
            self.lin2 = NoisyLinear(args.fc_hid_size, action_dim, args.noisy_std)
        else:
            self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
            self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
            
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
                
                self.lin2.weight.data = normalized_columns_initializer(
                    self.lin2.weight.data, 0.01)
                self.lin2.bias.data.fill_(0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # encode
        x = self.feature_extraction(x)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        act = self.lin2(x)
        
        act[:, 0] = self.max_action * self.sigm(act[:, 0])
        act[:, 1] = self.tanh(act[:, 1])
        
        return act
    
    def feature_extraction(self, state):
        # encode
        x = self.conv1(state)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        if self.dropoutconv:
            l4 = self.do(l4)
            l3 = self.do(l3)
        
        return l4


# for ppo only
class ACResnetLW(ActorResnetLW):
    def __init__(self, action_dim, max_action, args):
        super(ACResnetLW, self).__init__(action_dim, max_action, args)
        
        self.value_lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
        self.value_lin2 = nn.Linear(args.fc_hid_size, 1)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        # self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Parameter(torch.zeros(1, action_dim))
        
        if args.spec_init:
            print("use special initializer for Value_Net")
            self.apply(weights_init)
            self.value_lin2.weight.data = normalized_columns_initializer(
                self.value_lin2.weight.data, 1.0)
            self.value_lin2.bias.data.fill_(0)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        x = x.view(x.size(0), -1)  # flatten
        
        # output the state value
        x1 = self.lr(self.value_lin1(x))
        state_value = self.value_lin2(x1)
        
        # output the policy...
        x2 = self.lr(self.lin1(x))
        mean = self.mean_lin(x2)
        
        # log_std = self.log_std_lin(x2)
        log_std = self.log_std_lin.expand_as(mean)
        
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        pi = (mean, log_std)
        
        return state_value, pi


class ActorMobilenetLW(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorMobilenetLW, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        self.flat_size = flat_size = 320 * 4 * 5
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        """Net Definition"""
        mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                            [6, 24, 2, 2],
                            [6, 32, 3, 2],
                            [6, 64, 4, 2],
                            [6, 96, 3, 1],
                            [6, 160, 3, 2],
                            [6, 320, 1, 1],
                        ]
        in_planes = 32 # number of input channels
        self.layer1 = convbnrelu(3, in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(in_planes, c, expansion_factor=t, stride=s if idx == 0 else 1))
                in_planes = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1

        self._initialize_weights()

        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
            
            print("use noisy linear for actor")
            self.lin1 = NoisyLinear(flat_size, args.fc_hid_size, args.noisy_std)
            self.lin2 = NoisyLinear(args.fc_hid_size, action_dim, args.noisy_std)
        else:
            self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
            self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
            
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
                
                self.lin2.weight.data = normalized_columns_initializer(
                    self.lin2.weight.data, 0.01)
                self.lin2.bias.data.fill_(0)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # encode
        x = self.feature_extraction(x)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        act = self.lin2(x)
        
        act[:, 0] = self.max_action * self.sigm(act[:, 0])
        act[:, 1] = self.tanh(act[:, 1])
        
        return act
    
    def feature_extraction(self, state):
        x = self.layer1(state)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        
        return l8


# for ppo only
class ACMobilenetLW(ActorMobilenetLW):
    def __init__(self, action_dim, max_action, args):
        super(ACMobilenetLW, self).__init__(action_dim, max_action, args)
        
        self.value_lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
        self.value_lin2 = nn.Linear(args.fc_hid_size, 1)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        # self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Parameter(torch.zeros(1, action_dim))
        
        if args.spec_init:
            print("use special initializer for Value_Net")
            self.apply(weights_init)
            self.value_lin2.weight.data = normalized_columns_initializer(
                self.value_lin2.weight.data, 1.0)
            self.value_lin2.bias.data.fill_(0)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        x = x.view(x.size(0), -1)  # flatten
        
        # output the state value
        x1 = self.lr(self.value_lin1(x))
        state_value = self.value_lin2(x1)
        
        # output the policy...
        x2 = self.lr(self.lin1(x))
        mean = self.mean_lin(x2)
        
        # log_std = self.log_std_lin(x2)
        log_std = self.log_std_lin.expand_as(mean)
        
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        pi = (mean, log_std)
        
        return state_value, pi


class ActorFCDensenet(nn.Module):
    ### 103
    # down_blocks = [4,5,7,10,12]
    # up_blocks = [12,10,7,5,4]
    # growth_rate = 16
    # bottleneck_layers = 15
    ### 67
    down_blocks = [5,5,5,5,5]
    up_blocks = [5,5,5,5,5]
    growth_rate = 16
    bottleneck_layers = 5
    ### 57
    # down_blocks = [4,4,4,4,4]
    # up_blocks = [4,4,4,4,4]
    # growth_rate = 12
    # bottleneck_layers = 4
    def __init__(self, action_dim, max_action, args):
        super(ActorFCDensenet, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.n_pool = n_pool = args.net_layers
        down_blocks = self.down_blocks[:n_pool]
        if n_pool == 3:
            self.flat_size = 80 * 15 * 20
        elif n_pool == 4:
            self.flat_size = 80 * 7 * 10
        elif n_pool == 5:
            self.flat_size = 336 * 4 * 5
        else:
            raise ValueError("down layers: {}".format(n_pool))
        
        # First Convolution
        #added +2 in in_channels to have x and y coordinates
        self.add_module('firstconv', nn.Conv2d(in_channels=3, out_channels=48, 
                        kernel_size=3, stride=1, padding=1, bias=True))
        
        self.cur_channels_count = 48
        self.skip_connection_channel_counts = []
        # Downsampling path
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(self.cur_channels_count, self.growth_rate, down_blocks[i]))
            self.cur_channels_count += (self.growth_rate*down_blocks[i])
            self.skip_connection_channel_counts.insert(0, self.cur_channels_count)
            self.transDownBlocks.append(TransitionDown(self.cur_channels_count))
        
        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
            
            print("use noisy linear for actor")
            self.lin1 = NoisyLinear(self.flat_size, args.fc_hid_size, args.noisy_std)
            self.lin2 = NoisyLinear(args.fc_hid_size, action_dim, args.noisy_std)
        else:
            self.lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
            self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
            
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
                self.lin2.weight.data = normalized_columns_initializer(
                    self.lin2.weight.data, 0.01)
                self.lin2.bias.data.fill_(0)
    
    def forward(self, x):
        # encode
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)  # flatten
        
        x = self.lr(self.lin1(x))
        act = self.lin2(x)
        
        act[:, 0] = self.max_action * self.sigm(act[:, 0])
        act[:, 1] = self.tanh(act[:, 1])
        
        return act
    
    def feature_extraction(self, state):
        # encode
        out = self.firstconv(state)
        # Downsampling path
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            out = self.transDownBlocks[i](out)
        
        return out


# for ppo only
class ACFCDensenet(ActorFCDensenet):
    def __init__(self, action_dim, max_action, args):
        super(ACFCDensenet, self).__init__(action_dim, max_action, args)
        
        self.value_lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
        self.value_lin2 = nn.Linear(args.fc_hid_size, 1)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        # self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Parameter(torch.zeros(1, action_dim))
        
        if args.spec_init:
            print("use special initializer for Value_Net")
            self.apply(weights_init)
            self.value_lin2.weight.data = normalized_columns_initializer(
                self.value_lin2.weight.data, 1.0)
            self.value_lin2.bias.data.fill_(0)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        x = x.view(x.size(0), -1)  # flatten
        
        # output the state value
        x1 = self.lr(self.value_lin1(x))
        state_value = self.value_lin2(x1)
        
        # output the policy...
        x2 = self.lr(self.lin1(x))
        mean = self.mean_lin(x2)
        
        # log_std = self.log_std_lin(x2)
        log_std = self.log_std_lin.expand_as(mean)
        
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        pi = (mean, log_std)
        
        return state_value, pi


class ActorDeeplabv3(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorDeeplabv3, self).__init__()
        
        # ONLY TRU IN CASE OF DUCKIETOWN:
        self.flat_size = flat_size = 2048 * 4 * 5
        self.max_action = max_action
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        if args.net_layers == 3:
            self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        elif args.net_layers == 4:
            self.resnet = ResNet18_OS16()
        else:
            raise ValueError("net_layers is: {}".format(args.net_layers))
        
        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
            
            print("use noisy linear for actor")
            self.lin1 = NoisyLinear(flat_size, args.fc_hid_size, args.noisy_std)
            self.lin2 = NoisyLinear(args.fc_hid_size, action_dim, args.noisy_std)
        else:
            self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
            self.lin2 = nn.Linear(args.fc_hid_size, action_dim)
            
            if args.spec_init:
                print("use special initializer for actor")
                self.apply(weights_init)
                
                self.lin2.weight.data = normalized_columns_initializer(
                    self.lin2.weight.data, 0.01)
                self.lin2.bias.data.fill_(0)
    
    def forward(self, x):
        # encode
        x = self.feature_extraction(x)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        act = self.lin2(x)
        
        act[:, 0] = self.max_action * self.sigm(act[:, 0])
        act[:, 1] = self.tanh(act[:, 1])
        
        return act
    
    def feature_extraction(self, state):
        feature_map = self.resnet(state)
        return feature_map


# for ppo only
class ACDeeplabv3(ActorDeeplabv3):
    def __init__(self, action_dim, max_action, args):
        super(ACDeeplabv3, self).__init__(action_dim, max_action, args)
        
        self.value_lin1 = nn.Linear(self.flat_size, args.fc_hid_size)
        self.value_lin2 = nn.Linear(args.fc_hid_size, 1)
        
        self.mean_lin = nn.Linear(args.fc_hid_size, action_dim)
        # self.log_std_lin = nn.Linear(args.fc_hid_size, action_dim)
        self.log_std_lin = nn.Parameter(torch.zeros(1, action_dim))
        
        if args.spec_init:
            print("use special initializer for Value_Net")
            self.apply(weights_init)
            self.value_lin2.weight.data = normalized_columns_initializer(
                self.value_lin2.weight.data, 1.0)
            self.value_lin2.bias.data.fill_(0)
            self.mean_lin.weight.data = normalized_columns_initializer(
                self.mean_lin.weight.data, 0.01)
            self.mean_lin.bias.data.fill_(0)
    
    def forward(self, state):
        # encode
        x = self.feature_extraction(state)
        x = x.view(x.size(0), -1)  # flatten
        
        # output the state value
        x1 = self.lr(self.value_lin1(x))
        state_value = self.value_lin2(x1)
        
        # output the policy...
        x2 = self.lr(self.lin1(x))
        mean = self.mean_lin(x2)
        
        # log_std = self.log_std_lin(x2)
        log_std = self.log_std_lin.expand_as(mean)
        
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        pi = (mean, log_std)
        
        return state_value, pi


class CriticUnet(nn.Module):
    def __init__(self, action_dim, args):
        super(CriticUnet, self).__init__()
        
        flat_size = 128 * 7 * 10
        self.lr = nn.LeakyReLU()
        
        # Q1
        self.inc1 = inconv(3, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        # Q2
        self.inc2 = inconv(3, 16)
        self.down5 = down(16, 32)
        self.down6 = down(32, 64)
        self.down7 = down(64, 128)
        self.down8 = down(128, 128)
        
        if args.noisy_lin:
            if args.spec_init:
                print("use special initializer for critic")
                self.apply(weights_init)
            
            print("use noisy linear for critic")
            self.lin1 = NoisyLinear(flat_size, 512, args.noisy_std)
            self.lin2 = NoisyLinear(512 + action_dim, 256, args.noisy_std)
            self.lin3 = NoisyLinear(256, 1, args.noisy_std)
            self.lin4 = NoisyLinear(flat_size, 512, args.noisy_std)
            self.lin5 = NoisyLinear(512 + action_dim, 256, args.noisy_std)
            self.lin6 = NoisyLinear(256, 1, args.noisy_std)
        else:
            self.lin1 = nn.Linear(flat_size, 512)
            self.lin2 = nn.Linear(512 + action_dim, 256)
            self.lin3 = nn.Linear(256, 1)
            self.lin4 = nn.Linear(flat_size, 512)
            self.lin5 = nn.Linear(512 + action_dim, 256)
            self.lin6 = nn.Linear(256, 1)
            
            if args.spec_init:
                print("use special initializer for critic")
                self.apply(weights_init)
                
                self.lin3.weight.data = normalized_columns_initializer(
                    self.lin3.weight.data, 1.0)
                self.lin3.bias.data.fill_(0)
                
                self.lin6.weight.data = normalized_columns_initializer(
                    self.lin6.weight.data, 1.0)
                self.lin6.bias.data.fill_(0)
    
    def forward(self, states, actions):
        x1 = self.Q1(states, actions)
        
        x2 = self.inc2(states)
        x2 = self.down5(x2)
        x2 = self.down6(x2)
        x2 = self.down7(x2)
        x2 = self.down8(x2)
        
        x2 = x2.view(x2.size(0), -1)  # flatten
        x2 = self.lr(self.lin4(x2))
        x2 = self.lr(self.lin5(torch.cat([x2, actions], 1)))  # c
        x2 = self.lin6(x2)
        
        return x1, x2

    def Q1(self, states, actions):
        x1 = self.inc1(states)
        x1 = self.down1(x1)
        x1 = self.down2(x1)
        x1 = self.down3(x1)
        x1 = self.down4(x1)
        
        x1 = x1.view(x1.size(0), -1)  # flatten
        x1 = self.lr(self.lin1(x1))
        x1 = self.lr(self.lin2(torch.cat([x1, actions], 1)))  # c
        x1 = self.lin3(x1)
        
        return x1

