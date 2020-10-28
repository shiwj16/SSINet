import math
import functools
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_parts.unet_parts import *
from net_parts.fcdensenet_parts import *
from net_parts.deeplabv3_parts import ASPP
from net_parts.refinenetLW_parts import conv1x1, conv3x3, CRPBlock, Bottleneck


class MaskUnet(nn.Module):
    def __init__(self, pretrained_Actor, algo, env_name="duckietown"):
        super(MaskUnet, self).__init__()
        
        self.algo = algo
        self.env_name = env_name
        self.actor = pretrained_Actor
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False
        
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 16)
        self.up5 = up(32, 16)
        out_dim = 4 if self.env_name == "atari" else 1
        self.outc = outconv(16, out_dim)
    
    def forward(self, state):
        # encode
        x1 = self.actor.inc(state)
        x2 = self.actor.down1(x1)
        x3 = self.actor.down2(x2)
        if self.actor.net_layers == 3:
            x = self.actor.down2_1(x3)
        else:
            x4 = self.actor.down3(x3)
            if self.actor.net_layers == 4:
                x5 = self.actor.down3_1(x4)
                # decode
                x = self.up2(x5, x4)
            elif self.actor.net_layers == 5:
                x5 = self.actor.down4(x4)
                x6 = self.actor.down4_1(x5)
                # decode
                x6_1 = self.up1(x6, x5)
                x = self.up2(x6_1, x4)
        
        # decode
        x7 = self.up3(x, x3)
        x8 = self.up4(x7, x2)
        x9 = self.up5(x8, x1)
        x10 = self.outc(x9)
        
        # mask
        mask = self.actor.sigm(x10)
        masked_x = state.mul(mask)

        if self.algo == "td3":
            act = self.actor(masked_x)       
            return act, mask
        elif self.algo == "sac":
            mean, _ = self.actor(masked_x)       
            return mean, mask
        elif self.algo == "ppo":
            if self.env_name == "duckietown":
                _, pi = self.actor(masked_x)
                mean, _ = pi
                return mean, mask
            elif self.env_name == "atari":
                _, act_logit = self.actor(masked_x)
                act_softmax = F.softmax(act_logit, dim=1)
                return act_softmax, mask
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class MaskResnetLW(nn.Module):
    def __init__(self, pretrained_Actor, algo):
        super(MaskResnetLW, self).__init__()
        
        self.algo = algo
        self.actor = pretrained_Actor
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False
        
        self.p_ims1d2_outl1_dimred = conv1x1(1024, 256, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(256, 128, bias=False)
        
        self.p_ims1d2_outl2_dimred = conv1x1(512, 128, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(128, 128, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(128, 128, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(128, 128, bias=False)
        
        self.p_ims1d2_outl3_dimred = conv1x1(256, 128, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(128, 128, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(128, 128, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(128, 128, bias=False)
        
        self.p_ims1d2_outl4_dimred = conv1x1(128, 128, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(128, 128, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(128, 128, 4)
        
        self.clf_conv = nn.Conv2d(128, 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)
    
    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def forward(self, state):
        # encode
        x = self.actor.conv1(state)
        x = self.actor.bn1(x)
        x = self.actor.relu(x)
        x = self.actor.maxpool(x)
        
        l1 = self.actor.layer1(x)
        l2 = self.actor.layer2(l1)
        l3 = self.actor.layer3(l2)
        l4 = self.actor.layer4(l3)
        
        # decode
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = F.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)
        
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)
        
        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        
        out = self.clf_conv(x1)
        
        # mask
        out = self.actor.sigm(out)
        mask = F.interpolate(out, size=state.size()[2:], mode='bilinear', align_corners=False)
        masked_x = state.mul(mask)

        if self.algo == "td3":
            act = self.actor(masked_x)       
            return act, mask
        elif self.algo == "ppo":
            _, pi = self.actor(masked_x)
            mean, _ = pi
            return mean, mask
        else:
            raise NotImplementedError


class MaskMobilenetLW(nn.Module):
    def __init__(self, pretrained_Actor, algo):
        super(MaskMobilenetLW, self).__init__()
        
        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.segm = conv3x3(256, 1, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self._initialize_weights()

        self.algo = algo
        self.actor = pretrained_Actor
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def forward(self, state):
        # encode
        x = self.actor.layer1(state)
        x = self.actor.layer2(x) # x / 2
        l3 = self.actor.layer3(x) # 24, x / 4
        l4 = self.actor.layer4(l3) # 32, x / 8
        l5 = self.actor.layer5(l4) # 64, x / 16
        l6 = self.actor.layer6(l5) # 96, x / 16
        l7 = self.actor.layer7(l6) # 160, x / 32
        l8 = self.actor.layer8(l7) # 320, x / 32
        
        # decode
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=True)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=True)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        out = self.segm(l3)
        
        # mask
        out = self.actor.sigm(out)
        mask = F.interpolate(out, size=state.size()[2:], mode='bilinear', align_corners=False)
        masked_x = state.mul(mask)

        if self.algo == "td3":
            act = self.actor(masked_x)       
            return act, mask
        elif self.algo == "ppo":
            _, pi = self.actor(masked_x)
            mean, _ = pi
            return mean, mask
        else:
            raise NotImplementedError


class MaskFCDensenet(nn.Module):
    def __init__(self, pretrained_Actor, algo):
        super(MaskFCDensenet, self).__init__()
        
        self.algo = algo
        self.actor = pretrained_Actor
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False
        
        growth_rate = self.actor.growth_rate
        cur_channels_count = self.actor.cur_channels_count
        skip_connection_channel_counts = self.actor.skip_connection_channel_counts
        bottleneck_layers = self.actor.bottleneck_layers
        up_blocks = self.actor.up_blocks[-self.actor.n_pool:]
        
        # Bottleneck
        self.add_module('bottleneck',FCBottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers

        # Upsampling path
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels
        
        # Final DenseBlock
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        # final conv
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, state):
        out = self.actor.firstconv(state)

        skip_connections = []
        for i in range(self.actor.n_pool):
            out = self.actor.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.actor.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(self.actor.n_pool):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        
        # mask
        mask = self.actor.sigm(out)
        masked_x = state.mul(mask)

        if self.algo == "td3":
            act = self.actor(masked_x)       
            return act, mask
        elif self.algo == "ppo":
            _, pi = self.actor(masked_x)
            mean, _ = pi
            return mean, mask
        else:
            raise NotImplementedError


class MaskDeeplabv3(nn.Module):
    def __init__(self, pretrained_Actor, algo):
        super(MaskDeeplabv3, self).__init__()
        
        self.algo = algo
        self.actor = pretrained_Actor
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False
        
        # NOTE! if you use ResNet50-152, 
        # set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.aspp = ASPP(num_classes=1) 
        
    def forward(self, state):
        h = state.size()[2]
        w = state.size()[3]
        # encode
        feature_map = self.actor.resnet(state)
        # (shape: (batch_size, num_classes, h/16, w/16))
        output = self.aspp(feature_map)
        # (shape: (batch_size, num_classes, h, w))
        out = F.upsample(output, size=(h, w), mode="bilinear")
        
        # mask
        mask = self.actor.sigm(out)
        masked_x = state.mul(mask)

        if self.algo == "td3":
            act = self.actor(masked_x)       
            return act, mask
        elif self.algo == "ppo":
            _, pi = self.actor(masked_x)
            mean, _ = pi
            return mean, mask
        else:
            raise NotImplementedError
