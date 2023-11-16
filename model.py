# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:31:53 2020

@author: sergv
"""
import numpy as np
import torch
from torch.autograd import Variable
import sys
eps = sys.float_info.epsilon


#%% MONet
import torch.nn as nn
import torch.nn.functional as F

class model_block(torch.nn.Module):
    def __init__(self, in_feat, out_feat,kern):
        super(model_block, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kern)
        self.conv1_bn = nn.BatchNorm2d(out_feat)
        self.conv2 = nn.Conv2d(in_feat, out_feat, kern)
        self.conv2_bn = nn.BatchNorm2d(out_feat)
        self.conv3 = nn.Conv2d(in_feat, out_feat, kern)
        self.conv3_bn = nn.BatchNorm2d(out_feat)
        
    def forward(self, x_):
        x = F.relu(self.conv1_bn(self.conv1(x_)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x+x_[:,:,2:-2,2:-2])))
        return x
    
class MONet(nn.Module):
    def __init__(self,blocks=5):
        super(MONet, self).__init__()
        self.inn_layers1=[]
        self.inn_layers2=[]
        self.blocks=blocks
        
        #define first subnet
        self.conv_in1 = nn.Conv2d(1, 64, 3)
        self.inn_layers1 = nn.ModuleList([model_block(64,64,3) for i in range(self.blocks)])
        self.conv_out1 = nn.Conv2d(64, 1, 3)
        # self.net_scope
        
    def net_scope(self):
        blk=0
        blk += self.conv_in1.weight.shape[-1]-self.conv_in1.weight.shape[-1]//2
        for i in range(self.blocks):
            blk += self.inn_layers1[i].conv1.weight.shape[-1]-self.inn_layers1[i].conv1.weight.shape[-1]//2
            blk += self.inn_layers1[i].conv2.weight.shape[-1]-self.inn_layers1[i].conv2.weight.shape[-1]//2
            blk += self.inn_layers1[i].conv3.weight.shape[-1]-self.inn_layers1[i].conv3.weight.shape[-1]//2
            
        blk += self.conv_out1.weight.shape[-1]-self.conv_out1.weight.shape[-1]//2
        
        return blk//2
    
    def forward(self, x_in):
        
        x = F.relu(self.conv_in1(x_in))
        for i in range(self.blocks):
            x = self.inn_layers1[i](x)
        x_out1 = self.conv_out1(x)
        
        return x_out1

