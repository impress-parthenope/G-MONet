#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: sergio

This is the testing code of
S. Vitale, G. Ferraioli, A. C. Frery, V. Pascazio, D. -X. Yue and F. Xu, 
"SAR Despeckling Using Multiobjective Neural Network Trained With Generic Statistical Samples," 
(2023) in IEEE Transactions on Geoscience and Remote Sensing, 

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import os
import sys

"Path Selection"
model_path = './trained_weights/' # trained model path
testset_path = './data/test_img.mat'    # testing path
out_path = './output/'  # output saving path

"set gpu if available"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
"Model"
from model import MONet
net = MONet()
net.load_state_dict(torch.load(model_path+'/DNN_model',map_location=device))
blk = net.net_scope() #global receptive field of network

plt.close('all')
with torch.no_grad():

    "Loading data:"
    # if mat file
    I = sio.loadmat(testset_path,squeeze_me=True) 
    I = np.asarray(I['noisy'],dtype='float32')
    
    #if numpy file
    #I = np.load(testset_path) #if numpy file
    
    #input: Amplitude format
    I_in = np.abs(I)
       
    "Input Preparation:"
    I_in = np.where(np.equal(I_in,0),1e-16,I_in) # managing exceptions
    I_in = np.pad(I_in, ((blk,blk),(blk,blk)),mode='edge')#padding
    while len(I_in.shape)<4: #shape check
        I_in = np.expand_dims(I_in,axis=0)
    I_in = torch.from_numpy(I_in)
    	
    "Testing"
    net.eval()
    net.to(device)
    # output: Amplitude format
    I_out = net(I_in.to(device))
        
    I_out = I_out.cpu().detach().numpy()#detach from gpu
    I_out = I_out[0,0,:,:]
        
    "Saving"
    sio.savemat(out_path+'output.mat',{'I_out':I_out})
    
    "Visualization"
    plt.figure()        
    plt.subplot(131),plt.imshow(I, cmap = 'gray',vmin=0,vmax=255),plt.title('noisy')
    plt.subplot(132),plt.imshow(I_out, cmap = 'gray',vmin=0,vmax=255),plt.title('image -filtered')
    plt.subplot(133),plt.imshow(I/I_out, cmap = 'gray',vmin=0,vmax=4),plt.title('speckle')
        
