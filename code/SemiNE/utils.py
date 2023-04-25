# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:01:43 2020

@author: yqiu
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.decomposition import randomized_svd
from scipy.signal import find_peaks
from scipy.stats import pearsonr#Pearson correlation coefficient
import cv2
import scipy.optimize as opt

#import pytorch_ssim


def img_real2view(img):
    """
    Visualize image with gamma correction
    """
    gamma_correction=lambda x:np.power(x,1.0/2.2)
    img_shape=img.shape
    # gray image
    if np.size(img_shape)==2:
        #uint8
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img)/255.0#float32, 1.0
            temp_view=gamma_correction(temp_view)
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view*255)
            return temp_view2
        #float
        if np.max(img)<2:
            return gamma_correction(img)
    #color image
    if np.size(img_shape)==3:
        ''' #update on 2019.05.28
        #uint8, BGR
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img[...,::-1])/255.0#rgb,1.0
            temp_view[...,-1]=gamma_correction(temp_view[...,-1])
            temp_view[...,1]=gamma_correction(temp_view[...,1])
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view[...,::-1]*255)#bgr,255
            return temp_view2
        #float, RGB
        if np.max(img)<2:
            return gamma_correction(img)
        '''
        #uint8
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img)/255.0#1.0
            temp_view=gamma_correction(temp_view)
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view*255)#255
            return temp_view2
        #float
        if np.max(img)<2:
            return gamma_correction(img)


def read_movie_from_h5(filename):
    """
    real means after spectral calibration, without gamma correction for the screen
    """
    h5f = h5py.File(filename,'r')
    movie_bgr_h5=h5f['movie_bgr_real'][:]
    h5f.close()
    return movie_bgr_h5


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class Flatten3D(nn.Module):
    def forward(self, x):
        N, C, D, H, W = x.size() # read in N, C, D, H, W
        return x.view(N, -1)  # "flatten" the C *D* H * W values into a single vector per image

class Unflatten3D(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*D*H*W) and reshapes it
    to produce an output of shape (N, C, D,H, W).
    """
    def __init__(self, N=-1, C=128, D=7, H=7, W=7):
        super(Unflatten3D, self).__init__()
        self.N = N
        self.C = C
        self.D = D
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.D, self.H, self.W)

'''
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
'''
def init_weights_Kaiming(m):
    if type(m) == nn.Linear or type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d\
    or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        #torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)

def loss_vae(recon_x, x, mu, logvar):
    tempN=x.shape[0]
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return BCE + KLD
    return (MSE + KLD)/(tempN*10*2*56*56)#batch size tempN, to be comparable with other ae variants loss

loss_mse = nn.MSELoss()
#loss_ssim = pytorch_ssim.SSIM(window_size=3)

#loss with L2 and L1 regularizer
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempN=x.shape[0]
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    L1loss=beta*F.l1_loss(beta_y,torch.zeros_like(beta_y),reduction='sum')
    return (MSE+L2loss+L1loss)/(tempN*10*2*56*56)#batch size tempN, to be comparable with other ae variants loss

#loss with L2 and L1 regularizer, version2
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1v2(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempC, tempD, tempH, tempW =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    B, C, D, H, W = beta_y.size() # Batch*channel*depth*height*width
    temp1=beta_y.view(B,C,-1)
    temp2=torch.norm(temp1,p=2,dim=2)
    temp3=torch.sum(torch.abs(temp2))
    L1loss=beta*temp3
    #return (MSE+L2loss+L1loss)/(tempB* 2*10*12*12)#batch size tempN, to be comparable with other ae variants loss
    return (MSE+L2loss+L1loss)/(tempB* tempC* tempD* tempH* tempW)#to be comparable with other ae variants loss

#loss with L2 and L1 regularizer, for supervised encoded
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)
#loss with L2 and L1 regularizer, for supervised encoded, Poisson loss
#something like loss = Poissonloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def Ploss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (Ploss+L2loss+L1loss)/(tempB* tempN)
def Ploss_L2L1_SE2(recon_x, x, alpha, alpha2, beta, alpha_x, alpha_x2, beta_y): #different convkernels with different L2 penalty
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    #
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(2)
    L2loss=alpha*l2temp
    #
    l2temp2=0.0
    for temp in alpha_x2:
        l2temp2 = l2temp2 + temp.weight.norm(2)
    L2loss2=alpha2*l2temp2
    #
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (Ploss+L2loss+L2loss2+L1loss)/(tempB* tempN)
def Ploss_L2L1_SE_ST(recon_x, x, alpha1, alpha2, beta, alpha_x1, alpha_x2, beta_y): # for spatial and temporal separable model
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    l2temp=0.0
    for temp in alpha_x1:
        l2temp = l2temp+ temp.norm(2)
    l2temp2=0.0
    for temp in alpha_x2:
        l2temp2 = l2temp2+ temp.norm(2)
    L2loss=alpha1*l2temp+alpha2*l2temp2
    #
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    #return (Ploss+L2loss+L1loss)/(tempB* tempN)
    return Ploss+L2loss+L1loss
#loss for training, semi-supervised learning model, Poisson loss for neural data, MSE loss for natural images, autoencoder
def loss_SemiSL(recon_x, x, alpha1, alpha2, beta1, alpha_Wsc, alpha_Wtc, beta_Wf,\
                recon_z, z, alpha3,         beta2, alpha_Wsd,            beta_h, lossweight):
    if lossweight==0:
        lossweight_temp=1e-8
    elif lossweight==1:
        lossweight_temp=1-1e-8
    else:
        lossweight_temp=lossweight
    #Poisson loss for neural data, also share the conv l2 penalty with AE model
    tempB, tempN =x.size()
    Ploss = lossweight*F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    #
    l2temp=0.0
    for temp in alpha_Wsc: #spatial conv
        l2temp = l2temp+ temp.norm(2)
    L2loss=lossweight*alpha1*l2temp
    #
    l2temp2=0.0
    for temp in alpha_Wtc: #temporal conv
        l2temp2 = l2temp2+ temp.weight.norm(2)
    L2loss2=lossweight*alpha2*l2temp2/lossweight_temp
    #
    l1temp=0.0
    for temp in beta_Wf: #FC layer
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=lossweight*beta1*l1temp/lossweight_temp
    loss_neuraldata=(Ploss + L2loss + L2loss2 + L1loss)/(tempB* tempN)
    #MSE loss for natural images, autoencoder
    tempB_z, tempC_z, tempD_z, tempH_z, tempW_z =z.size()
    MSE = (1-lossweight)*F.mse_loss(recon_z, z,reduction='sum')
    #
    l2temp_z=0.0
    for temp in alpha_Wsc: #spatial conv
        l2temp_z = l2temp_z+ temp.norm(2)
    L2loss_z=(1-lossweight)*alpha3*l2temp_z
    #
    l2temp_z2=0.0
    for temp in alpha_Wsd: #spatial deconv
        l2temp_z2 = l2temp_z2+ temp.weight.norm(2)
    L2loss_z2=(1-lossweight)*alpha3*l2temp_z2/(1-lossweight_temp)
    #
    l1temp_z=0.0
    for temp in beta_h: #encoder output
        l1temp_z = l1temp_z + temp.norm(1)
    L1loss_z=(1-lossweight)*beta2*l1temp_z/(1-lossweight_temp)
    #
    loss_ae=(MSE + L2loss_z + L2loss_z2 + L1loss_z)/(tempB_z* tempC_z* tempH_z* tempW_z*tempD_z)
    return loss_neuraldata+loss_ae
def loss_SemiSL_valloss(recon_x, x, alpha1, alpha2, beta1, alpha_Wsc, alpha_Wtc, beta_Wf,\
                recon_z, z, alpha3,         beta2, alpha_Wsd,            beta_h, lossweight):
    """
    This semi-loss function is for training models while recording training and val loss for each epoch,
    usually we do not use it for training, instead we use the function loss_SemiSL()
    """
    if lossweight==0:
        lossweight_temp=1e-8
    elif lossweight==1:
        lossweight_temp=1-1e-8
    else:
        lossweight_temp=lossweight
    #Poisson loss for neural data, also share the conv l2 penalty with AE model
    tempB, tempN =x.size()
    Ploss = F.poisson_nll_loss(recon_x, x,log_input=False, reduction='sum')
    #
    l2temp=0.0
    for temp in alpha_Wsc: #spatial conv
        l2temp = l2temp+ temp.norm(2)
    L2loss=alpha1*l2temp
    #
    l2temp2=0.0
    for temp in alpha_Wtc: #temporal conv
        l2temp2 = l2temp2+ temp.weight.norm(2)
    L2loss2=alpha2*l2temp2/lossweight_temp
    #
    l1temp=0.0
    for temp in beta_Wf: #FC layer
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta1*l1temp/lossweight_temp
    loss_neuraldata=(Ploss + L2loss + L2loss2 + L1loss)/(tempB* tempN)
    #MSE loss for natural images, autoencoder
    tempB_z, tempC_z, tempD_z, tempH_z, tempW_z =z.size()
    MSE = F.mse_loss(recon_z, z,reduction='sum')
    #
    l2temp_z=0.0
    for temp in alpha_Wsc: #spatial conv
        l2temp_z = l2temp_z+ temp.norm(2)
    L2loss_z=alpha3*l2temp_z
    #
    l2temp_z2=0.0
    for temp in alpha_Wsd: #spatial deconv
        l2temp_z2 = l2temp_z2+ temp.weight.norm(2)
    L2loss_z2=alpha3*l2temp_z2/(1-lossweight_temp)
    #
    l1temp_z=0.0
    for temp in beta_h: #encoder output
        l1temp_z = l1temp_z + temp.norm(1)
    L1loss_z=beta2*l1temp_z/(1-lossweight_temp)
    #
    loss_ae=(MSE + L2loss_z + L2loss_z2 + L1loss_z)/(tempB_z* tempC_z* tempH_z* tempW_z*tempD_z)
    return loss_neuraldata,loss_ae

def loss_L1L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    for temp in alpha_x:
        l2temp = l2temp+ temp.weight.norm(1)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)
'''
#loss with L2 and L1 regularizer, for supervised encoded, L2 for conv kernel smoothness
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2lapL1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    laplacian=torch.tensor([[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]], requires_grad=False)#laplacian kernel
    for temp in alpha_x:
        #l2temp = l2temp+ temp.weight.norm(2)
        NN,CC=temp.weight.shape[0],temp.weight.shape[1]
        laplacians=laplacian.repeat(CC, CC, 1, 1).requires_grad_(False).to(device)
        temp2=F.conv2d(temp.weight,laplacians)
        l2temp = l2temp+ temp2.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)
'''
#visualize DNN
#https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/NetworkVisualization-PyTorch.ipynb
#https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
#https://jacobgil.github.io/deeplearning/filter-visualizations
#https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
#simple version: compute the gradient of the output channel wrt a blank image
#complex version: performa gradient ascend on the target channel, start with noise image
def vis_model_fl(model,device,xxshape):#visualize for final layer
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    model=model.eval()
    (tempB,tempC,tempH,tempW)=xxshape#tempB should be equal to 1
    #xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
    xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
    if xx.grad is not None:
        xx.grad.data.zero_()
    out=model(xx)
    outlen=out.shape[1]
    yy=torch.zeros(outlen,tempC,tempH,tempW)
    for ii in range(outlen):
        if xx.grad is not None:
            xx.grad.data.zero_()
        out=model(xx)
        temp=out[0,ii]
        temp.backward()
        yy[ii]=xx.grad.data
        #if xx.grad is not None:
        #    xx.grad.data.zero_()
    return yy
def vis_model_fl_3d(model,device,xxshape):#visualize for final layer
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    model=model.eval()
    (tempB,tempC,tempD,tempH,tempW)=xxshape#tempB should be equal to 1
    #xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
    xx=torch.zeros((tempB,tempC,tempD,tempH,tempW),requires_grad=True)
    if xx.grad is not None:
        xx.grad.data.zero_()
    out=model(xx)
    outlen=out.shape[1]
    yy=torch.zeros(outlen,tempC,tempD,tempH,tempW)
    for ii in range(outlen):
        if xx.grad is not None:
            xx.grad.data.zero_()
        out=model(xx)
        temp=out[0,ii]
        temp.backward()
        yy[ii]=xx.grad.data
        #if xx.grad is not None:
        #    xx.grad.data.zero_()
    return yy

#https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4
'''
class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            #return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din
'''
'''
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        #self.noise = torch.tensor(0).to(device)
        #self.noise = torch.randn_like(input, dtype=None, device=device, requires_grad=False)

    def forward(self, x):
        if self.training and self.sigma != 0:
            #scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            #sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            sampled_noise = torch.randn(x.size(),requires_grad=False).to(device)* self.sigma
            x = x + x*sampled_noise
        return x
'''
class GaussianNoise(nn.Module):
    """
    Add gaussian noise to the output of one layer
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        #self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            #scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * self.sigma
            #sampled_noise = torch.randn(x.size(),requires_grad=False).to(device)* self.sigma
            x = x + sampled_noise #x + sampled_noise
        return x


def mySVD(w, dims):
    """
    SVD for 3d or 2d kernels
    only apply to Gaussian or DOG kernel, the center RF is near the frame center
    """
    if len(dims) == 3:
        dims_tRF = dims[0]
        dims_sRF = dims[1:]
        w_old=np.copy(w)
        w=np.reshape(w,(dims_tRF, np.prod(dims_sRF)))
        # Data matrix X, centered X
        w=w-np.mean(w,axis=0)
        U, S, Vt = randomized_svd(w, 3)
        sRF = Vt[0].reshape(*dims_sRF)
        tRF = U[:, 0]
        #change the sign of sRF and tRF to map with 3d RFs, which is unpredicable in SVD
        '''
        tempcc,_=pearsonr(sRF.flatten(), w[-2])
        if tempcc<0:
            sRF = -1 * sRF
            tRF = -1 * tRF
        '''
        '''
        #keep sRF and w[-2] the same scale
        sRF = np.max(w[-2])/np.max(sRF) *sRF
        tRF = np.max(sRF)/np.max(w[-2]) *tRF
        '''
        '''
        tempscale=np.max(np.abs(w[-2]))/np.max(np.abs(sRF))
        sRF = tempscale *sRF
        neww=np.einsum('i,jk->ijk',tRF,sRF)
        tempscale=np.max(w)/np.max(neww)
        tRF = tempscale *tRF
        '''
        '''
        tempscale=np.max(np.abs(w[-2]))/np.max(np.abs(sRF))
        sRF = tempscale *sRF
        tRF = tempscale *tRF
        '''
        peaks, _ = find_peaks(np.abs(tRF)) #peak index
        peak_close2t0=peaks[-1] #peak index close to time point 0
        w_old_peak_blur=cv2.GaussianBlur(w_old[peak_close2t0],(5,5),0)
        sRF_blur=cv2.GaussianBlur(sRF,(5,5),0)
        #sRF, same polarity and magnitude as w_ol
        tempcc,_=pearsonr(sRF_blur.flatten(), w_old_peak_blur.flatten())
        if tempcc<0:
            sRF = -1 * sRF
        tempscale=np.max(np.abs(w_old_peak_blur))/np.max(np.abs(sRF_blur))
        sRF = tempscale *sRF
        #tRF
        #tRF = tempscale *tRF
        #if -np.min(w_old_peak_blur)>np.max(w_old_peak_blur) and tRF[peak_close2t0]>0: #w_old: off, tRF: on
        #    tRF = -1 *tRF
        #elif -np.min(w_old_peak_blur)<np.max(w_old_peak_blur) and tRF[peak_close2t0]<0: #w_old: on, tRF: off
        #    tRF = -1 *tRF
        if tempcc<0:
            tRF = -1 * tRF
    else:
        sRF = w
        tRF = None
    return [sRF, tRF]

class Fit_2dGaussian():
    """
    fitting to 2d Gaussian and measure the fitting goodness
    input_data: input y-values, 1d array
    data_shape: raw data shape, tuple, e.g. for a 9x9 RF of a neuron: (9,9)
    
    if fail to fit, r2=0; if the parametre sigma_x or sigma_y > the size of the RF(9), then r2=0
    
    #https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    #https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    #https://stackoverflow.com/questions/29003241/how-to-quantitatively-measure-goodness-of-fit-in-scipy
    
    test examples:
        #define model function and pass independant variables x and y as a list
        def twoD_Gaussian(xydata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
            (x, y) = xydata_tuple 
            xo = float(xo)
            yo = float(yo)    
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
            return g.ravel()
        # Create x and y indices
        x = np.arange(9)
        x,y = np.meshgrid(x, x)
        #create data
        data = twoD_Gaussian((x, y), 2, 4, 4, 3, 5, 0, 1)
        data_noisy = data + 0.02*np.random.normal(size=data.shape)
        #
        fit_2dgaussian=Fit_2dGaussian(data_noisy,(9,9))
        fit_2dgaussian.fit()
        fit_2dgaussian.plot()
        print (fit_2dgaussian.cal_r2())
    
    """
    def __init__(self,input_data,data_shape):
        super().__init__()
        self.input_data=input_data
        self.data_shape=data_shape
        #
        self.sigma_thresh=max(self.data_shape) # threshold of sigma after fitting
        self.data_fitted=None
        self.fit_sucess=None # indicate the success of failure of fit
        self.popt=None
        self.large_sigma=None # indicate the sigma_x or sigma_y is too large
    
    def _twoD_Gaussian(self,xydata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        """define model function and pass independant variables x and y as a list.
        The output of twoD_Gaussian needs to be 1D
        """
        (x, y) = xydata_tuple 
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
        return g.ravel()

    def _fitting_r2(self,y,y_fit):
        """
        Use coefficient of determination (aka the R2 value) to measure goodness-of-fit
        y refers to your input y-values, y_fit refers to your fitted y-values. Both are 1d arrays
    
        """
        # residual sum of squares
        ss_res = np.sum((y - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def fit(self):
        """
        fitting
        in case of failing to fit:
        #https://stackoverflow.com/questions/9172574/scipy-curve-fit-runtime-error-stopping-iteration
        """
        x=np.arange(self.data_shape[0])
        y=np.arange(self.data_shape[1])
        x,y = np.meshgrid(x, y)
        #
        initial_guess = (1,int(self.data_shape[0]/2),int(self.data_shape[1]/2),\
                         int(self.data_shape[0]/2),int(self.data_shape[1]/2),0,1)
        #popt, pcov = opt.curve_fit(self._twoD_Gaussian, (x, y), self.input_data, p0=initial_guess)
        #self.data_fitted = self._twoD_Gaussian((x, y), *popt)
        #self.fit_sucess=False
        try:
            popt,pcov = opt.curve_fit(self._twoD_Gaussian, (x, y), self.input_data, p0=initial_guess)
        except RuntimeError:
            self.fit_sucess=False
        else:
            self.popt=popt
            self.data_fitted = self._twoD_Gaussian((x, y), *popt)
            self.fit_sucess=True
            if self.popt[-3]>self.sigma_thresh or self.popt[-4]>self.sigma_thresh:
                self.large_sigma=True
        return None
    
    def plot(self):
        """
        Plot the fitting result
        """
        if self.fit_sucess==True:
            x=np.arange(self.data_shape[0])
            y=np.arange(self.data_shape[1])
            x,y = np.meshgrid(x, y)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(self.input_data,self.data_shape))
            ax.contour(x, y, np.reshape(self.data_fitted,self.data_shape), 3, colors='w')
        elif self.fit_sucess==False:
            print ('Fail to fit, no plot!')
    
    def cal_r2(self):
        """
        calculate r2 - goodness of fitting
        """
        if self.fit_sucess==True:
            if self.large_sigma==True:
                return 0
            else:
                return self._fitting_r2(self.input_data,self.data_fitted)
        elif self.fit_sucess==False:
            return 0

    def get_sigma(self):
        """
        get mean of two sigma_x and sigma_y
        """
        if self.fit_sucess==True:
            if self.large_sigma==True:
                return 0
            else:
                return (np.abs(self.popt[-3]) + np.abs(self.popt[-4]))/2
        elif self.fit_sucess==False:
            return 0
