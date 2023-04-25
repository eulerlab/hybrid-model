# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:01:43 2020

@author: yqiu
"""

import torch
#import torch.nn as nn
#from torch.nn import functional as F

import numpy as np
import datetime
#from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr#Pearson correlation coefficient

#from SemiNE.utils import img_real2view


#function for training,(alpha,beta) for L2L1v2 regularizer
def model_train(model,data,optimizer,device,EPOCH,loss_func,train_mean,valdata,\
                alpha1=None, alpha2=None, beta1=None, alpha3=None,\
                beta2=None, lossweight=1, earlystop=False,verbose=True):
    """
    Function for training neural network models
    Parameters:
        train_mean: the mean of training data for each color channels separately
        valdata: validation data
    """
    print(datetime.datetime.now())
    #model=model.to(device)
    #model=model.train()
    loss=0.0 # loss of each batch, for training and backpropogation
    #semilosses=[] # monitoring supervised encoder loss and ae loss during semi-supervised learning
    vallosses=np.zeros((EPOCH)) # save validation losses of all epochs until early stopping
    #scheduler = MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)
    for epoch in range(EPOCH):
        model=model.to(device)
        model=model.train()
        for step, (x,y,z) in enumerate(data):
            #neural data
            x=torch.from_numpy(x).float() #input
            y=torch.from_numpy(y).float() #responses
            b_x = x.to(device)
            b_y = y.to(device)
            #nature image data
            z=z/255.0
            #z[:,0,:,:]=z[:,0,:,:]-train_mean[0]
            #z[:,1,:,:]=z[:,1,:,:]-train_mean[1]
            #z = z[:,:,np.newaxis,:,:]
            #b_z = torch.from_numpy(z).float().to(device)
            #b_zz= torch.from_numpy(z).float().to(device)
            z[:,0,:,:,:]=z[:,0,:,:,:]-train_mean[0]
            z[:,1,:,:,:]=z[:,1,:,:,:]-train_mean[1]
            if 'Conv3d_1' in model.__class__.__name__:
                #when temporal kernel size = 8, temporal dimension of input = 8
                #b_z = torch.from_numpy(z).float().to(device)
                #b_zz = torch.from_numpy(z[:,:,-2,:,:]).float().to(device)
                #when temporal kernel size = 8, temporal dimension of input = 9
                b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
                b_zz = torch.from_numpy(z[:,:,-3,:,:]).float().to(device)
            elif 'Conv3d_2' in model.__class__.__name__:
                #when temporal kernel size = 8, temporal dimension of input = 8
                #b_z = torch.from_numpy(z).float().to(device)
                #b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
                #b_zz = torch.from_numpy(z[:,:,-1,:,:]).float().to(device)
                #when temporal kernel size = 8, temporal dimension of input = 9
                b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
                b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
                b_zz = torch.from_numpy(z[:,:,-2,:,:]).float().to(device)
            elif 'Conv3d_4' in model.__class__.__name__:
                #when temporal kernel size = 8, temporal dimension of input = 9
                b_z = torch.from_numpy(z[:,:,:z.shape[2]-1,:,:]).float().to(device)
                b_zz = torch.from_numpy(z[:,:,-1,:,:]).float().to(device)
            #
            if 'Semi' in model.__class__.__name__:
                encoded, ae_encoded, ae_decoded = model(b_x,b_z)
                # spatial temporal separable kernels
                #loss=loss_func(encoded, b_y, alpha1, alpha2, beta1, [model.conv1_ss], [model.conv1_st], [model.fc1],\
                #ae_decoded, b_zz, alpha3, beta2, [model.ae_dconv1], ae_encoded, lossweight)
                # conv3d
                loss=loss_func(encoded, b_y, alpha1, alpha2, beta1, [model.conv1], None, [model.fc1],\
                ae_decoded, b_zz, alpha3, beta2, [model.ae_dconv1], ae_encoded, lossweight)
            else:
                print ('Not Semi model!')
                break
            #
            #last epoch to get the training loss, keep the same sample size as validation
            #if epoch==EPOCH-1:
            #    train_loss=train_loss+loss.cpu().data.numpy()
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            #
            if step % 100 == 0 and verbose==True:
                print('Model: ',model.__class__.__name__,'|Epoch: ', epoch,\
                      '| train loss: %.4f' % loss.cpu().data.numpy())
        #one epoch done
        #scheduler.step()
        if epoch>20 and earlystop==True: # early stopping check after each epoch, use CC as a metric
            temploss=model_val(model,valdata,1,device,train_mean)
            vallosses[epoch]=temploss
            if epoch-np.argmax(vallosses)>4:
                break
    print ('Epoch: {:} val loss: {:.4f}, finish training!'.format(epoch,vallosses[epoch]))
    print(datetime.datetime.now())
    #return trainlosses,vallosses #semilosses #test
    #train_loss=train_loss/len(data)
    #print ('Model: ',model.__class__.__name__,'|train loss: %.4f' % train_loss)
    #return train_loss,semilosses
#_=model_train(ae3D_4,'normalize',sky',train_loader_shuffle,optimizer,device,EPOCH,loss_mse)


#use the trained model to test the validation loss
#show one example results
#val_eg: the example used to show results
#val_num: the number of validation dataset, when using gpu, may have memory problem, then set it small
def model_val(model,data,val_eg,device,train_mean,loss_func=None):
    model=model.to(device)
    model=model.eval()
    #
    (x,y,z)=data
    x=torch.from_numpy(x).float()
    b_x = x.to(device)
    #
    z=z/255.0
    #z[:,0,:,:]=z[:,0,:,:]-train_mean[0]
    #z[:,1,:,:]=z[:,1,:,:]-train_mean[1]
    #z = z[:,:,np.newaxis,:,:]
    #b_z = torch.from_numpy(z).float().to(device)
    z[:,0,:,:,:]=z[:,0,:,:,:]-train_mean[0]
    z[:,1,:,:,:]=z[:,1,:,:,:]-train_mean[1]
    if 'Conv3d_1' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
    elif 'Conv3d_2' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        #b_z[:,:,-1,:,:]=0 # 7th frame is set to 0 after preprocessing
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
        b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
    elif 'Conv3d_4' in model.__class__.__name__:
        b_z = torch.from_numpy(z[:,:,:z.shape[2]-1,:,:]).float().to(device)
        
    #
    with torch.no_grad():
        if 'vae' in model.__class__.__name__:
            encoded, mu, logvar, decoded = model(b_x)
            #val_loss = loss_func(decoded, b_y, mu, logvar)
        elif 'Semi' in model.__class__.__name__:
            encoded,_,_ = model(b_x,b_z)
        else:
            encoded = model(b_x)
    #
    #CC as metric
    encoded_np=encoded.cpu().data.numpy()
    valcc,valpV=pearsonr(encoded_np.T.flatten(), y.T.flatten())
    if valpV>0.05:
        valcc=0
    #Poisson loss metric
    #y=torch.from_numpy(y).float()
    #b_y = y.to(device)
    #valcc = F.poisson_nll_loss(encoded, b_y,log_input=False, reduction='sum')
    #
    #print ('Model: ',model.__class__.__name__,'|validation cc: %.4f' % valcc)
    #show one example
    #fig,ax=plt.subplots(nrows=1, ncols=1,figsize=(10,2))
    #ax.plot(data[1][:,val_eg],color='r',label='Target')
    #ax.plot(encoded.cpu().data.numpy()[:,val_eg],color='g',label='Predict')
    #ax.legend(loc='best',fontsize=12)
    return valcc
#_=model_val(ae3D_4,'normalize','sky',val_loader_shuffle_sky,200,1000,device_cpu,loss_mse)


#using pearson correlation as metric
def model_test(model,data,device,train_mean,use_pad0_sti=True,verbose=True):
    model=model.to(device)
    model=model.eval()
    (x,y,z)=data
    x=torch.from_numpy(x).float()
    b_x = x.to(device)
    z=z/255.0
    #z[:,0,:,:]=z[:,0,:,:]-train_mean[0]
    #z[:,1,:,:]=z[:,1,:,:]-train_mean[1]
    #z = z[:,:,np.newaxis,:,:]
    #b_z = torch.from_numpy(z).float().to(device)
    z[:,0,:,:,:]=z[:,0,:,:,:]-train_mean[0]
    z[:,1,:,:,:]=z[:,1,:,:,:]-train_mean[1]
    if 'Conv3d_1' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
    elif 'Conv3d_2' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        #b_z[:,:,-1,:,:]=0 # 7th frame is set to 0 after preprocessing
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
        b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
    elif 'Conv3d_4' in model.__class__.__name__:
        b_z = torch.from_numpy(z[:,:,:z.shape[2]-1,:,:]).float().to(device)
        
    #
    with torch.no_grad():
        if 'Semi' in model.__class__.__name__:
            encoded,_,_ = model(b_x,b_z)
        else:
            encoded = model(b_x)
    encoded_np=encoded.cpu().data.numpy()
    if use_pad0_sti==False: # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
        encoded_np=encoded_np[x.shape[2]-1:,:]
        y=y[x.shape[2]-1:,:]
    testcc,testpvalue=pearsonr(encoded_np.T.flatten(), y.T.flatten())
    if verbose == True:
        #show the best example
        testccs=np.zeros(y.shape[1])
        encoded_np=encoded_np+1e-5 #in case all zeros
        for ii in range(len(testccs)):
            testccs[ii],_=pearsonr(encoded_np[:,ii], y[:,ii])
        testccs[np.isnan(testccs)] = 0 #nan to 0
        test_best=np.argmax(testccs)
        fig,ax=plt.subplots(nrows=1, ncols=1,figsize=(10,2))
        ax.plot(y[:,test_best],'o',color='r',linestyle='-',alpha=0.5,label='Target')
        ax.plot(encoded_np[:,test_best],'o',color='g',linestyle='-',alpha=0.5,label='Predict')
        ax.legend(loc='best',fontsize=12)
    print ('Overall pearson correlation coefficient: ',testcc, ' and p-value: ',testpvalue)
    return testcc,testpvalue
#model_test(ae3D_4,'normalize',sky',test_loader_shuffle,2000,device,loss_mse)


def model_train_valloss(model,data,optimizer,device,EPOCH,loss_func,train_mean,valdata,\
                        alpha1=None, alpha2=None, beta1=None, alpha3=None,\
                        beta2=None, lossweight=1, earlystop=False,verbose=True):
    """
    This function is for training models while recording training and val loss for each epoch,
    usually we do not use it for training, instead we use the function model_train()
    """
    print(datetime.datetime.now())
    #model=model.to(device)
    #model=model.train()
    loss=0.0 # loss of each batch, for training and backpropogation
    #semilosses=[] # monitoring supervised encoder loss and ae loss during semi-supervised learning
    trainlosses=np.zeros((3,EPOCH)) # neural data loss, ae loss, weighted loss of neural data and ae
    vallosses=np.zeros((EPOCH)) # save validation losses of all epochs until early stopping
    #scheduler = MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)
    for epoch in range(EPOCH):
        model=model.to(device)
        model=model.train()
        for step, (x,y,z) in enumerate(data):
            #neural data
            x=torch.from_numpy(x).float() #input
            y=torch.from_numpy(y).float() #responses
            b_x = x.to(device)
            b_y = y.to(device)
            #nature image data
            z=z/255.0
            #z[:,0,:,:]=z[:,0,:,:]-train_mean[0]
            #z[:,1,:,:]=z[:,1,:,:]-train_mean[1]
            #z = z[:,:,np.newaxis,:,:]
            #b_z = torch.from_numpy(z).float().to(device)
            #b_zz= torch.from_numpy(z).float().to(device)
            z[:,0,:,:,:]=z[:,0,:,:,:]-train_mean[0]
            z[:,1,:,:,:]=z[:,1,:,:,:]-train_mean[1]
            if 'Conv3d_1' in model.__class__.__name__:
                #b_z = torch.from_numpy(z).float().to(device)
                #b_zz = torch.from_numpy(z[:,:,-2,:,:]).float().to(device)
                b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
                b_zz = torch.from_numpy(z[:,:,-3,:,:]).float().to(device)
            elif 'Conv3d_2' in model.__class__.__name__:
                #b_z = torch.from_numpy(z).float().to(device)
                #b_z[:,:,-1,:,:]=0 # 7th frame is set to 0 after preprocessing
                #b_zz = torch.from_numpy(z[:,:,-1,:,:]).float().to(device)
                b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
                b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
                b_zz = torch.from_numpy(z[:,:,-2,:,:]).float().to(device)
            elif 'Conv3d_4' in model.__class__.__name__:
                b_z = torch.from_numpy(z[:,:,:z.shape[2]-1,:,:]).float().to(device)
                b_zz = torch.from_numpy(z[:,:,-1,:,:]).float().to(device)

            #
            if 'Semi' in model.__class__.__name__:
                encoded, ae_encoded, ae_decoded = model(b_x,b_z)
                # spatial temporal separable kernels
                #loss_neuraldata,loss_ae=loss_func(encoded, b_y, alpha1, alpha2, beta1, [model.conv1_ss], [model.conv1_st],\
                #                     [model.fc1],ae_decoded, b_zz, alpha3, beta2, [model.ae_dconv1], ae_encoded, lossweight)
                # conv3d
                loss_neuraldata,loss_ae=loss_func(encoded, b_y, alpha1, alpha2, beta1, [model.conv1], None, [model.fc1],\
                                                  ae_decoded, b_zz, alpha3, beta2, [model.ae_dconv1], ae_encoded, lossweight)

                loss=lossweight*loss_neuraldata+(1-lossweight)*loss_ae
            else:
                print ('Not Semi model!')
                break
            #
            #last epoch to get the training loss, keep the same sample size as validation
            trainlosses[0,epoch]=trainlosses[0,epoch]+loss_neuraldata.detach().clone().cpu().data.numpy()
            trainlosses[1,epoch]=trainlosses[1,epoch]+loss_ae.detach().clone().cpu().data.numpy()
            trainlosses[2,epoch]=trainlosses[2,epoch]+loss.detach().clone().cpu().data.numpy()
            #if epoch==EPOCH-1:
            #    train_loss=train_loss+loss.cpu().data.numpy()
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            #
            if step % 100 == 0 and verbose==True:
                print('Model: ',model.__class__.__name__,'|Epoch: ', epoch,\
                      '| train loss: %.4f' % loss.cpu().data.numpy())
        #one epoch done
        #scheduler.step()
        if epoch>20 and earlystop==True: # early stopping check after each epoch, use CC as a metric
            temploss=model_val(model,valdata,1,device,train_mean)
            vallosses[epoch]=temploss
            if epoch-np.argmax(vallosses)>4:
                break
        #test
        trainlosses[:,epoch] =trainlosses[:,epoch]/len(data)
        temploss=model_val(model,valdata,1,device,train_mean)
        vallosses[epoch]=temploss
    print ('Epoch: {:} val loss: {:.4f}, finish training!'.format(epoch,vallosses[epoch]))
    print(datetime.datetime.now())
    return trainlosses,vallosses #semilosses #test
    #train_loss=train_loss/len(data)
    #print ('Model: ',model.__class__.__name__,'|train loss: %.4f' % train_loss)
    #return train_loss,semilosses
#_=model_train(ae3D_4,'normalize',sky',train_loader_shuffle,optimizer,device,EPOCH,loss_mse)


#--------------------------------------------------------#
'''
def model_test_save(model,data,device,use_pad0_sti=True):
    model=model.to(device)
    model=model.eval()
    (x,y,z)=data
    x=torch.from_numpy(x).float()
    b_x = x.to(device)
    z=z/255.0
    z[:,0,:,:,:]=z[:,0,:,:,:]-sky_bg_mean_f[0]
    z[:,1,:,:,:]=z[:,1,:,:,:]-sky_bg_mean_f[1]
    if 'Conv3d_1' in model.__class__.__name__:
        b_z = torch.from_numpy(z).float().to(device)
    elif 'Conv3d_2' in model.__class__.__name__:
        b_z = torch.from_numpy(z).float().to(device)
        b_z[:,:,-1,:,:]=0 # 7th frame is set to 0 after preprocessing
    #
    with torch.no_grad():
        if 'Semi' in model.__class__.__name__:
            encoded,_,_ = model(b_x,b_z)
        else:
            encoded = model(b_x)
    encoded_np=encoded.cpu().data.numpy()
    if use_pad0_sti==False: # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
        encoded_np=encoded_np[7:,:]
        y=y[7:,:]
    testccs=np.zeros((encoded_np.shape[1]))
    for ii in range(encoded_np.shape[1]):
        testccs[ii],testpvalue=pearsonr(encoded_np[:,ii], y[:,ii])
    print (np.median(testccs))
    return testccs
'''
def model_test_save(model,data,device,train_mean,use_pad0_sti=True):
    """
    Save CC of each neuron for test data.
    """
    model=model.to(device)
    model=model.eval()
    (x,y,z)=data
    x=torch.from_numpy(x).float()
    b_x = x.to(device)
    z=z/255.0
    z[:,0,:,:,:]=z[:,0,:,:,:]-train_mean[0]
    z[:,1,:,:,:]=z[:,1,:,:,:]-train_mean[1]
    if 'Conv3d_1' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
    elif 'Conv3d_2' in model.__class__.__name__:
        #b_z = torch.from_numpy(z).float().to(device)
        #b_z[:,:,-1,:,:]=0 # 7th frame is set to 0 after preprocessing
        b_z = torch.from_numpy(z[:,:,:-1,:,:]).float().to(device)
        b_z[:,:,-1,:,:]=0 # the last frame is set to 0 after preprocessing
    elif 'Conv3d_4' in model.__class__.__name__:
        b_z = torch.from_numpy(z[:,:,:z.shape[2]-1,:,:]).float().to(device)
    #
    with torch.no_grad():
        if 'Semi' in model.__class__.__name__:
            encoded,_,_ = model(b_x,b_z)
        else:
            encoded = model(b_x)
    encoded_np=encoded.cpu().data.numpy()
    if use_pad0_sti==False: # do not use reponses of 0-padding stimulus, here 7 because we use 8 time lags
        encoded_np=encoded_np[x.shape[2]-1:,:]
        y=y[x.shape[2]-1:,:]
    testccs=np.zeros((encoded_np.shape[1]))
    for ii in range(encoded_np.shape[1]):
        testccs[ii],testpvalue=pearsonr(encoded_np[:,ii], y[:,ii])
    print (np.median(testccs))
    return testccs
