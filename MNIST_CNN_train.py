#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sklearn.decomposition
import sklearn.linear_model
from load_mnist import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import time
import scipy.io as sio
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mnist', help='folder name')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--lr', type=float, default=0.1, help='SGD: learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD: momentum term')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--gamma', type=float, default=0.7, help='adam: momentum term')

opt = parser.parse_args()
print(opt)

batch_size = opt.batch_size
c_dim = opt.c_dim
lr = opt.lr

class_use = np.array([3,8])
class_use_str = np.array2string(class_use)
y_dim = class_use.shape[0]
newClass = range(0,y_dim)
test_size = 100

#save_folder = '/home/mnorko/Documents/Tensorflow/causal_vae/' + opt.model + '_batch' + str(batch_size)  + '_lr' + str(opt.lr) + '/'
save_folder = 'C:/Users/Dylan/Desktop/FACT/Figure3/' + opt.model + '_batch' + str(batch_size) + '_lr' + str(opt.lr) + '_class' + class_use_str[1:(len(class_use_str)-1):2] + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load data
if opt.model == 'mnist':
    trX, trY, tridx = load_mnist_classSelect('train',class_use,newClass)
    vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
    teX, teY, teidx = load_mnist_classSelect('test',class_use,newClass)
elif opt.model == 'fmnist':
    trX, trY, tridx = load_fashion_mnist_classSelect('train',class_use,newClass)
    vaX, vaY, vaidx = load_fashion_mnist_classSelect('val',class_use,newClass)
    teX, teY, teidx = load_fashion_mnist_classSelect('test',class_use,newClass)


batch_idxs = len(trX) // opt.batch_size    
batch_idxs_val = len(vaX) // test_size    
ce_loss = nn.CrossEntropyLoss()

from models.CNN_classifier import CNN
classifier = CNN(y_dim).to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=opt.momentum)
scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)

loss_total = np.zeros((opt.epochs*batch_idxs))
test_loss_total = np.zeros((opt.epochs))
percent_correct = np.zeros((opt.epochs))
start_time = time.time()
counter = 0
for epoch in range(0,opt.epochs):
    for idx in range(0, batch_idxs):
        batch_labels = torch.from_numpy(trY[idx*opt.batch_size:(idx+1)*opt.batch_size]).long().to(device)
        batch_images = trX[idx*opt.batch_size:(idx+1)*opt.batch_size]
        batch_images_torch = torch.from_numpy(batch_images)
        batch_images_torch = batch_images_torch.permute(0,3,1,2).float()
        batch_images_torch = batch_images_torch.to(device)
        
        optimizer.zero_grad()
        prob_output,output = classifier(batch_images_torch)
        loss = ce_loss(output,batch_labels)
        loss.backward()
        optimizer.step()
        
        loss_total[counter] = loss.item()
        counter = counter+1
        
        print ("[Train Epoch %d/%d] [Batch %d/%d] time: %4.4f [loss: %f]" % (epoch, opt.epochs, idx, batch_idxs,time.time() - start_time,
                                                             loss.item()))
    #Get validation loss
    test_loss = 0.0
    correct = 0
    for idx in range(0, batch_idxs_val):
        val_labels = torch.from_numpy(vaY[idx*test_size:(idx+1)*test_size]).long().to(device)
        val_images = vaX[idx*test_size:(idx+1)*test_size]
        val_images_torch = torch.from_numpy(val_images)
        val_images_torch = val_images_torch.permute(0,3,1,2).float()
        val_images_torch = val_images_torch.to(device)
        
        prob_output_val,output_val = classifier(val_images_torch)
        pred = prob_output_val.argmax(dim=1)
        test_loss +=  ce_loss(output_val,val_labels)
        correct += pred.eq(val_labels.view_as(pred)).sum().item()/float(test_size)
    test_loss = test_loss/batch_idxs_val
    percent_correct[epoch] = 100.0*correct/batch_idxs_val
    print ("[Test Epoch %d/%d] [loss: %f] [corr: %f]" % (epoch, opt.epochs,test_loss.item(),percent_correct[epoch]))
    test_loss_total[epoch] =  test_loss.item()   
    scheduler.step()
    
    torch.save({
        'step': counter,
        'epoch': epoch,
        'batch': idx,
        'model_state_dict_classifier': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_total,
        }, save_folder + 'network_batch' + str(opt.batch_size) + '.pt')
    sio.savemat(save_folder + 'lossVal.mat',{'loss_total':loss_total[:counter],'percent_correct':percent_correct[:epoch],'test_loss_total':test_loss_total[:epoch]});
