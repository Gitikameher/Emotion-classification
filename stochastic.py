# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:54:49 2019

@author: meher
"""
import matplotlib.pyplot as plt
import numpy as np
import data_loader as dl
import princ_comp as pc
import pca as pr
import sys
import label as lb
import picking_data as ch

#do one hot encoding for every column of the hypothesis matrix
#def one_hot(hyp):
#    hyp_hot = (hyp == hyp.max(axis=0)[None,:]).astype(int)
#    return hyp_hot

def gradient(X,h,y):
    ''' Calculates the gradient function given an input vector, 
    a vector showing the hypothesis and a vector showing the
    expected output
    '''
    u=X.reshape(1,21)
    v=(h-y).reshape(1,6)
    return np.matmul(u.T, v) 

def one_hot(a, num_classes):
    '''Given a matrix, this function converts all the entries into
    one-hot vector encodings
    '''
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])  

def randomize(Tr):
    '''Randomizes the order of inputs in the training set and returns
    the training set along with the corresponding output encodings in
    a random fashion.
    Emotions as encoded:
    a=0,d=1,f=2,h=3,m=4,s=5
    '''
    Tr=np.column_stack((Tr,(list(range(6))*8)))
    np.random.shuffle(Tr)
    training_data=Tr[:,0:-1]
    y= np.array(Tr[:,-1],dtype='int32')
    lbls=one_hot(y,6)
    return training_data,lbls

def weight_update(w_old,X,h,y,lr):
    '''Updates weight using the stochastic gradient. 
    '''
    for i in range(y.shape[0]):
        w_new=w_old-lr*(gradient(X[i,:],h[:,i].T,y[i,:])).T
        w_old=w_new
    return w_old

def softmax(x):
    '''Computes the softmax values given any input'''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

n = 48
feat = 20
cla = 6
images, labels = dl.load_data(dl.data_dir);
data=np.asarray(images,dtype='float64').reshape((60,380*240))
data1=np.transpose(data)
Loss_imgs_tr_ = np.zeros([10,51])
Y = np.column_stack((np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6)));
for testing in range(10):
    X_train, X_hold, X_test = ch.cherries(data1,0) 
    
    (re_data, evalues, evectors)=pc.PCA_1(X_train, dims_rescaled_data=20)
    X = np.column_stack((evectors,np.ones((48,1))))
    
    
    weights = np.zeros([cla,feat+1])
    lr =0.00003
    cr_loss_tr_= np.zeros([1,51])
    for epoch in range(52):
        tr,y= randomize(X)
        h=np.matmul(weights,tr.T)
        h=softmax(h)
        w=weight_update(weights,tr,h,y,lr)
        weights=w
        #h_hot = sf.one_hot(h)
        cr_loss_tr_[:,int(epoch)-1] = loss_func(Y,h)
    Loss_imgs_tr_[testing,:] = cr_loss_tr_
    
    
 #Plot training error
yerr_=np.std(Loss_imgs_tr_,axis=0)
x_=np.arange(51)
plt.figure()
plt.errorbar(x_,np.mean(Loss_imgs_tr_, axis = 0),yerr_,errorevery=10)
plt.ylabel('Cross-entropy Loss')
plt.xlabel('Epochs')
plt.title("Training set error")   
