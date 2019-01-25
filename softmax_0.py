# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:44:51 2019

@author: Nash
"""
import matplotlib.pyplot as plt
import numpy as np
import data_loader as dl
import princ_comp as pc
import sys
import picking_data as ch

n = 48;
feat = 20
cla = 6;
def plot_cr_loss():
    x=np.mean(error, axis=0)
    xerr=np.std(error,axis=0)
    y=np.arange(10)
    plt.figure()
    plt.errorbar(y,x,xerr)
    plt.xlabel("epochs")
    plt.ylabel("Cross-entropy error")
    plt.title("Training set error")
def get_hyp(A,s):
    scale = np.sum(A, 0)  
    for q in range(s):
        hyp[:,q] = A[:,q]/ (scale[q]+sys.float_info.epsilon)
    return hyp

#do one hot encoding for every column of the hypothesis matrix
def one_hot(hyp):
    hyp_hot = (hyp == hyp.max(axis=0)[None,:]).astype(int)
    return hyp_hot

def grad(hyp,Y,X):
    return np.matmul((Y-hyp),X)

##get cross-entropy loss
def loss_func(target, hypothesis):
    c, n = target.shape
    cr_loss = 0
    hyp = np.log(hypothesis+sys.float_info.epsilon)
    for j in range(n):
        cr_loss += -np.dot((target[:,j]), hyp[:,j])
    return cr_loss/(c*n)

##testing the weights
def hold_out(weights, re_data,evalues,X_hold,feat):
    """Function returns the hypthesis on the holdout set given the wweights
    """
    re_data_0 = re_data/(((evalues[0:feat])**(0.5))*2304);
    X_hold_0 = X_hold.T
    X_hold_0 -= X_hold.mean(axis=1)
    
    test_img=(np.matmul(X_hold_0,re_data_0))
    
    z_hold=np.matmul(weights[:,0:feat], test_img.T)
    z_hold=z_hold+weights[:,-1]
    z_hold_e = np.exp(z_hold)
    h_hold= z_hold_e/np.sum(z_hold_e, 0) ;
#    h_hot_hold = one_hot(h_hold)
    return h_hold, re_data_0

#Test data
images, labels = dl.load_data(dl.data_dir);
data=np.asarray(images,dtype='float64').reshape((80,380*240))
data1=np.transpose(data)
data1_n = data1/np.linalg.norm(data1,axis = 0)
H_H_hold = np.zeros([6,6])
H_testy = np.zeros([6,6])
#Start of loop
Loss_imgs = np.zeros([10,51])
Loss_imgs_tr = np.zeros([10,51])
weights_test = np.zeros([cla*10,feat+1])
for testing in range(9):
    X_train, X_hold, X_test = ch.cherries(data1,testing) 
    (re_data, evalues, evectors, mf)=pc.PCA_1(X_train, dims_rescaled_data=feat)
    #PCA test images
    X = np.column_stack((evectors[0:48,:],np.ones((48,1))))
    #Target labels
    Y = np.column_stack((np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6),np.eye(6)));
    
    
    #initialize the weights[]
    weights = np.zeros([cla,feat+1])
    #weights = np.random.rand(cla,feat+1)
    lr =0.0019
    
    cr_loss = np.zeros([1,51])
    cr_loss_tr= np.zeros([1,51])
    cr_loss_min =5
    weights_best = np.zeros([cla,feat+1])
    #This will eventually go inside some sort of loop
    #Start of loop
    for epoch in range(52):
        #to get the difference  between the hypothesis and the targets
        A_all = np.matmul(weights,X.T)
        #take the exponent of every element of this
        A_all_exp = np.exp(A_all);
        #get the hypothesis matrix
        hyp = np.zeros([cla,n])
        hyp = A_all_exp/np.sum(A_all_exp, 0) ;
        #do one hot encoding
        hyp_hot = one_hot(hyp)
        #find the training error
        cr_loss_tr[:,int(epoch)-1] = loss_func(Y,hyp)
        #new weights
        weight_new = weights+lr*grad(hyp,Y,X)
        weights = weight_new

        #Store weights at this epoch
        h_hold,m = hold_out(weights,re_data,evalues,X_hold,feat)
        
        cr_loss[:,int(epoch)-1] = loss_func(np.eye(6),h_hold)
        if cr_loss[:,int(epoch)-1] < cr_loss_min:
            cr_loss_min = cr_loss[:,int(epoch)-1]
            weights_best = weights
            
            #print((epoch/10)-1)
        #end of loop
        Loss_imgs[testing,:] = cr_loss
        Loss_imgs_tr[testing,:] = cr_loss_tr
        
    #get squared error
    h_hot_hold = one_hot(h_hold)
    H_H_hold += h_hot_hold
    weights_test[testing*6:(testing+1)*6,:] = weights_best
    h_test,m = hold_out(weights_best,re_data,evalues,X_test,feat)
    h_test_hold = one_hot(h_test)
    H_testy += h_test_hold

#PLot holdout set error
yerr=np.std(Loss_imgs,axis=0)
x=np.arange(51)
plt.figure()
plt.errorbar(x,np.mean(Loss_imgs, axis = 0),yerr,errorevery=10)
plt.ylabel('Cross-entropy Loss')
plt.xlabel('Epochs')
plt.title("Hold out set error")
H_H_hold = H_H_hold/10

#Plot training error
yerr_=np.std(Loss_imgs_tr,axis=0)
x_=np.arange(51)
plt.figure()
plt.errorbar(x_,np.mean(Loss_imgs_tr, axis = 0),yerr_,errorevery=10)
plt.ylabel('Cross-entropy Loss')
plt.xlabel('Epochs')
plt.title("Training set error")


#Get the test set matrix
#for holding in range(10):
#    x_test = data1[:,holding*6:(holding+1)*6]
#    h_test,m = hold_out(weights_test[holding*6:(holding+1)*6,:],re_data,evalues,x_test,feat)
#    h_test_hold = one_hot(h_test)  
#    H_testy += h_test_hold
#    
#H_testy = H_testy*10
for line in H_testy:
    print(*line)
#Visualize the weights
from data_loader import data_dir, display_face
re_data_0 = re_data/(((evalues[0:feat])**(0.5)));
for emot in range(6):
    im_0 = np.matmul(re_data_0, weights[emot,0:feat])
    a = im_0.reshape(380,240);
    out_im =np.interp(a, (a.min(), a.max()), (0, 255))
    display_face(out_im);