# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:00:52 2019

@author: meher
"""
import sys
import matplotlib.pyplot as plt
import random
import math

def sigmoid(z):
    '''Computes the output of the sigmoid model 
    '''
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    
    '''Computes the cross-entropy loss given the expected outputs and
    the hypothesis vectors
    '''
    return (-y * np.log(h+sys.float_info.epsilon) - (1 - y) * np.log(1 - h-sys.float_info.epsilon)).mean()
    
def gradient(X,h,y):
    '''Calculates the gradient function for the cross entropy loss
    given the input, hypothesis and the expected vectors.
    '''
    return np.dot(X.T, (h - y)) / y.shape[0]


(imgs, lbls)=load_data()#Collect images and labels of the dataset
data=np.asarray(imgs,dtype='float64').reshape((80,380*240))
data1=np.transpose(data)
#labels associated with the training data
#happy=1, maudlin=0
y= np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
#initialization of training set error, weights, holdout set errors
error=np.zeros((10,10))
y_test=np.array([1,0])
w_best=[[]]*10
h_final_test=[[]]*10
ho_error_final=np.zeros((10,10))
#There are a total of 10 subjects
Total=np.arange(10)
#Collection of errors and weights for every subject that is chosen to be the test subject
for i in Total:
    #Initialization of training set and test sets
    test_subject=i
    Tr=np.setdiff1d(Total,np.array([i]))
    ho_subject=random.choice(Tr)
    Tr=np.setdiff1d(Tr,np.array([ho_subject]))
    ind=list((Tr*8)+4)
    ind=ind+list((Tr*8)+5)
    Training_data=data1[:,ind]
    test=data1[:,[(test_subject*8)+4,(test_subject*8)+5]]
    ho=data1[:,[(ho_subject*8)+4,(ho_subject*8)+5]]
    #Using PCA for dimensionality reduction.
    re_data,mf,evals=PCA_git(Training_data)
    #Transforming training and validation data
    new_dat=project_onto_ev(Training_data,re_data,mf)
    new_dat=new_dat[:,0:10]/((evals[0:10]**0.5))
    ho_img=project_onto_ev(ho,re_data,mf)
    ho_img=ho_img[:,0:10]/((evals[0:10]**0.5))
   
    #Weights set to zero before every epoch
    w=np.zeros(11)
    #Additional column of ones added for bias
    X=np.column_stack((new_dat,np.ones((16,1))))
    ho_img=np.column_stack((ho_img,np.ones((2,1))))
    #Initialization of learning rate and best validation set errors
    lr=0.0052
    ho_best=math.inf
    #j represents a single epoch
    for j in range(10):
        #Calculation of the logistic function for validation and training
        z = np.dot(X, w.T)
        h = sigmoid(z)
        z_test=np.dot(ho_img, w.T)
        h_test=sigmoid(z_test)
        #Calculation of error for training and validation sets
        error[i,j]=loss(h,y)
        ho_error=loss(h_test,y_test)
        ho_error_final[i,j]=ho_error
        #Saving the best weights and implementation of early stopping
        if ho_error<ho_best:
            ho_best=ho_error
            w_best[i]=w
            h_best=h_test
        w_new=w-lr*(gradient(X,h,y))
        w=w_new
    #Test set prediction
    test_img=project_onto_ev(test,re_data[:,0:10],mf)
    test_img=test_img/((evals[0:10]**0.5))
    test_img=np.column_stack((test_img,np.ones((2,1))))
    z_final_test=np.dot(test_img, w_best[i].T)
    h_final_test[i]=sigmoid(z_final_test)

#Test set accuracy
z=np.zeros((10,2))
for i in range(10):
    if h_final_test[i][0]>0.5:
        z[i,0]=1
    else:
        z[i,0]=0
    if h_final_test[i][1]<=0.5:
        z[i,1]=1
    else:
        z[i,0]=0
acc=z.sum()/20   
    
        
    

