# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:44:29 2019

@author: meher
"""
import matplotlib.pyplot as plt
(imgs, lbls)=load_data()
data=np.asarray(imgs,dtype='float64').reshape((80,380*240))
data1=np.transpose(data)
train_data=np.delete(data1, np.s_[64:80], axis=1)
train_data=np.delete(train_data, [4,6,12,14,20,22,28,30,36,38,44,46,52,54,60,62], axis=1)

def PCA_git(data):
    mean_face=np.mean(data,axis=1).reshape(91200,1)
    
    Tr= data-mean_face
    C= np.cov(Tr.T)
    s, vh = np.linalg.eigh(C)
    idx = np.argsort(s)[::-1]
    vh = vh[:,idx]
    s = s[idx]
    basis=np.dot(Tr,vh)
    
    basis=basis/np.linalg.norm(basis,axis=0)
    return basis,mean_face,s

def PCA(data, dims_rescaled_data=10):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    mean_face=data.mean(axis=0)
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs,mean_face

def project_onto_ev(dat,basis,mean_face):
    '''This function takes input the set of eigen vectors and transforms 
    the data to the new Principal components
    '''
    x=(dat-mean_face)
    #x=x/np.linalg.norm(x,axis=0)
    return np.dot(x.T,basis)





    
    
    
