# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:50:55 2019

@author: Nash
"""
import random
import numpy as np


def cherries(data,idx):
    """Select the training, test and holdout test sets from the complete dataset
    """
    index_test = idx
    r = np.delete(range(9),idx)
    #generate random indices
    index_hold = random.sample(list(r),1)
    #get hold out and test sets based on random indices
    x_test = data[:,index_test*6:(index_test+1)*6]
    x_hold =data[:,index_hold[0]*6:(index_hold[0]+1)*6]
  
    
    x_train = np.delete(data,[i for j in (range(index_test*6,(index_test+1)*6), range(index_hold[0]*6,(index_hold[0]+1)*6)) for i in j]
, axis =1)
     

    return x_train,x_test,x_hold