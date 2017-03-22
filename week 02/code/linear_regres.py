from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import numpy as np

def load_data():
    datafile = 'data/ex1data1.txt'
    cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True) #Read in comma separated data
    #Form the usual "X" matrix and "y" vector
    X = np.transpose(np.array(cols[:-1]))
    y = np.transpose(np.array(cols[-1:]))
    m = y.size # number of training examples
    #Insert the usual column of 1's into the "X" matrix
    X = np.insert(X,0,1,axis=1)

    return X, y

X, y = load_data()

from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X, y)
print(lr.coef_)

import matplotlib.pyplot as plt
theta = lr.coef_[0]

def myfit(xval, theta):
    return theta[0] + theta[1]*xval

plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
plt.plot(X[:,1],myfit(X[:,1], theta),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()