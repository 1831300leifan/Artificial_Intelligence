from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import numpy as np
import tensorflow as tf

def load_data():
    datafile = 'data/ex1data2.txt'
    #Read into the data file
    cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
    #Form the usual "X" matrix and "y" vector
    X = np.transpose(np.array(cols[:-1]))
    y = np.transpose(np.array(cols[-1:]))

    stored_feature_means, stored_feature_stds = [], []
    Xnorm = X.copy()
    for icol in range(Xnorm.shape[1]):
        stored_feature_means.append(np.mean(Xnorm[:,icol]))
        stored_feature_stds.append(np.std(Xnorm[:,icol]))
        #Skip the first column
        if not icol: continue
        #Faster to not recompute the mean and std again, just used stored values
        Xnorm[:,icol] = (Xnorm[:,icol] - stored_feature_means[-1])/stored_feature_stds[-1]

        return Xnorm, y

train_X, train_y = load_data()

# placeholder
X = tf.placeholder("float", [None, 2])

# model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([2, 1]))
y = tf.matmul(X, W) + b

# minimize mean squared error
y_ = tf.placeholder("float", [None, 1])
loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# initialize variable
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

sess.run(train, feed_dict={X: train_X, y_: train_y})
print(sess.run(W), sess.run(b))
