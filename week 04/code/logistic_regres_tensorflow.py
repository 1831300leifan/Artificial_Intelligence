import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def load_data():
    datafile = 'data/ex2data1.txt'
    #!head $datafile
    cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
    ##Form the usual "X" matrix and "y" vector
    X = np.transpose(np.array(cols[:-1]))
    y = np.transpose(np.array(cols[-1:]))
    m = y.size # number of training examples
    ##Insert the usual column of 1's into the "X" matrix
    # X = np.insert(X,0,1,axis=1)

    return X, y

train_X, train_y = load_data()

x = tf.placeholder("float", shape=[None, 2])
y_ = tf.placeholder("float", shape=[None, 1])

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

import tensorflow as tf
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

y = tf.nn.sigmoid(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step.run(feed_dict={x: train_X, y_: train_y})
print(sess.run(W), sess.run(b))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: train_X, y_: train_y}))

