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

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def next_batch(num, data):
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data[i] for i in idx]  # get list of `num` random samples
    data_shuffle = np.asarray(data_shuffle)  # get back numpy array

    return data_shuffle

train_X, train_y = load_data()


x = tf.placeholder("float", shape=[None, 2])
y_ = tf.placeholder("float", shape=[None, 2])

W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))

import tensorflow as tf
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train_step.run(feed_dict={x: train_X, y_: train_y})
# print(sess.run(W), sess.run(b))

for i in range(1000):
    batch_X = next_batch(10, train_X)
    batch_y = next_batch(10, train_y)

    train_step.run(feed_dict={x: batch_X, y_: to_categorical(batch_y, num_classes=2)})
    print(i, sess.run(W), sess.run(b))

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: train_X, y_: to_categorical(train_y)}))

