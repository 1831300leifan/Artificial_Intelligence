from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

batch_size = 100
training_epochs = 15

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

h1_W = tf.Variable(tf.random_normal([784, 256]))
h1_b = tf.Variable(tf.random_normal([256]))

out_W = tf.Variable(tf.random_normal([256, 10]))
out_b = tf.Variable(tf.random_normal([10]))

h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h1_W), h1_b))
y_ = tf.matmul(h1, out_W) + out_b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) # Adam Optimizer

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

        print('epoch: %d, cost: %.9f' % (epoch+1, avg_cost))

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))