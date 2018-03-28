
# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import time
import math
df1 = pd.read_csv("btceUSD.csv")
N=100

import tensorflow as tf
import numpy as np
def getminmaxstep(nparry,num):
    min=nparry.min()
    max=nparry.max()
    step=(max-min)/num
    return min,max,step
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# Make up some real data
y_data=df1["price"].as_matrix()[:, np.newaxis][0:N]
min,max,step=getminmaxstep(y_data,N)
x_data = np.arange(min,max,step)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise
#x_data=df1["time"].as_matrix()[:, np.newaxis][0:1000]

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
# add hidden layer
l1 = add_layer(xs, 1,20, n_layer=1,activation_function=tf.nn.tanh)
l2 = add_layer(l1, 20,20,n_layer=2, activation_function=tf.nn.tanh)
l3=add_layer(l2, 20,20, n_layer=3,activation_function=tf.nn.tanh)
# add output layer
prediction = add_layer(l3, 20, 1,n_layer=4, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(x_data,y_data)
plt.ion()
plt.show()
axes = plt.gca()
axes.set_ylim([min,max])
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000000):
    # training
    start = time.time()

    # Train the model on X_train and Y_train
    # Get the time it took to train the model (in seconds)

    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    training_time = math.floor(time.time() - start)
    if i % 50 == 0:
        # to see the step improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(training_time)
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=1)
        plt.pause(0.1)
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)


