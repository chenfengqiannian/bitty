

## Keras for deep learning
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential

## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error

#for logging
import time

##matrix math
import math

from sklearn.preprocessing import MinMaxScaler

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import time
import math
df1 = pd.read_csv("btcusdtbitfinex.csv")



import tensorflow as tf
import numpy as np
def getminmaxstep(nparry,num):
    min=nparry.min()
    max=nparry.max()
    step=(max-min)/num
    return min,max,step
def initialize_model(X_train,window_size, dropout_value, activation_function, loss_function, optimizer):
    """
    Initializes and creates the model to be used

    Arguments:
    window_size -- An integer that represents how many days of X_values the model can look at at once
    dropout_value -- A decimal representing how much dropout should be incorporated at each level, in this case 0.2
    activation_function -- A string to define the activation_function, in this case it is linear
    loss_function -- A string to define the loss function to be used, in the case it is mean squared error
    optimizer -- A string to define the optimizer to be used, in the case it is adam

    Returns:
    model -- A 3 layer RNN with 100*dropout_value dropout in each layer that uses activation_function as its activation
             function, loss_function as its loss function, and optimizer as its optimizer
    """
    # Create a Sequential model using Keras
    model = Sequential()

    # First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1]), ))
    model.add(Dropout(dropout_value))

    # Second recurrent layer with dropout
    model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True)))
    model.add(Dropout(dropout_value))

    # Third recurrent layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))

    # Output layer (returns the predicted value)
    model.add(Dense(units=1))

    # Set activation function
    model.add(Activation(activation_function))

    # Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)

    return model


def fit_model(model, X_train, Y_train, batch_num, num_epoch, val_split):
    """
    Fits the model to the training data

    Arguments:
    model -- The previously initalized 3 layer Recurrent Neural Network
    X_train -- A tensor of shape (2400, 49, 35) that represents the x values of the training data
    Y_train -- A tensor of shape (2400,) that represents the y values of the training data
    batch_num -- An integer representing the batch size to be used, in this case 1024
    num_epoch -- An integer defining the number of epochs to be run, in this case 100
    val_split -- A decimal representing the proportion of training data to be used as validation data

    Returns:
    model -- The 3 layer Recurrent Neural Network that has been fitted to the training data
    training_time -- An integer representing the amount of time (in seconds) that the model was training
    """
    # Record the time the model starts training
    start = time.time()

    # Train the model on X_train and Y_train
    model.fit(X_train, Y_train, batch_size=batch_num, nb_epoch=num_epoch, validation_split=val_split)

    # Get the time it took to train the model (in seconds)
    training_time = int(tf.math.floor(time.time() - start))
    return model, training_time

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
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def loadata():
	dataframe = pd.read_csv('btcusdtbitfinex.csv', usecols=[1], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.8)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
	# reshape into X=t and Y=t+1
	look_back = 3
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


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


