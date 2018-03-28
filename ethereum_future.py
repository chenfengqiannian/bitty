
# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

## Keras for deep learning
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential

## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import time

df1 = pd.read_csv("btceUSD.csv")
N=100

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


def test_model(model, X_test, Y_test, unnormalized_bases):
    """
    Test the model on the testing data

    Arguments:
    model -- The previously fitted 3 layer Recurrent Neural Network
    X_test -- A tensor of shape (267, 49, 35) that represents the x values of the testing data
    Y_test -- A tensor of shape (267,) that represents the y values of the testing data
    unnormalized_bases -- A tensor of shape (267,) that can be used to get unnormalized data points

    Returns:
    y_predict -- A tensor of shape (267,) that represnts the normalized values that the model predicts based on X_test
    real_y_test -- A tensor of shape (267,) that represents the actual prices of bitcoin throughout the testing period
    real_y_predict -- A tensor of shape (267,) that represents the model's predicted prices of bitcoin
    fig -- A branch of the graph of the real predicted prices of bitcoin versus the real prices of bitcoin
    """
    # Test the model on X_Test
    y_predict = model.predict(X_test)

    # Create empty 2D arrays to store unnormalized values
    real_y_test = np.zeros_like(Y_test)
    real_y_predict = np.zeros_like(y_predict)

    # Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    for i in range(Y_test.shape[0]):
        y = Y_test[i]
        predict = y_predict[i]
        real_y_test[i] = (y + 1) * unnormalized_bases[i]
        real_y_predict[i] = (predict + 1) * unnormalized_bases[i]

    # Plot of the predicted prices versus the real prices
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Bitcoin Price Over Time")
    plt.plot(real_y_predict, color='green', label='Predicted Price')
    plt.plot(real_y_test, color='red', label='Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Days)")
    ax.legend()

    return y_predict, real_y_test, real_y_predict, fig


def price_change(Y_daybefore, Y_test, y_predict):
    """
    Calculate the percent change between each value and the day before

    Arguments:
    Y_daybefore -- A tensor of shape (267,) that represents the prices of each day before each price in Y_test
    Y_test -- A tensor of shape (267,) that represents the normalized y values of the testing data
    y_predict -- A tensor of shape (267,) that represents the normalized y values of the model's predictions

    Returns:
    Y_daybefore -- A tensor of shape (267, 1) that represents the prices of each day before each price in Y_test
    Y_test -- A tensor of shape (267, 1) that represents the normalized y values of the testing data
    delta_predict -- A tensor of shape (267, 1) that represents the difference between predicted and day before values
    delta_real -- A tensor of shape (267, 1) that represents the difference between real and day before values
    fig -- A plot representing percent change in bitcoin price per day,
    """
    # Reshaping Y_daybefore and Y_test
    Y_daybefore = np.reshape(Y_daybefore, (-1, 1))
    Y_test = np.reshape(Y_test, (-1, 1))

    # The difference between each predicted value and the value from the day before
    delta_predict = (y_predict - Y_daybefore) / (1 + Y_daybefore)

    # The difference between each true value and the value from the day before
    delta_real = (Y_test - Y_daybefore) / (1 + Y_daybefore)

    # Plotting the predicted percent change versus the real percent change
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Percent Change in Bitcoin Price Per Day")
    plt.plot(delta_predict, color='green', label='Predicted Percent Change')
    plt.plot(delta_real, color='red', label='Real Percent Change')
    plt.ylabel("Percent Change")
    plt.xlabel("Time (Days)")
    ax.legend()
    plt.show()

    return Y_daybefore, Y_test, delta_predict, delta_real, fig