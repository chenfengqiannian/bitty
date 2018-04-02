
# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import time
## Keras for deep learning
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential, load_model

## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
import numpy as np
def getminmaxstep(nparry,num):
    min=nparry.min()
    max=nparry.max()
    step=(max-min)/num
    return min,max,step

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def loadata():

    dataframe = pd.read_csv('btceUSD.csv', usecols=[1], engine='python',nrows=1000)
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset

    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler().fit(dataset)
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 30
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    return trainX, trainY,testX, testY,scaler




def load_data(filename, sequence_length):
    """
    Loads the bitcoin data

    Arguments:
    filename -- A string that represents where the .csv file can be located
    sequence_length -- An integer of how many days should be looked at in a row

    Returns:
    X_train -- A tensor of shape (2400, 49, 35) that will be inputed into the model to train it
    Y_train -- A tensor of shape (2400,) that will be inputed into the model to train it
    X_test -- A tensor of shape (267, 49, 35) that will be used to test the model's proficiency
    Y_test -- A tensor of shape (267,) that will be used to check the model's predictions
    Y_daybefore -- A tensor of shape (267,) that represents the price of bitcoin the day before each Y_test value
    unnormalized_bases -- A tensor of shape (267,) that will be used to get the true prices from the normalized ones
    window_size -- An integer that represents how many days of X values the model can look at at once
    """
    # Read the data file
    raw_data = pd.read_csv(filename, dtype=float).values

    # Change all zeros to the number before the zero occurs
    # for x in range(0, raw_data.shape[0]):
    #     for y in range(0, raw_data.shape[1]):
    #         if (raw_data[x][y] == 0):
    #             raw_data[x][y] = raw_data[x - 1][y]

    # Convert the file to a list
    data = raw_data.tolist()

    # Convert the data to a 3D array (a x b x c)
    # Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    # Normalizing data by going through each window
    # Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:, 1:, :] = d0[:, 1:, :] / d0[:, 0:1, :] - 1

    # Keeping the unnormalized prices for Y_test
    # Useful when graphing bitcoin price over time later
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end, 0:1, 20]

    # Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    split_line = round(0.9 * dr.shape[0])
    training_data = dr[:int(split_line), :]

    # Shuffle the data
    np.random.shuffle(training_data)

    # Training Data
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1]
    Y_train = Y_train[:, 20]

    # Testing data
    X_test = dr[int(split_line):, :-1]
    Y_test = dr[int(split_line):, 49, :]
    Y_test = Y_test[:, 20]

    # Get the day before Y_test's price
    Y_daybefore = dr[int(split_line):, 48, :]
    Y_daybefore = Y_daybefore[:, 20]

    # Get window size and sequence length
    sequence_length = sequence_length
    window_size = sequence_length - 1  # because the last value is reserved as the y value

    return X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size
    """
    Loads the bitcoin data
    
    Arguments:
    filename -- A string that represents where the .csv file can be located
    sequence_length -- An integer of how many days should be looked at in a row
    
    Returns:
    X_train -- A tensor of shape (2400, 49, 35) that will be inputed into the model to train it
    Y_train -- A tensor of shape (2400,) that will be inputed into the model to train it
    X_test -- A tensor of shape (267, 49, 35) that will be used to test the model's proficiency
    Y_test -- A tensor of shape (267,) that will be used to check the model's predictions
    Y_daybefore -- A tensor of shape (267,) that represents the price of bitcoin the day before each Y_test value
    unnormalized_bases -- A tensor of shape (267,) that will be used to get the true prices from the normalized ones
    window_size -- An integer that represents how many days of X values the model can look at at once
    """
    #Read the data file
    raw_data = pd.read_csv(filename, dtype = float).values

    #Change all zeros to the number before the zero occurs
    for x in range(0, raw_data.shape[0]):
        for y in range(0, raw_data.shape[1]):
            if(raw_data[x][y] == 0):
                raw_data[x][y] = raw_data[x-1][y]

    #Convert the file to a list
    data = raw_data.tolist()

    #Convert the data to a 3D array (a x b x c)
    #Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    #Normalizing data by going through each window
    #Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:,1:,:] = d0[:,1:,:] / d0[:,0:1,:] - 1

    #Keeping the unnormalized prices for Y_test
    #Useful when graphing bitcoin price over time later
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end,0:1,20]

    #Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    split_line = round(0.9 * dr.shape[0])
    training_data = dr[:int(split_line), :]

    #Shuffle the data
    np.random.shuffle(training_data)

    #Training Data
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1]
    Y_train = Y_train[:, 20]

    #Testing data
    X_test = dr[int(split_line):, :-1]
    Y_test = dr[int(split_line):, 49, :]
    Y_test = Y_test[:, 20]

    #Get the day before Y_test's price
    Y_daybefore = dr[int(split_line):, 48, :]
    Y_daybefore = Y_daybefore[:, 20]

    #Get window size and sequence length
    sequence_length = sequence_length
    window_size = sequence_length - 1 #because the last value is reserved as the y value

    return X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size
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
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1]) ))
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
    training_time = int(math.floor(time.time() - start))
    return model, training_time


def test_model(model, X_test, Y_test, scaler):
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
    #real_y_test = np.zeros_like(Y_test)
    #real_y_predict = np.zeros_like(y_predict)

    # Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    # for i in range(Y_test.shape[0]):
    #     y = Y_test[i]
    #     predict = y_predict[i]
    #     real_y_test[i] = (y + 1) * unnormalized_bases[i]
    #     real_y_predict[i] = (predict + 1) * unnormalized_bases[i]
    real_y_test=scaler.inverse_transform(Y_test)
    real_y_predict =scaler.inverse_transform(y_predict)
    X_test=scaler.inverse_transform(X_test)
    Y_test = scaler.inverse_transform(Y_test)
    # Plot of the predicted prices versus the real prices
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Bitcoin Price Over Time")
    plt.plot(real_y_predict, color='green', label='Predicted Price')
    plt.plot(real_y_test, color='red', label='Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Days)")
    ax.legend()
    plt.show()

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
def predict_point_by_point(model, data):
    #每次只预测1步长
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
def predict_sequence_full(model, data, seq_len):
    #根据训练模型和第一段用来预测的时间序列长度逐步预测整个时间序列
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [seq_len-1], predicted[-1], axis=0)
    return predicted
x_data, y_data,testX, testY,scaler=loadata()
model = load_model('my_model.h5')
out=predict_sequence_full(model,x_data,30)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
y_data=scaler.inverse_transform(y_data)
out=scaler.inverse_transform(out)

plt.plot(out, color='red', label='pre Percent Change')
plt.plot(y_data, color='green', label='Real Percent Change')
plt.ylabel("Percent Change")
plt.xlabel("Time (Days)")
ax.legend()
plt.show()







# # y_data=df1["price"].as_matrix()[:, np.newaxis][0:N]
# # min,max,step=getminmaxstep(y_data,N)
# # x_data = df1.as_matrix()[0:N]
# model=initialize_model(x_data,30,0.2,'linear', 'mse', 'adam')
# # print (model.summary())
# model, training_time = fit_model(model, x_data, y_data, 1024, 100, .05)
#model = load_model('my_model.h5')
#test_model(model,testX,testY,scaler)
# # print ("Training time", training_time, "seconds")
#model.save('my_model.h5')

