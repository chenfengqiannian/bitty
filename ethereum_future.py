# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import argparse
import time

## Keras for deep learning
from keras.callbacks import TensorBoard
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
import log
import sys
import tensorflow as tf
import numpy as np

logger = log.getLogger("ethereum_future")

LOOK_BACK = 50
SETPFUTURE = 3600


class LSTMmodel(object):

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def loadata(self, data=None, file_name=None, look_back=5):
        if file_name:
            dataframe = pd.read_csv(file_name, usecols=[1], engine='c')
            dataset = dataframe.as_matrix()
        else:
            dataset = data[:, 1][:, np.newaxis]
            dataset = dataset.astype('float32')

        # normalize the dataset

        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler().fit(dataset)
        dataset = scaler.fit_transform(dataset)
        # split into train and test sets
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        # reshape into X=t and Y=t+1

        trainX, trainY = self.create_dataset(train, look_back)
        testX, testY = self.create_dataset(test, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        return trainX, trainY, testX, testY, scaler

    def initialize_model(self, X_train, window_size, dropout_value, activation_function, loss_function, optimizer):
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
        model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1])))
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

    def fit_model(self, model, X_train, Y_train, batch_num, num_epoch, val_split):
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
        model.fit(X_train, Y_train, batch_size=batch_num, nb_epoch=num_epoch, validation_split=val_split,
                  callbacks=[TensorBoard(log_dir='./logs')])

        # Get the time it took to train the model (in seconds)
        training_time = int(math.floor(time.time() - start))
        return model, training_time

    def test_model(self, model, X_test, Y_test, scaler):
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
        # real_y_test = np.zeros_like(Y_test)
        # real_y_predict = np.zeros_like(y_predict)

        # Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
        # for i in range(Y_test.shape[0]):
        #     y = Y_test[i]
        #     predict = y_predict[i]
        #     real_y_test[i] = (y + 1) * unnormalized_bases[i]
        #     real_y_predict[i] = (predict + 1) * unnormalized_bases[i]
        real_y_test = scaler.inverse_transform(Y_test)
        real_y_predict = scaler.inverse_transform(y_predict)
        X_test = scaler.inverse_transform(X_test)
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

    def price_change(self, Y_daybefore, Y_test, y_predict):
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

    @staticmethod
    def predict_sequence(model, test_data, seq_len, step, scaler=None):
        # 根据训练模型和段用来预测的时间序列长度逐步预测整个时间序列
        curr_frame = test_data[-1]
        predicted = []
        a = 0
        for i in range(step):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [seq_len - 1], predicted[-1], axis=0)
        if scaler:
            predicted = scaler.inverse_transform(predicted)
        return predicted

    def show_image(self, data):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title("Bitcoin Price Over Time")
        plt.plot(data, color='red', label='Price')
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Time (Days)")
        ax.legend()
        plt.show()


class SupplementaryData(object):
    trueData = None
    model = None
    supplementaryData = None

    def __init__(self, filename=None):

        self.filename = filename

        # predict_list = predict_sequence(self.model, data, 50)

    def load_data(self):
        dataframe = pd.read_csv(self.filename, engine='c', names=["timestamp", "price", "volume"])
        self.trueData = dataframe.as_matrix()

    def timestamp_to_int(self):
        # 火币处理
        timestamp = (self.trueData[:, 0] / 1000).astype(int).T.reshape(self.trueData.shape[0], 1)
        no_timestamp = self.trueData[:, 1:]
        self.trueData = np.hstack((timestamp, no_timestamp))

    def removeDuplicate(self):
        dataframe = pd.DataFrame(self.trueData)
        dataframe.columns = ['timestamp', 'price', 'volume']
        dataframe = dataframe.drop_duplicates("timestamp")
        self.trueData = dataframe.as_matrix()

    def insert_data(self):
        supplementaryList = list()
        # supplementaryList.append(self.trueData[0, :].reshape(1, 3))
        for i in range(1, self.trueData.shape[0]):
            poor = self.trueData[i, 0] - self.trueData[i - 1, 0]
            supplementary = None
            for j in range(self.trueData.shape[1]):
                if supplementary is None:
                    supplementary = np.linspace(self.trueData[i - 1, j], self.trueData[i, j], poor + 1)
                else:
                    supplementary = np.vstack(
                        (supplementary, np.linspace(self.trueData[i - 1, j], self.trueData[i, j], poor + 1)))
            supplementary = supplementary.T
            supplementary = supplementary[:-1, :]

            supplementaryList.append(supplementary)
        supplementaryList.append(self.trueData[-1].reshape(1, 3))
        self.supplementaryData = np.vstack(tuple(supplementaryList))

    def showImages(self):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title("Bitcoin Price Over Time")
        showData = self.supplementaryData[:, 1]
        plt.plot(showData, color='green', label='Predicted Price')
        logger.info("max=" + str(np.max(showData)))
        logger.info("min=" + str(np.min(showData)))

        # #plt.plot(real_y_test, color='red', label='Real Price')
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Time (Days)")
        ax.legend()
        plt.show()


class TradingException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class LackBalanceException(TradingException):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class MarketSimulationBase(object):
    transactionRecords = list()
    listDealPrice = 0.0
    USDTAmount = 0.0
    BTCAmount = 0.0
    currentTimestamp = 0.0
    data = None
    model = None
    startTimestamp = 0.0
    endTimestamp = 0.0
    gas_price = 0.0
    trading_time = 900
    scaler = None
    outList = list()

    def __init__(self, data=None, model=None, USDTAmount=0.0, BTCAmount=0.0, scaler=None):
        self.anchor = 0
        self.data = data
        self.model = model
        self.USDTAmount = USDTAmount
        self.BTCAmount = BTCAmount
        self.startTimestamp = int(self.data[0, 0])
        self.endTimestamp = int(self.data[-1, 0])
        self.listDealPrice = self.data[-1, 1]
        self.transactionRecords.append({"action": 0,
                                        "BTCChange": 0,
                                        "USDTChange": 0,
                                        "BTCNow": BTCAmount,
                                        "USDTNow": USDTAmount,
                                        "timestamp": self.startTimestamp,
                                        "price": data[0, 1],
                                        "gas_price": 0.0,
                                        "poundage": 0.0,
                                        "nowEarnings": self.USDTAmount,
                                        "predictPrice": 0.0,

                                        })

    def run(self):
        self.setup()
        for i in range(self.startTimestamp, self.endTimestamp):
            self.handel(i)
        self.finish()

    def buy(self, count, timestamp, predictPrice):
        if self.USDTAmount <= 0.0:
            logger.info("buy fail no enough money")
            return
        self.dealBase(count, timestamp, 0, predictPrice)

    def sell(self, count, timestamp, predictPrice):
        if self.BTCAmount <= 0.0:
            logger.info("sell fail no enough money")
            return
        self.dealBase(count, timestamp, 1, predictPrice)

    def dealBase(self, count, timestamp, action, predictPrice):
        index = timestamp - self.startTimestamp
        price = self.data[index, 1]
        totalPrices = count * price
        #minPrice = min(self.USDTAmount, totalPrices)
        if action == 0:
            minPrice = min(self.USDTAmount, totalPrices)
            realCount = self.accuracy(minPrice / price/1.002)
            poundage=self.poundage(realCount)
            gas_price = self.accuracy(poundage * self.data[-1,1])
            self.BTCAmount = self.accuracy(self.BTCAmount + realCount)
            self.USDTAmount = self.accuracy(self.USDTAmount - minPrice)

        elif action == 1:
            minCount = min(self.BTCAmount, count)
            realPrice = minCount * price/1.002
            poundage = self.poundage(realPrice)
            gas_price=poundage

            self.BTCAmount = self.accuracy(self.BTCAmount - minCount)
            self.USDTAmount = self.accuracy(self.USDTAmount + realPrice)
        self.listDealPrice = price
        transactionRecords = {"action": action,
                              "BTCChange": realCount if action == 0 else -minCount,
                              "USDTChange": -minPrice if action == 0 else realPrice,
                              "BTCNow": self.BTCAmount,
                              "USDTNow": self.USDTAmount,
                              "timestamp": timestamp,
                              "truePrice": price,
                              "gas_price": gas_price,
                              "poundage":poundage,
                              "nowEarnings": self.nowEarnings(),
                              "predictPrice": predictPrice
                              }
        self.transactionRecords.append(transactionRecords)
        logger.info(transactionRecords)

    def accuracy(self, num):
        return round(num, 8)
    def poundage(self,count):
        return count*0.002


    def equivalentUSDT(self, price=None, USDT=None, BTC=None):
        if price == None:
            price = self.data[-1, 1]
        if USDT == None:
            USDT = self.USDTAmount
        if BTC == None:
            BTC = self.BTCAmount
        return self.accuracy(BTC * price) + USDT

    def earnings(self, USDT1, USDT2):
        return self.accuracy(USDT1 - USDT2)

    def earningsGains(self, USDT1, USDT2):
        return self.accuracy(self.earnings(USDT1, USDT2) / USDT1)

    def nowEarnings(self, type=0):
        totalUSDT = self.equivalentUSDT(price=self.listDealPrice)
        initUSDT = self.transactionRecords[0]["USDTNow"]
        if type == 0:
            return self.earnings(totalUSDT,initUSDT )
        elif type == 1:
            return self.earningsGains(totalUSDT,initUSDT)

    def handel(self, timestamp):
        pass

    def setup(self):
        # print(np.max(self.data[:,1],0))
        # maxindex=np.argmax(self.data[:, 1], 0)
        # print(np.min(self.data[:, 1], 0))
        # minindex=np.argmin(self.data[:, 1], 0)
        # print(self.data[maxindex,0])
        # print(self.data[minindex, 0])

        pass

    def finish(self):
        pass
        #  print(self.nowEarnings())
        # print(self.nowEarnings(1))


class MarketSimulation(MarketSimulationBase):
    def handel(self, timestamp):

        index = timestamp - self.startTimestamp

        if timestamp - self.anchor < SETPFUTURE:
            return
        if index <= LOOK_BACK - 1:
            return
        self.anchor = timestamp

        test_data = np.reshape(self.test_data[index - 50:index, 0], (1, LOOK_BACK, 1))

        predictList = LSTMmodel.predict_sequence(self.model, test_data, LOOK_BACK, SETPFUTURE,self.scaler)



        logger.info("true date after=====>" + str(self.data[index + 50, 1]))
        logger.info("true data first=====>" + str(self.data[index, 1]))
        logger.info("predict data ===>" + str(predictList[-1]))
        self.outList.append(predictList)

    def setup(self):
        self.test_data = self.data[:, 1, np.newaxis]
        self.scaler = StandardScaler().fit(self.test_data)
        self.test_data = self.scaler.fit_transform(self.test_data)

    def finish(self):
        logger.info(self.nowEarnings())
        logger.info(self.transactionRecords)
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(111)
        # ax.set_title("Bitcoin Price Over Time")
        # plt.plot(self.outList, color='green', label='Predicted Price')
        # #plt.plot(real_y_test, color='red', label='Real Price')
        # ax.set_ylabel("Price (USD)")
        # ax.set_xlabel("Time (Days)")
        # ax.legend()
        # plt.show()

from collections import OrderedDict
class MarketSimulationBB(MarketSimulationBase):

    tradingOrderDict=OrderedDict()
    def handel(self, timestamp):
        tradingOrderDict=self.tradingOrderDict.get(timestamp)
        if tradingOrderDict:
            if tradingOrderDict["action"]==1:
                self.buy(tradingOrderDict["count"],timestamp,tradingOrderDict["predictValue"])
            else:
                self.sell(tradingOrderDict["count"],timestamp,tradingOrderDict["predictValue"])

        index = timestamp - self.startTimestamp


        if timestamp - self.anchor < SETPFUTURE:
            return
        if index <= LOOK_BACK - 1:
            return
        self.anchor = timestamp
        test_data = np.reshape(self.test_data[index-50:index,0], (1, LOOK_BACK, 1))
        predictList = LSTMmodel.predict_sequence(self.model, test_data, LOOK_BACK, SETPFUTURE, self.scaler)
        maxIndex=np.argmax(predictList)
        minIndex=np.argmin(predictList)
        maxValue=np.max(predictList)
        minValue=np.min(predictList)
        if maxIndex>minIndex:
            self.tradingOrderDict[minIndex+timestamp]={"action":1,"predictValue":minValue,"count":1000}
            self.tradingOrderDict[maxIndex+timestamp]={"action": -1, "predictValue": maxValue, "count": 1000}
        else:
            self.tradingOrderDict[maxIndex+timestamp]={"action": -1, "predictValue": maxValue, "count": 1000}
            self.tradingOrderDict[minIndex+timestamp]={"action": 1, "predictValue": minValue, "count": 1000}
        #self.tradingOrder=self.tradingOrder.tradingOrdert.sort(key=lambda k: k[0])
        logger.info(self.tradingOrderDict)
        self.outList.append(predictList)

    def setup(self):
        self.test_data = self.data[:, 1, np.newaxis]
        self.scaler = StandardScaler().fit(self.test_data)
        self.test_data = self.scaler.fit_transform(self.test_data)


    def finish(self):
        logger.info(self.nowEarnings())
        logger.info(self.transactionRecords)
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(111)
        # ax.set_title("Bitcoin Price Over Time")
        # plt.plot(self.outList, color='green', label='Predicted Price')
        # #plt.plot(real_y_test, color='red', label='Real Price')
        # ax.set_ylabel("Price (USD)")
        # ax.set_xlabel("Time (Days)")
        # ax.legend()
        # plt.show()


a = SupplementaryData(filename="btcusdt.csv")
a.load_data()
a.timestamp_to_int()
a.removeDuplicate()
a.insert_data()
# a.showImages()
#
#
# model = load_model("my_model3.h5")
# b = MarketSimulationBB(a.supplementaryData, USDTAmount=1000.0, model=model)
# b.run()
# exit(0)

lSTMmodel = LSTMmodel()
x_data, y_data, testX, testY, scaler = lSTMmodel.loadata(data=a.supplementaryData, look_back=50)
# # # y_data=df1["price"].as_matrix()[:, np.newaxis][0:N]
# # # min,max,step=getminmaxstep(y_data,N)
# # # x_data = df1.as_matrix()[0:N]
# exit(0)
model=lSTMmodel.initialize_model(x_data,50,0.2,'linear', 'mse', 'adam')
print (model.summary())
model, training_time = lSTMmodel.fit_model(model, x_data, y_data, 1024, 100, .05)
# #model.save('my_model3.h5')
# #predict_list=lSTMmodel.predict_sequence(model,testX,50,50,scaler)
# #print(predict_list)
# # show_image(predict_list)
# lSTMmodel.test_model(model, testX, testY, scaler)
print("Training time", training_time, "seconds")
model.save('my_model4.h5')
