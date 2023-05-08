#!/usr/bin/python

import os
import sys
import time
import csv
import datetime
import requests
import math
import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import sim_logging
import constants
from alpaca_trade_api.rest import REST, TimeFrame

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, GaussianNoise, Activation
from tensorflow.keras import regularizers

class CRYPTO:
    def __init__(self, i_AI_model_name, i_values_list):
        self.model_name = i_AI_model_name
        self.rmse = i_values_list[0]
        self.sharpe_ratio = i_values_list[1]
        self.next_day_price = i_values_list[2]
        self.percentage_change = None

class ALPACA:
    def __init__(self, i_simlog):
        self.simlog = i_simlog
        self.api = tradeapi.REST('PKCSBUUOKJN32C5LN716', 'aR9GddDWEcfgRHXUPOUtP6X7YI46JNOJsDUaFUBl',
                                  base_url='https://paper-api.alpaca.markets')
        self.list_positions = self.api.list_positions()
        self.log_file = self.processLogFile()
        self.data = None
        self.master_list = []


    def process_action(self, i_crypto, i_action):
        # It is possible that there was an issue with pulling stock data from alpaca.
        # So retry with yahoo finance
        try:
            try:
                i_current_price = float(self.api.get_bars(i_crypto.replace('-','/'), TimeFrame.Hour, limit=1)[0].c)
            except Exception as e:
                self.simlog.warning(str(e))
                tickerData = yf.Ticker(i_crypto)
                i_current_price = tickerData.history(period='1d')['Close'][0]
        except IndexError as e:
            self.simlog.error("Failed to get current crypto price for " + str(i_crypto))
            self.simlog.error(str(e))
            return
        except Exception as e:
            self.simlog.error(str(e))
            raise Exception

        i_current_quantity = int(0)  # Number of crypto for this particular symbol
        i_current_invested = float(0)  # Calculated by multiplying quantity with market_value

        for i in range(len(self.list_positions)):
            if i_crypto == self.list_positions[i].symbol:
                i_current_quantity = float(self.list_positions[i].qty)
                i_current_invested = float(self.list_positions[i].qty) * float(self.list_positions[i].current_price)

        # No Action Needed at this time for the particular stock
        if i_action == constants.CRYPTO_LEAVE:
            return

        elif i_action == constants.CRYPTO_BUY:
            if i_current_invested < 100:
                i_qty = float(10 / float(i_current_price))
                self.simlog.info("We are going to BUY " + str(i_crypto))
                l_result = self.api.submit_order(symbol=i_crypto.replace('-','/'), qty=i_qty,
                                                 side='buy', type='market', time_in_force='gtc')
                x = 1

        # Sell all current quantity of this stock
        elif i_action == constants.CRYPTO_SELL:
            if i_current_quantity > 0:
                self.simlog.info("We are going to SELL " + str(i_crypto))
                l_result = self.api.submit_order(symbol=i_crypto.replace('-','/'), qty=i_current_quantity,
                                                 side='sell', type='market', time_in_force='gtc')
                x = 1
        else:
            print("The following action is undefined: " + str(i_action))
            raise Exception

    def processLogFile(self):
        log_master_filename = "MasterList_AI_model.csv"
        date = str(datetime.date.today())
        log_master_filename = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0] + \
                              "crypto_trading/logs//" + date + "/" + log_master_filename

        # Create a new file if the file doesn't already exists
        if not os.path.isfile(log_master_filename):
            # Create directories in the path if they don't exist
            os.makedirs(os.path.dirname(log_master_filename), exist_ok=True)
            # If the file doesn't exist, create it and append to it
            with open(log_master_filename, 'a') as file:
                file.write("cryptoName,modelName,RMSE,sharpeRatio,expectedValue\n")

        return log_master_filename


    # Retrieve all crypto currencies supported by Alpaca
    def get_all_crypto_currencies(self):
        i_master_list = []
        assets = self.api.list_assets(asset_class='crypto')

        for asset in assets:
            i_master_list.append(asset.symbol)
        return i_master_list

    def get_AI_data(self, i_crypto):
        self.master_list = []

        # Now check if the tickerName exists in that file or not
        if len(self.getTicker(i_crypto)) == 0:
            self.df = self.getHistoricalData(i_crypto)
            self.master_list.insert(-1, CRYPTO('LSTM', self.LSTM()))
            self.master_list.insert(-1, CRYPTO('RNN', self.RNN()))
            self.master_list.insert(-1, CRYPTO('ANN', self.ANN()))
            self.master_list.insert(-1, CRYPTO('CNN', self.CNN()))
            self.master_list.insert(-1, CRYPTO('Random Forrest', self.RandomForest()))

            with open(self.log_file, 'a') as file:
                for i in range(len(self.master_list)):
                    file.write(str(i_crypto) + ",")
                    file.write(self.master_list[i].model_name + ",")
                    file.write(str(self.master_list[i].rmse) + ",")
                    file.write(str(self.master_list[i].sharpe_ratio) + ",")
                    file.write(str(self.master_list[i].next_day_price) + "\n")

        # Now read the csv file to get stock model data
        i_master_list = self.getTicker(i_crypto)
        return i_master_list


    def getHistoricalData(self, i_crypto):
        i_base_directory = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
        i_log_directory = i_base_directory + "crypto_trading" + "/" + "logs" + "/"
        i_csv_file_name = i_log_directory + i_crypto + '.csv'
        try:
            data = yf.download(tickers=i_crypto, period='120mo', interval='1d')
            data.to_csv(i_csv_file_name)
        except Exception as e:
            print("Some error has occurred. Continue!!!!!")
            print(e)

        # Load the data back from the csv files
        df = pd.read_csv(i_csv_file_name)
        df.drop(df.tail(1).index, inplace=True)
        return df


    def getTicker(self, i_crypto):
        i_list = []
        # Open the CSV file
        with open(self.log_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['cryptoName'] == i_crypto:
                    i_list.append(row)
        return i_list

    def crypto_prediction(self, i_crypto):
        i_score_sell = 0
        i_score_buy = 0

        # Get data on the ticker
        try:
            tickerData = yf.Ticker(i_crypto);
            i_currentPrice = tickerData.history(period='1d')['Close'][0]
        except IndexError as e:
            self.simlog.error("Error found while trying to get current stock price for " + str(i_crypto))
            self.simlog.error(str(e))
            return constants.CRYPTO_LEAVE

        self.master_list = self.get_AI_data(i_crypto)
        self.simlog.info("AI model result for stock:  " + str(i_crypto))
        self.simlog.info("Current Price = $" + str(i_currentPrice))
        for i in range(len(self.master_list)):

            # Calculate the percentage change in price
            i_percentage_change = None
            if not self.master_list[i]['expectedValue'] == 'None':
                i_percentage_change = (
                            ((float(self.master_list[i]['expectedValue']) - i_currentPrice) / i_currentPrice) * 100)

            self.simlog.info("\nmodel_name = " + str(self.master_list[i]['modelName']))
            self.simlog.info("RMSE= " + str(self.master_list[i]['RMSE']))
            self.simlog.info("sharpeRatio= " + str(self.master_list[i]['sharpeRatio']))
            self.simlog.info("Next Day expectedValue = $" + str(self.master_list[i]['expectedValue']))
            self.simlog.info("Percentage Change = " + str(i_percentage_change))
            if float(self.master_list[i]['RMSE']) < 5:
                if i_percentage_change:
                    if i_percentage_change > 10:
                        i_score_buy += 1
                    elif i_percentage_change < 5:
                        i_score_sell += 1
                    else:
                        continue

        if i_score_sell >= 3:
            self.simlog.info("The current action is to SELL")
            return constants.CRYPTO_SELL
        elif i_score_buy >= 3:
            self.simlog.info("The current action is to BUY")
            return constants.CRYPTO_BUY
        else:
            self.simlog.info("The current action is to do nothing")
            return constants.CRYPTO_LEAVE

    def RNN(self):

        # THis is the number of days the prediction is based on
        seq_len = 30

        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # Preprocess the data
        data = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Split the data into training and testing sets
        training_data_len = math.ceil(len(scaled_data) * .9)
        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - seq_len:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the RNN model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(1))

        # Train the RNN model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)

        # Test the RNN model
        test_data = scaled_data[training_data_len - seq_len:, :]
        x_test = []
        y_test = data[training_data_len:, :]

        for i in range(seq_len, len(test_data)):
            x_test.append(test_data[i - seq_len:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions_scaled = model.predict(x_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)

        # Calculate the RMSE. Both formulae generates the same result
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

        # Calculate the Sharpe ratio
        mean_return = self.df['Close'].pct_change().mean()
        volatility = self.df['Close'].pct_change().std()
        sharpe_ratio_actual = (mean_return / volatility)

        df_predicted = pd.DataFrame(predictions)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        last_days = scaler.fit_transform(df.tail(seq_len)['Close'].values.reshape(-1, 1))
        next_day = model.predict(np.array([last_days]), verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio_predicted, next_day]

    # Artifical Neural Network
    def ANN(self):
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # THis is the number of days the prediction is based on
        seq_len = 30

        # Prepare the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df['Close'].values.reshape(-1, 1))

        # Split the data into training and testing sets
        training_data_len = int(len(scaled_data) * 0.9)
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len:, :]

        # Define the training data
        X_train = []
        y_train = []
        for i in range(seq_len, len(train_data)):
            X_train.append(train_data[i - seq_len:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Test the model
        X_test = []
        y_test = []
        for i in range(seq_len, len(test_data)):
            X_test.append(test_data[i - seq_len:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price_scaled = model.predict(X_test, verbose=0)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price_scaled)

        # Calculate the root mean squared error
        # rmse = math.sqrt(mean_squared_error(y_test, predicted_stock_price_scaled)) #Shouldn't use scaled prices
        rmse = math.sqrt(mean_squared_error(df['Close'][training_data_len + seq_len:], predicted_stock_price))

        # Calculate the Sharpe ratio based on the Actual prediction
        mean_return = df['Close'][training_data_len + seq_len:].pct_change().mean()
        volatility = df['Close'][training_data_len + seq_len:].pct_change().std()
        sharpe_ratio_actual = (mean_return / volatility)

        df_predicted = pd.DataFrame(predicted_stock_price)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        last_days = scaler.fit_transform(df.tail(seq_len)['Close'].values.reshape(-1, 1))
        next_day = model.predict(np.array([last_days]))
        next_day = scaler.inverse_transform(next_day)[0][0]

        # Visualize the results
        # plt.plot(df['Close'][training_data_len:])
        # plt.plot(df['Close'][training_data_len + seq_len:].index, predicted_stock_price)
        # plt.legend(['Actual', 'Predicted'])
        # plt.show()
        return [rmse, sharpe_ratio_predicted, next_day]

    def RandomForest(self):
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)

        # Split into training and testing sets
        X = df.drop('Close', axis=1);
        X = X.drop('Date', axis=1)
        y = df['Close']
        test_size = math.ceil(len(df) * .9)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)
        rf.fit(X_train, y_train)

        # Make predictions on testing set
        y_pred = rf.predict(X_test)

        # Calculate root mean squared error
        rmse = mean_squared_error(y_test, y_pred)

        # Calculate the Sharpe ratio based on the Actual prediction
        mean_return = df['Close'][test_size:].pct_change().mean()
        volatility = df['Close'][test_size:].pct_change().std()
        sharpe_ratio_actual = (mean_return / volatility)

        df_predicted = pd.DataFrame(y_pred)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the stock price for next day
        next_day = None

        return [rmse, sharpe_ratio_predicted, next_day]

    def LSTM(self):

        # `look_back` is the number of previous time steps to use as input to the LSTM network
        # (e.g. 1 for using only the previous day's price)
        look_back = 30

        # Create a copy of df to prevent overwrite
        dataset = self.df.copy(deep=True)

        # Normalize the data
        dataset = dataset.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # Split into training and testing sets
        train_size = int(len(dataset) * 0.9)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        trainX, trainY = create_dataset_LSTM(train, look_back)
        testX, testY = create_dataset_LSTM(test, look_back)

        # Reshape data for LSTM input (samples, time steps, features)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Create LSTM model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)

        # Make predictions on testing set
        testPredict = model.predict(testX, verbose=0)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Calculate root mean squared error
        rmse = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        df_predicted = pd.DataFrame(testPredict)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Predict the next days price
        # Create a copy of df to prevent overwrite
        df = self.df.copy(deep=True)
        last_days = df.tail(look_back)
        last_days = scaler.fit_transform(last_days.filter(['Close']).values)
        last_days = np.reshape(last_days, (1, 1, last_days.shape[0]))
        next_day = model.predict(last_days, verbose=0)
        next_day = scaler.inverse_transform(next_day)[0][0]

        return [rmse, sharpe_ratio_predicted, next_day]

    # Convolutional Neural Networks (CNNs): CNNs are a type of neural network that are
    # commonly used for image recognition tasks. They have also been applied to stock
    # price prediction by treating historical stock prices as a type of image.
    # The CNN can then learn patterns and trends in the stock prices over time to make predictions.
    def CNN(self):

        seq_len = 30

        # Create a copy of df to prevent overwrite
        data = self.df.copy(deep=True)
        data = data.drop(['Date'], axis=1);
        data = data.drop(['Volume'], axis=1)

        # Use last seq_len days of data to predict next day's closing price
        X = []
        y = []
        for i in range(seq_len, len(data)):
            X.append(data.iloc[i - seq_len:i, 1:].values)
            y.append(data.iloc[i, -1])
        X = np.array(X)
        y = np.array(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        # Define CNN model
        model = Sequential()
        model.add(
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        # Compile model
        model.compile(loss='mse', optimizer='adam')

        # Train model
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        predictions = model.predict(X_test, verbose=0)

        # Calculate the RMSE. Both formulae generates the same result
        rmse = math.sqrt(mean_squared_error(y_test, predictions))

        # Calculate the Sharpe ratio
        df_predicted = pd.DataFrame(predictions)
        mean_return = df_predicted.pct_change().mean()[0]
        volatility = df_predicted.pct_change().std()[0]
        sharpe_ratio_predicted = (mean_return / volatility)

        # Use last 30 days of data to make prediction
        last_days = data.iloc[-seq_len:, 1:].values
        last_days = np.reshape(last_days, (1, seq_len, last_days.shape[1]))
        next_day = model.predict(last_days, verbose=0)[0][0]

        return rmse, sharpe_ratio_predicted, next_day

# Reshape data for LSTM input
def create_dataset_LSTM(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
