# Script for Model Training for Water Consumption Forecasting
"""
Created on Tue Jun 08 00:55:57 2021
@author = Karol


"""

# Import Dependencies
import os
import yaml
from numpy import array
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import LSTM, Dropout,Dense,RepeatVector,TimeDistributed,Input,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam as adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from sklearn import preprocessing
import joblib
import datetime


# Define the Path to the Config File.
path_to_config_file = "config.yaml"

def config_params():

    with open(path_to_config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    data_path = cfg["water_consumption"]["preprocessed_data"]
    n_past_cfg = cfg["Settings"]["n_past"]
    n_future_cfg = cfg["Settings"]["n_future"]
    n_features_cfg = cfg["features"]["n_features"]

    return data_path, n_future_cfg, n_past_cfg, n_features_cfg


# Helper Functions

'''
method: split_series
input_parameters: series, n_past, n_future
output_parameters: np.array(X), np.array(y)
description: This method takes the data and splits it for supervised learning where input is the
             last n (n_past) observations and output is the future m (n_future) observations. 
'''

def split_series(series, n_past:int, n_future:int):
    #
    # n_past ==> no of past observations
    #
    # n_future ==> no of future observations
    #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


'''
method: training
input_parameters: model, X_train, y_train,X_test, y_test
output_parameters: none
description: This method trains the neural network based on the model passed and also
             plots the training results.
'''

def training(model, X_train, y_train, X_val, y_val):

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # compile the model
    model.compile(optimizer=opt, loss='mse', metrics = ['mae']) 
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=16, verbose=1)

    # plot training performance over the epochs
    plot_training(history)


'''
method: plot_training
input_parameters: history
output_parameters: none
description: This method plots the training and validation performance over the epochs.
'''

def plot_training(history):
    print(history.history.keys())

    # "MAE"
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# Create the CNN LSTM model
def create_model_prediction(y_test, window, n_features, n_future):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(y_test.shape[1]))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(TimeDistributed(Dense(n_features)))
    model.summary()

    return model


""" 
method name : main
method description : Driver Function for the Program.
input args : 
    None
return :
    None
"""

def main():
    
    data_path, n_future_cfg, n_past_cfg, n_features_cfg = config_params()

    # Load the Data
    data = pd.read_csv(data_path)
    data = data.rename({"Unnamed: 0":"time"}, axis = 1)
    data = data.set_index('time')


    # Look into Last 48 hours and predict next 1 hour 
    n_past = window = n_past_cfg
    n_future = n_future_cfg
    n_features = n_features_cfg

    # Scaling the Data
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Train-Test Split
    train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
    train_df,val_df = train_test_split(train_df, test_size=0.35, shuffle=False)

    # Split data into past and future observations and reshape
    train = train_df
    test = test_df

    X_train, y_train = split_series(train, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

    X_test, y_test = split_series(test, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

    X_val, y_val = split_series(val_df, n_past, n_future)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
    y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], n_features))

    print("Input Shape: ", X_train.shape, X_test.shape, X_val.shape)
    print("Output Shape: ", y_train.shape, y_test.shape, y_val.shape)   

    # Create and Train the Model
    model = create_model_prediction(y_test, window, n_features, n_future)
    print("Training Started !!")
    training(model, X_train, y_train, X_val, y_val)

    # Test the Model
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=16)
    print("Losses are:", results)

    print("Training Sucessfull !!")


if __name__ == "__main__":
    main()