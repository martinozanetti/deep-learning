#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import pandas as pd



# FORECASTING TIME SERIES
    
def main():
    
    # We provide numpy arrays for daily measurements performed in blocks of 10 days.
    # The data was already filtered and normalized.
    # Download the following datasets:
    # TRAINING SET
    # wget https://raw.githubusercontent.com/scarrazza/DL2022/main/Lecture_6/training_data.npy
    # wget https://raw.githubusercontent.com/scarrazza/DL2022/main/Lecture_6/training_label.npy
    # TEST SET
    # wget https://raw.githubusercontent.com/scarrazza/DL2022/main/Lecture_6/test_data.npy
    # wget https://raw.githubusercontent.com/scarrazza/DL2022/main/Lecture_6/test_label.npy
    # and check the corresponding sizes:
    # - training_data.npy: (1000, 10, 1) >> 1000 blocks of 10 days each
    # - training_label.npy: (1000, 1)
    # - test_data.npy: (100, 10, 1)
    # - test_label.npy: (100, 1)
    # Load the data using numpy.load.
    # The data is a time series of 1D measurements.
    # The label is the next day measurement.
    training_data  = np.load('training_data.npy')
    training_label = np.load('training_label.npy')
    test_data      = np.load('test_data.npy')
    test_label     = np.load('test_label.npy')

    # Build and train an LSTM model using Adam with MSE loss, 25 epochs, batch size 32.
    # Use the following architecture:
    # - 1 LSTM layer with 10 units
    # - 1 Dense layer with 1 unit
    
    # Define the model.
    model = keras.Sequential()
    model.add(keras.layers.LSTM(10, input_shape=(10,1)))
    model.add(keras.layers.Dense(1))
    model.summary()

    # Compile the model.
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(training_data, training_label, epochs=25, batch_size=32, validation_data=(test_data, test_label))
        
    # Print the final MSE for the test set.
    print('Final test MSE: ', history.history['val_loss'][-1])

    # Plot the following quantities:
    # - training and test data vs days,
    # - the LSTM predictions for the test data,
    # - the LSTM predictions for the first 100 days,
    # - the residual (y_test - prediction),
    # - and the scatter plot between the true test data vs predictions.
    
    # Plot the training and test data vs days.
    plt.figure()
    plt.title('Training and test data vs days')
    plt.plot(training_data[:,0,0], label='training data')
    plt.plot(np.arange(0,490), test_data[:,0,0], label='test data')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Data')
    
    # Plot the LSTM predictions for the test data.
    plt.figure()
    plt.title('LSTM predictions vs test data')
    plt.plot(np.arange(0,490), model.predict(test_data)[:,0], label='LSTM predictions')
    plt.plot(np.arange(0,490), test_label[:,0], label='test data')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Data')

    # Plot the LSTM predictions for the first 100 days.
    plt.figure()
    plt.title('LSTM predictions vs training data (fist 100 days))')
    plt.plot(np.arange(100), model.predict(training_data[:100])[:,0], label='LSTM predictions')
    plt.plot(np.arange(100), training_label[:100,0], label='training data')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Data')

    # Plot the residual (y_test - prediction).
    plt.figure()
    plt.title('Residual (y_test - prediction)')
    plt.plot(np.arange(0,490), test_label[:,0] - model.predict(test_data)[:,0], label='residual')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Data')

    # Plot the scatter plot between the true test data vs predictions.
    plt.figure()
    plt.scatter(test_label[:,0], model.predict(test_data)[:,0], label='scatter plot')
    plt.plot([0,1], [0,1], label='bisettrice', color='red')
    plt.legend()
    plt.xlabel('True test data')
    plt.ylabel('Predictions')

    plt.show()




  


if __name__=='__main__':
    main()
