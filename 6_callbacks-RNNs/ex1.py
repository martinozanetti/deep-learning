#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import hyperopt
from hyperopt import hp, pyll
import time
from hyperopt import fmin, tpe, rand, STATUS_OK, Trials
import sklearn.datasets
import pandas as pd



# EARLY STOPPING
    
def main():
    # Download the IRIS dataset using the method sklearn.dataset.load_iris
    dataset = sklearn.datasets.load_iris()

    # The load_iris method returns a dictionary with the following keys:
    # - "data" a matrix (150,4) with the sepal and petal lengths and widths, 
    # - "target" an array (150) with the corresponding flower label id,
    # - "feature_names" the corresponding list of strings for the data names,
    # - "target_names" the list of strings for the flower names.
    print("\nData shape:      ", dataset.data.shape)
    print("Target shape:    ", dataset.target.shape)
    print("Feature names:   ", dataset.feature_names)
    print("Target names:    ", dataset.target_names, '\n')

    # Load data in a pandas DataFrame with the feature names and target names.
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['label'] = dataset.target_names[dataset.target]
    
    # This should include the data values and the flower label.
    # Verify that the DataFrame is correct (print the DataFrame to screen)
    print(df)

    # (Optional) Inspect the data with e.g. correlation/scatter plots.

    # Given that the label column is a categorical feature, we should convert it to a one-hot encoding. 
    # Perform this one-hot encoding operation.
    # Replace the label column with three new columns: label_setosa, label_versicolor, label_virginica
    label = pd.get_dummies(df['label'], prefix='label')
    df = pd.concat([df, label], axis=1)
    # drop old label
    df.drop(['label'], axis=1, inplace=True)

    print(df)
    

    # Extract 80% of the data for training and keep 20% for test, using the DataFrame.sample method.
    # Use the random_state=0 parameter to ensure replicability.
    # Split the data into X_train, y_train, X_test, y_test.
    # X_train and X_test should be a numpy array with shape (120,4) and (30,4) respectively.
    # y_train and y_test should be a numpy array with shape (120,3) and (30,3) respectively.
    random_state = 0

    train = df.sample(frac=0.8, random_state=0)
    test = df.drop(train.index)
    X_train = train.drop(['label_setosa', 'label_versicolor', 'label_virginica'], axis=1).values
    y_train = train[['label_setosa', 'label_versicolor', 'label_virginica']].values

    # Define a sequential model with 3 output nodes with softmax activation function.
    model = keras.Sequential()
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    # Perform a fit using Adam and the categorical cross-entropy loss function for 200 epochs, validation split 
    # of 40% and batch size of 32.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, 4))
    model.save_weights('my_rough_model.h5')
    history = model.fit(X_train, y_train, epochs=200, validation_split=0.4, batch_size=32)

    # Plot the learning curves (loss vs epochs) for the training and validation datasets.
    plt.figure("Learning curves - no early stopping")
    plt.title("Learning curves - no early stopping")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    #plt.show()

    # Modify the previous point in order to use early stopping on the validation loss with patience=10.
    my_callbacks = [keras.callbacks.EarlyStopping(patience=10),]
    model.load_weights('my_rough_model.h5')
    history2 = model.fit(X_train, y_train, epochs=200, validation_split=0.4, batch_size=32, callbacks=my_callbacks)

    # Plot the learning curves and check the stopping epoch.
    plt.figure("Learning curves - early stopping")
    plt.title("Learning curves - early stopping")
    plt.plot(history2.history['loss'], label='loss')
    plt.plot(history2.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # (Optional) Include the TensorBoard callback. Integrate the hyperopt pipeline implemented 
    # in the previous lecture using a loss the accuracy on the test set obtained in point 4.    


if __name__=='__main__':
    main()
