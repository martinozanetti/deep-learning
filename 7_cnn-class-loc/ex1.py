#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import sklearn.datasets
import pandas as pd



#A simple CNN classifier
    
def main():
    # set random seed for reproducibility
    np.random.seed(0)

    # Download the CIFAR10 dataset using tensorflow.keras.datasets.cifar10.load_data()
    dataset = tf.keras.datasets.cifar10.load_data()
    train_images = dataset[0][0]
    train_labels = dataset[0][1]
    test_images = dataset[1][0]
    test_labels = dataset[1][1]
    print("\nTraining images shape: ", train_images.shape)
    print("Training labels shape: ", train_labels.shape)
    print("Test images shape:     ", test_images.shape)
    print("Test labels shape:     ", test_labels.shape,"\n")

    # This dataset contains 60k (50k training / 10k test) low resolution color images for 10 classes:
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Use a dictionary to map the class index to the class name.
    my_dict = {
        0:'airplane', 
        1:'automobile', 
        2:'bird', 
        3:'cat', 
        4:'deer', 
        5:'dog', 
        6:'frog', 
        7:'horse', 
        8:'ship', 
        9:'truck'}

    # Verify data by plotting image samples and labels (from the dictionary).
    # Plot 25 random images of the training set:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index = np.random.randint(0,50000)
        plt.imshow(dataset[0][0][index], cmap=plt.cm.binary)
        plt.xlabel(my_dict[dataset[0][1][index][0]])

    #plt.show()

    # Allocate a sequential model containing a normalization layer after determining the
    # best choice for this particular dataset.
    # After this initial normalization layer, threat images as a flatten layer
    # and build a classifier.
    model = keras.Sequential([
        #keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        #keras.layers.MaxPooling2D((2,2)),
        keras.layers.Normalization(), 
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")])

    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])

    # Train the classifier using the test data as validation set.
    # Store and plot the accuracy for the training and validation achieved with this approach.
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    plt.figure(figsize=(8,8))
    plt.title("MODEL1")
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("MODEL1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Redesign the previous model by replacing the initial flatten layer with
    # convolutional layers (tf.keras.layers.Conv2D) followed by a Max Pooling 2D
    # layer (tf.keras.layers.MaxPooling2D). Try this model with 3 consecutive layers (3 LAYERS COME?).
    # Compare the results with the previous model.
    model2 = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Normalization(), 
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")])

    model2.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])

    # Train the classifier using the test data as validation set.
    # Store and plot the accuracy for the training and validation achieved with this approach.
    history2 = model2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    plt.figure(figsize=(8,8))
    plt.title("MODEL2")
    plt.plot(history2.history["accuracy"], label="accuracy")
    plt.plot(history2.history["val_accuracy"], label="val_accuracy")
    plt.title("MODEL2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


if __name__=='__main__':
    main()