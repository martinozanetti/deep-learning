
#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import sklearn.datasets
import pandas as pd



# Localization and classification

    
def main():
    # set random seed for reproducibility
    np.random.seed(0)

    # We provide images of 75x75 pixels containing MNIST digits in the data.tgz folder above.
    # Each image contains only one digit of 28x28 pixels placed in a random position of the image.
    # The files training_images.npy and validation_images.npy contains the images (60'000 and 10'000),
    # training_labels.npy and validation_labels.npy the labels of each image,
    # training_boxes.npy and validation_boxes.npy the 4 coordinates of the bounding
    # boxes (xmin, ymin, xmax, ymax).
    # Load data and plot samples.
    train_images = np.load('data/training_images.npy')
    train_labels = np.load('data/training_labels.npy')
    train_boxes = np.load('data/training_boxes.npy')
    val_images = np.load('data/validation_images.npy')
    val_labels = np.load('data/validation_labels.npy')
    val_boxes = np.load('data/validation_boxes.npy')

    print("\nTraining images shape: ", train_images.shape)
    print("Training labels shape: ", train_labels.shape)
    print("Training boxes shape:  ", train_boxes.shape)
    print("Validation images shape: ", val_images.shape)
    print("Validation labels shape: ", val_labels.shape)
    print("Validation boxes shape:  ", val_boxes.shape,"\n")

    # Given that the label data are a one-hot encoding, we should create a dictionary
    # to access it as a categorical feature. 
    my_dict = {
        "[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":'0',
        "[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]":'1',
        "[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]":'2',
        "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]":'3',
        "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]":'4',
        "[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]":'5',
        "[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]":'6',
        "[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]":'7',
        "[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]":'8',
        "[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]":'9'
    }

    # plot 25 random images of the training set:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index = np.random.randint(0,60000)
        plt.imshow(train_images[index], cmap=plt.cm.binary)
        plt.xlabel(my_dict[str(train_labels[index])])

    plt.show()

    # Construct a custom Keras model (using the functional API, https://keras.io/guides/functional_api/)
    # (vedi slide lab, lez8) with the following components:
    # a feature extractor using a CNN followed by a flatten and a dense layer and two
    # final end-points: a classifier (10 classes) and a bounding box regressor (4 coordinates).
    # Use the categorical cross-entropy loss function for the classifier and the MSE
    # for the bounding box regressor.

    # Plot the classification and bounding box losses. Verify the results on the
    # validation dataset by plotting samples and computing the IoU. Evaluate the total
    # number of good and bad bounding box predictions using an IoU threshold of 0.6

    plt.show()


if __name__=='__main__':
    main()