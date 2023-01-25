
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


    # Construct a custom Keras model (using the functional API, https://keras.io/guides/functional_api/)
    # (vedi slide lab, lez8) with the following components:
    # a feature extractor using a CNN followed by a flatten and a dense layer and two
    # final end-points: a classifier (10 classes) and a bounding box regressor (4 coordinates).
    # Use the categorical cross-entropy loss function for the classifier and the MSE
    # for the bounding box regressor.

    inputs = tf.keras.layers.Input(shape=(75, 75, 1))
    c1 = tf.keras.layers.Conv2D(75, (3,3), activation='relu')(inputs)
    c2 = tf.keras.layers.MaxPooling2D((3, 3))(c1)
    c3 = tf.keras.layers.Flatten()(c2)
    x = tf.keras.layers.Dense(64, activation='relu')(c3)

    # output for classifier
    o1 = tf.keras.layers.Dense(10, activation='softmax', name='classifier')(x)
    # output for bbox regressor
    o2 = tf.keras.layers.Dense(4, name='regressor')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[o1, o2])

    # Compile the model with the appropriate loss functions and metrics.
    model.compile(
        optimizer='adam',
        loss = {'classifier': 'categorical_crossentropy',
              'regressor': 'mse'},
        metrics = {'classifier': 'acc',
                   'regressor': 'mse'}
    )

    model.summary()

    history = model.fit(train_images, (train_labels, train_boxes), epochs=5, batch_size=64,
                                       validation_data=(val_images, (val_labels, val_boxes)))
    
    model.save('saved-model')

    # Plot the classification and bounding box losses.
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(history.history['classifier_loss'], label='classifier_loss')
    plt.plot(history.history['val_classifier_loss'], label = 'val_classifier_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(history.history['regressor_loss'], label='regressor_loss')
    plt.plot(history.history['val_regressor_loss'], label = 'val_regressor_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    

    # Verify the results on the validation dataset by plotting samples and computing the IoU.
    # Evaluate the total number of good and bad bounding box predictions using an IoU threshold of 0.6

    plt.show()


if __name__=='__main__':
    main()