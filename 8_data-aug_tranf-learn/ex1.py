#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import sklearn.datasets
import pandas as pd
import sys
from keras.callbacks import CSVLogger
from matplotlib import patches

import pathlib


# CLASSIFICATION WITH DATA AUGMENTATION

img_height = 180
img_width = 180
    
def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage:")
        print("      python ex1.py --model load")
        print("      python ex1.py --model train-no-aug")
        print("      python ex1.py --model train-aug")
        exit(1)

    print()

    # Load the following dataset with 3670 photos of flowers with shape (180, 180, 3) each.
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    # Construct a training and validation dataset using tf.keras.utils.image_dataset_from_directory,
    # 20% split, 32 batch sizes. Inspect the training dataset by plotting image samples.
    training_data = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.2, 
        subset="training", 
        seed=123, 
        image_size=(img_height, img_width), 
        batch_size=32)
    valid_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32)

    class_names = training_data.class_names

    # print some samples
    plt.figure(figsize=(10, 10))
    for images, labels in training_data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    # perform one-hot ecoding of dataset 
    training_data = training_data.map(lambda x, y: (x, tf.one_hot(y, 5)))
    valid_data = valid_data.map(lambda x, y: (x, tf.one_hot(y, 5)))

    print()

    # get the model
    if args[0] == "--model":
        if args[1] == "train-no-aug":
            model = build_and_train(training_data=training_data, valid_data=valid_data)
        
        # build the model with augmentation
        elif args[1] == "train-aug":
            model = build_and_train_aug(training_data=training_data, valid_data=valid_data)

        # load the model
        elif args[1] == "load":
            model = tf.keras.models.load_model('saved-model')

        else:
            print("Invalid argument")
            exit(1)
    else:
        print("Invalid argument")
        exit(1)

    history = pd.read_csv('training.log', sep=',', engine='python')

    # Plot the evolution of the loss function and accuracy for the train and validation set.
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss evolution')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy evolution')

    plt.show()
#==================================================================================================

def build_and_train(training_data, valid_data):
    # Build a CNN classifier by applying a rescaling layer (normalizing by 255), 3 convolutional
    # layers with [15, 32, 64] filters, 3x3 kernels, ReLU activations, interchanged with max pooling layers.
    # After flattening, apply a dense layer with 128 nodes and ReLU activation.
    # Finally, apply a dense layer with 5 nodes and softmax activation.
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Compile the model with categorical crossentropy loss, Adam optimizer and accuracy metric.
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Perform a fit for few epochs (< 10). In order to optimize dataset performance
    # cache and prefetch the original datasets with:
    training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    csv_logger = CSVLogger('training.log', separator=',', append=False)

    print(training_data)
    print(valid_data)

    
    model.fit(training_data, epochs=6, batch_size = 32, validation_data=valid_data, callbacks=[csv_logger])
    model.save('saved-model')

    return model

#==================================================================================================

def build_and_train_aug(training_data, valid_data):
    # Build a sequential model with data augmentation layers, in particular with horizontal random
    # flip, random rotation (0.1) and random zoom (0.1).
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                input_shape=(img_height, img_width, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    # Plot samples of augmented data.
    plt.figure()
    for images, _ in training_data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    # Attach this layer at the beginning of the previous model and introduce a dropout layer
    # before flatten in order to reduce overfitting.
    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Compile the model with categorical crossentropy loss, Adam optimizer and accuracy metric.
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Perform a fit for few epochs (< 10). In order to optimize dataset performance
    # cache and prefetch the original datasets with:
    training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    csv_logger = CSVLogger('training.log', separator=',', append=False)

    print(training_data)
    print(valid_data)

    model.fit(training_data, epochs=6, batch_size = 32, validation_data=valid_data, callbacks=[csv_logger])
    model.save('saved-model')

    return model

#==================================================================================================

if __name__=='__main__':
    main()