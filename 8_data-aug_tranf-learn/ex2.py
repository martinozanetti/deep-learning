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


# TRANSFER LEARNING

img_height = 160
img_width = 160  

def main():

    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage:")
        print("      python ex1.py --model load")
        print("      python ex1.py --model train-aug")
        exit(1)

    print()
    # Load the following dataset containing thousands of images of cats and dogs:
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = pathlib.Path(path) / 'train'
    validation_dir = pathlib.Path(path) / 'validation'

    # Construct a training and validation dataset using tf.keras.utils.image_dataset_from_directory,
    # 20% split, 32 batch sizes.
    training_data = tf.keras.utils.image_dataset_from_directory(
        train_dir, 
        validation_split=0.2, 
        subset="training", 
        seed=123, 
        image_size=(img_height, img_width), 
        batch_size=32)
    valid_data = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
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
    # ...
    
    # get the model
    if args[0] == "--model":
        # build the model with augmentation
        if args[1] == "train-aug":
            # Allocate a MobileNetV2 base model passing the input image shape, excluding the
            # classification layers at the top of the model (include_top=False) and using weights
            # from ImageNet (weights='imagenet'). Freeze this model by calling base_model.trainable = False.
            base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                include_top=False,
                weights='imagenet')
            base_model.trainable = False

            # verify the augmentation function
            # ...


            # Construct the final model using the functional API. The input passes through: a data
            # augmentation layer, a preprocessing input (which normalizes images for MobileNetV2)
            # using tf.keras.applications.mobilenet_v2.preprocess_input, the freeze base model MobileNetV2,
            # an average over the spatial locations using tf.keras.layers.GlobalAveragePooling2D(),
            # a dropout layer (0.2), and finally a dense layer with a single unit.
            # The network output should be considered as logit, i.e. positive numbers predict
            # class 1 while negative class 0.
            inputs = tf.keras.Input(shape=(img_height, img_width, 3))
            x = augmentation_layer(inputs)
            c1 = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            c2 = base_model(c1, training=False)
            c3 = tf.keras.layers.GlobalAveragePooling2D()(c2)
            c4 = tf.keras.layers.Dropout(0.2)(c3)
            outputs = tf.keras.layers.Dense(1)(c4)

            model = tf.keras.Model(inputs, outputs)


            # Train the model with Adam and learning rate 1e-4, binary cross-entropy loss function
            # and few epochs (< 10).
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

            training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            csv_logger = CSVLogger('training.log', separator=',', append=False)

            model.fit(training_data, epochs=6, batch_size = 32, validation_data=valid_data, callbacks=[csv_logger])
            model.save('saved-model')

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

#===================================================================================================

def augmentation_layer(inputs):
    x1 = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
    x2 = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(x1)
    return x2
    
#===================================================================================================

def data_augmentation(training_data):
    # Build a sequential model with data augmentation layers, with horizontal random
    # flip and random rotation (0.2).
    # ...

    # Plot samples of augmented data.
    plt.figure()
    for images, _ in training_data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    plt.show()
    
    return data_augmentation

#===================================================================================================

if __name__=='__main__':
    main()