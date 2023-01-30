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

    
def main():

    # Load the following dataset containing thousands of images of cats and dogs:
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = pathlib.Path(path) / 'train'
    validation_dir = pathlib.Path(path) / 'validation'

    # Construct a training and validation dataset using tf.keras.utils.image_dataset_from_directory,
    # 20% split, 32 batch sizes. Inspect the training dataset by plotting image samples.
    # Prefetch data following the same approach implemented in the previous exercise.

    # Construct a data augmentation model with horizontal random flip and rotation (0.2).
    # Plot samples of augmented data.

    # Allocate a MobileNetV2 base model passing the input image shape, excluding the
    # classification layers at the top of the model (include_top=False) and using weights
    # from ImageNet (weights='imagenet'). Freeze this model by calling base_model.trainable = False.

    # Construct the final model using the functional API. The input passes through: a data
    # augmentation layer, a preprocessing input (which normalizes images for MobileNetV2)
    # using tf.keras.applications.mobilenet_v2.preprocess_input, the freeze base model MobileNetV2,
    # an average over the spatial locations using tf.keras.layers.GlobalAveragePooling2D(),
    # a dropout layer (0.2), and finally a dense layer with a single unit.
    # The network output should be considered as logit, i.e. positive numbers predict
    # class 1 while negative class 0.

    # Train the model with Adam and learning rate 1e-4, binary cross-entropy loss function
    # and few epochs (< 10). Monitor and plot the loss function and accuracy for each epoch,
    # for training and validation sets.


def nothing():
    pass

if __name__=='__main__':
    main()