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

    
def main():
    # Load the following dataset with 3670 photos of flowers with shape (180, 180, 3) each.
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    # Construct a training and validation dataset using tf.keras.utils.image_dataset_from_directory,
    # 20% split, 32 batch sizes. Inspect the training dataset by plotting image samples.

    # Build a CNN classifier by applying a rescaling layer (normalizing by 255), 3 convolutional
    # layers with [15, 32, 64] filters, 3x3 kernels, ReLU activations, interchanged with max pooling layers.
    # After flattening, apply a dense layer with 128 nodes and ReLU activation.

    # Perform a fit for few epochs (< 10), monitor and plot the evolution of the loss function
    # and accuracy for the train and validation set. In order to optimize dataset performance
    # cache and prefetch the original datasets with:

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build a sequential model with data augmentation layers, in particular with horizontal random
    # flip, random rotation (0.1) and random zoom (0.1). Plot samples of this layer.
    # Attach this layer at the beginning of the previous model and introduce a dropout layer
    # before flatten in order to reduce overfitting.


def nothing():
    pass

if __name__=='__main__':
    main()