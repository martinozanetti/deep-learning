
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


# Localization and classification

    
def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: \n      python ex2.py --load-model saved_model\n      python ex2.py --train-model true")
        exit(1)


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

    if len(args) == 2 and args[0] == '--load-model':
        model = tf.keras.models.load_model(args[1])
        history = pd.read_csv('training.log', sep=',', engine='python')

    elif len(args) == 2 and args[0] == '--train-model' and args[1] == 'true':
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
        x = tf.keras.layers.Dense(64, activation='relu')(c3) #64

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

        csv_logger = CSVLogger('training.log', separator=',', append=False)

        history = model.fit(train_images, (train_labels, train_boxes), epochs=5, batch_size=64,
                                        validation_data=(val_images, (val_labels, val_boxes)), callbacks=[csv_logger])
        # 5 ep, 64 batch
        
        model.save('saved-model')
        history = pd.read_csv('training.log', sep=',', engine='python')

    # Plot the classification and bounding box losses.
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(history['classifier_loss'], label='classifier_loss')
    plt.plot(history['val_classifier_loss'], label = 'val_classifier_loss')
    plt.ylabel('Loss')
    plt.title('Classification losses')
    plt.legend()
    plt.xticks(np.arange(0, 5, 1.0))

    plt.subplot(2,1,2)
    plt.plot(history['regressor_loss'], label='regressor_loss')
    plt.plot(history['val_regressor_loss'], label = 'val_regressor_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bounding box regressor losses')
    plt.legend()
    plt.xticks(np.arange(0, 5, 1.0))

    
    # Verify the results on the validation dataset by plotting samples and computing the IoU.
    # Plot 9 random images of the validation set with the predicted bounding box and the
    # ground truth bounding box.
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index = np.random.randint(0,10000)
        plt.imshow(val_images[index], cmap=plt.cm.binary)
        # plot the predicted bounding box
        pred_box = model.predict(val_images[index].reshape(1,75,75,1))[1][0]
        scale=75
        rect = patches.Rectangle((pred_box[0]*scale, pred_box[1]*scale), pred_box[2]*scale, pred_box[3]*scale, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        # plot the ground truth bounding box
        rect = patches.Rectangle((val_boxes[index][0]*scale, val_boxes[index][1]*scale), val_boxes[index][2]*scale, val_boxes[index][3]*scale, linewidth=1, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)
        plt.xlabel("ciph: "+my_dict[str(val_labels[index])] + ", IoU: " + str(compute_iou(pred_box, val_boxes[index])))


    # Compute the IoU for the predicted bounding boxes and the ground truth bounding boxes.
    # Plot the histogram of the IoU values.
    iou = []
    '''
    for i in range(10000):
        index = np.random.randint(0,10000)
        pred_box = model.predict(val_images[i].reshape(1,75,75,1))[1][0]
        iou.append(compute_iou(pred_box, val_boxes[i]))
        print(i)
    '''
    
    # save iou values in file
    '''
    with open('iou.txt', 'w') as f:
        i=0
        for item in iou:
            f.write(str(iou[i]))
            f.write("\n")
            i+=1
    '''
    # load iou values from file
    with open('iou.txt', 'r') as f:
        iou = f.readlines()
        iou = [float(i.strip()) for i in iou]

    # Evaluate the total number of good and bad bounding box predictions using an IoU threshold of 0.6
    good = 0
    bad = 0
    for i in range(10000):
        if iou[i] > 0.6:
            good += 1
        else:
            bad += 1

    plt.figure(figsize=(10,10))
    plt.hist(iou, bins=100, label=('IoU values\nGood: '+str(good)+'\nBad: '+str(bad)))
    plt.xlabel('IoU')
    plt.ylabel('Number of samples')
    plt.axvline(x=0.6, color='r', linestyle='--', label='IoU threshold')
    plt.title('IoU histogram')
    plt.legend()


    
    

    plt.show()

# Compute the IoU between two bounding boxes
def compute_iou(pred_box, val_box):
    # pred_box: predicted bounding box (x, y, w, h)
    # val_box: ground truth bounding box (x, y, w, h)
    # return: the IoU value
    x1 = max(pred_box[0], val_box[0])
    y1 = max(pred_box[1], val_box[1])
    x2 = min(pred_box[0]+pred_box[2], val_box[0]+val_box[2])
    y2 = min(pred_box[1]+pred_box[3], val_box[1]+val_box[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = pred_box[2]*pred_box[3] + val_box[2]*val_box[3] - intersection
    return intersection/union

if __name__=='__main__':
    main()