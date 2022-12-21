# Write a ML classification model using keras with the following steps:

#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
tf.random.set_seed(0)


def main():

# Data loading

#     Load the fashion mnist dataset from tensorflow.keras.datasets.fashion_mnist.
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

#     Study the dataset size (pixel shape) and plot some sample images.
    print("\nTraining images shape: ", img_train.shape)
    print("Training labels shape: ", label_train.shape)
    print("Test images shape:     ", img_test.shape)
    print("Test labels shape:     ", label_test.shape)

#     This dataset contains the following classes:
#     ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'].
#     Create a dictionary to map the class number to the class name.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure()

    n1, n2 = 0, 2        

    plt.subplot(1, 2, 1)
    plt.imshow(img_train[n1], cmap='gray')
    plt.title('Label: {}'.format(class_names[label_train[n1]]))

    plt.subplot(1, 2, 2)
    plt.imshow(img_train[n2], cmap='gray')
    plt.title('Label: {}'.format(class_names[label_train[n2]]))


#     Normalize images for training and test, considering the maximum pixel value of 255.
    img_train = img_train / 255.0
    img_test = img_test / 255.0


# NN model fit

#     Build a NN model which flattens images and applies 2 dense layers with 128 and 10 units respectively.
#     The first layer uses relu while the last layer softmax. Determine the number of trainable parameters.
    img_train = img_train.reshape(img_train.shape[0], 28*28)
    img_test = img_test.reshape(img_test.shape[0], 28*28)
    
    model = keras.Sequential([ keras.layers.Dense(128, activation="relu", input_dim=784),])
    model.add(keras.layers.Dense(10, activation="softmax"))

    print('\n')
    model.summary()

#     Fit the dataset with 5 epochs, using adam's optimizer, and the sparse_categorical_crossentropy loss function.
#     The Sequential.compile method supports extra arguments, such as metrics=['accuracy'] in order to monitor
#     extra statistical estimators during epochs.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(img_train, label_train, epochs = 5)

#     Evaluate test accuracy.
    test_loss, test_acc = model.evaluate(img_test, label_test)
    print('Test accuracy:', test_acc)

#    Plot the loss and accuracy curves.
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('Model loss and accuracy')
    plt.ylabel('Loss and accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')

#     Identify examples of bad classification.
    bad_class_index = []
    label_guess = model.predict(img_test)

    label_guess = np.argmax(label_guess, axis=1)
    for i in range(len(label_guess)):
        if label_guess[i] != label_test[i]:
            bad_class_index.append(i)


    plt.figure()
    n1, n2, n3, n4, n5, n6, n7, n8 = 0, 1, 2, 3, 4, 5, 6, 7 

    plt.subplot(2, 4, 1)
    plt.imshow(img_test[bad_class_index[n1]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n1]]], class_names[label_guess[bad_class_index[n1]]]))

    plt.subplot(2, 4, 2)
    plt.imshow(img_test[bad_class_index[n2]].reshape(28, 28), cmap='gray') 
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n2]]], class_names[label_guess[bad_class_index[n2]]]))

    plt.subplot(2, 4, 3)
    plt.imshow(img_test[bad_class_index[n3]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n3]]], class_names[label_guess[bad_class_index[n3]]]))

    plt.subplot(2, 4, 4)
    plt.imshow(img_test[bad_class_index[n4]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n4]]], class_names[label_guess[bad_class_index[n4]]]))

    plt.subplot(2, 4, 5)
    plt.imshow(img_test[bad_class_index[n5]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n5]]], class_names[label_guess[bad_class_index[n5]]]))

    plt.subplot(2, 4, 6)
    plt.imshow(img_test[bad_class_index[n6]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n6]]], class_names[label_guess[bad_class_index[n6]]]))

    plt.subplot(2, 4, 7)
    plt.imshow(img_test[bad_class_index[n7]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n7]]], class_names[label_guess[bad_class_index[n7]]]))

    plt.subplot(2, 4, 8)
    plt.imshow(img_test[bad_class_index[n8]].reshape(28, 28), cmap='gray')
    plt.title('Label: {}\n Prediction: {}'.format(class_names[label_test[bad_class_index[n8]]], class_names[label_guess[bad_class_index[n8]]]))


    plt.show()


if __name__=='__main__':
    main()
