

#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
tf.random.set_seed(0)

def train(imgs, labels, parameters):
    model = keras.Sequential()
    # add input convolutional 5x5 layer for 28x28 images in black and white
    model.add(keras.layers.Conv2D(  parameters['units'],
                                    (5, 5),
                                    activation='relu',
                                    input_shape=(parameters['units'],
                                                 parameters['units'],
                                                 1)
                                                 )
                                 )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    #model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    # (now i have a 1D vector of 1024 elements)
    #model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(500, activation='relu'))
    # last layer with 10 neurons (one for each class) and softmax activation
    model.add(keras.layers.Dense(10, activation='softmax'))

    my_adam = keras.optimizers.Adam(learning_rate=parameters['learning_rate'])
    model.compile(  optimizer=my_adam,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                 )
    model.fit(imgs, labels, epochs=2)

    return model

def test(model, imgs, labels):
    loss_acc = model.evaluate(imgs, labels)
    return loss_acc

def main():

# Write a DL regression model using Keras with the following steps:

# Data loading

    # Load the mnist dataset from tensorflow.keras.datasets.mnist.
    (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

    # Study the dataset size (shape) and normalize the pixels.
    print("\nTraining images shape: ", img_train.shape)
    print("Training labels shape: ", label_train.shape)
    print("Test images shape:     ", img_test.shape)
    print("Test labels shape:     ", label_test.shape)

# DNN model

#     # Design a NN architecture for the classification of all digits with input shape (28, 28) and output shape (10).
#     model = keras.Sequential()

#     # add input convolutional 5x5 layer for 28x28 images in black and white
#     model.add(keras.layers.Conv2D(28, (5, 5), activation='relu', input_shape=(28, 28, 1)))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Flatten())
#     # (now i have a 1D vector of 1024 elements)
#     model.add(keras.layers.Dense(1024, activation='relu'))
#     model.add(keras.layers.Dense(500, activation='relu'))
#     # last layer with 10 neurons (one for each class) and softmax activation
#     model.add(keras.layers.Dense(10, activation='softmax'))

#     # Determine the number of trainable parameters.
#     print('\n')
#     model.summary()

# #     Fit the dataset with 5 epochs, using adam's optimizer, and the sparse_categorical_crossentropy loss function.
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(img_train, label_train, epochs=5)

#     # Evaluate the model on the test set.
#     #test_loss, test_acc = model.evaluate(img_test, label_test)
#     test_loss, test_acc = test(model, img_test, label_test)
#     print('Test accuracy:', test_acc)

    # # Plot the loss and accuracy curves for training and validation.
    # # Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['loss'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')

    # # plot some examples of classification
    # # get the predictions for the test images
    # predictions = model.predict(img_test)
    # # get the index of the highest probability for each prediction
    # predicted_classes = np.argmax(predictions, axis=1)
    # # plot the first 15 test images, their predicted labels, and the true labels
    # plt.figure(figsize=(10, 10))
    # for i in range(15):
    #     plt.subplot(3, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(img_test[i], cmap=plt.cm.binary)
    #     plt.xlabel("Label: {}\n Predicted: {}".format(predicted_classes[i], label_test[i]))

    # plt.show()


# Hyperparameter scan

#     Define a function which parametrizes the learning rate and the number of units of the DNN model using a python dict.
    test_setup = {
        'learning_rate': 0.001,
        'units': 28,
    }

    model = train(img_train, label_train, test_setup)
    print('\n\nAccuracy on test set is:', test(model, img_test, label_test)[1])

#     Use the Tree-structured Parzen Estimator with the hyperopt library.
    def hyper_func(parameters):

        model = train(img_train, label_train, parameters)
        test_acc = test(model, img_test, label_test)

        return {'loss': -test_acc[1], 'status': STATUS_OK}

    search_space = {
        'layer_size': hp.choice('layer_size', np.arange(10, 100, 20)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0)
    }

    trials = Trials()
    best = fmin(hyper_func, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(space_eval(search_space, best))

#     Plot the accuracy vs learning rate and number of layers for each trial.

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['layer_size'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('learning_rate')
    ax3.set_ylabel('Accuracy')
    plt.show()


if __name__=='__main__':
    main()
