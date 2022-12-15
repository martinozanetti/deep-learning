# Using TensorFlow primitives perform the following steps:
# allocate random normal variables for weight and bias representation
# of a multi-layer perceptron (MLP) with n_input size, two hidden layers
# with n_hidden_1 and n_hidden_2 neurons respectively and n_output size.

# Define a function which takes a tensor as input and returns the MLP
# prediction. Use the sigmoid function as activation function for all
# nodes in the network except for the output layer, which should be linear.

# Test the model prediction for 10 values in x linearly spaced from [-1,1]
# with n_input=1, n_hidden_1=5, n_hidden_2=2, n_output=1.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf

tf.random.set_seed(0)

# create a class MLP

class MLP:
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        self.n_imput = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output = n_output

        # define the weights and biases
        self.weights = [tf.Variable(tf.random.normal([n_input, n_hidden_1])),
                        tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
                        tf.Variable(tf.random.normal([n_hidden_2, n_output]))]

        self.biases = [ tf.Variable(tf.random.normal([n_hidden_1])),
                        tf.Variable(tf.random.normal([n_hidden_2])),
                        tf.Variable(tf.random.normal([n_output]))]

    # Define a method which takes a tensor as input and returns the MLP
    # prediction. Use the sigmoid function as activation function for all
    # nodes in the network except for the output layer, which should be linear
    def __call__(self, x):
        
        for i in range(len(self.weights)-1):
            x = tf.math.sigmoid(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))
            
        x = tf.matmul(x, self.weights[len(self.weights)-1])+self.biases[len(self.weights)-1]
        
        return x
    

def main():

    myMPL = MLP(1, 5, 2, 1)
    x = tf.linspace([-1.0], [1.0], 10)
    y = myMPL.__call__(x)

    print(y)

    # plot the result
    #plt.plot(x, y, 'o', label='MLP prediction')
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()


