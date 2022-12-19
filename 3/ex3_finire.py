


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf

tf.random.set_seed(0)

#    Define a custom self using tf.Module inheritance which
#    returns the functional form w * x + b where w and b are 
#    tensor variables initialized with random values.
class custom_model(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.uniform([],-5,5))
        self.b = tf.Variable(tf.random.uniform([],-5,5))
        self.history = []
        self.loss_history = []
            
    def __call__(self, x):
        return self.w * x + self.b

    #    Define a loss function matching the mean squared error.
    def mse_loss(self, y_1, y_2):
        return tf.reduce_mean(tf.square(y_1 - y_2))

    #Training loop
    #    Define a train function which computes the loss function gradient and performs a full batch SGD (manually).
    def train(self, x_train, y_train, learning_rate):
        with tf.GradientTape() as t:
            current_loss = self.mse_loss(self(x_train), y_train)
            dw, db = t.gradient(current_loss, [self.w, self.b])

        self.w.assign_sub(learning_rate * dw)
        self.b.assign_sub(learning_rate * db)

    #    Define a training_loop function which performs 10 epochs, 
    #    prints the loss function at each iteration to screen and stores the self weights.
    def training_loop(self, x_train, y_train):
        print("Initial loss: {:.3f}".format(self.mse_loss(self(x_train), y_train)))

        for epoch in range(10):
            self.train(x_train, y_train, 2/float(epoch+1))
            loss = self.mse_loss(x_train, y_train)
            print("Epoch %2d: loss=%2.5f, w=%2.5f, b=%2.5f"
                %(epoch, loss, self.w, self.b))
            self.history.append([self.w, self.b])
            self.loss_history.append(loss)
            
    
def func(x, a, b):
    return a * x + b

# ==============================================

def main():

    #Data generation
    #    Generate predictions of f(x) = 3 * x + 2 for 200 linearly spaced x points between [-2, 2] in single precision.
    x = tf.linspace(-2.0, 2.0, 200)
    y = func(x, 3.0, 2.0)
    y_truth = y

    #    Include random normal noise (mu=0, sigma=1) to all predictions.
    y += tf.random.normal([200], 0.0, 1.0, dtype=tf.float32)
    y_data = y

    
    #    Plot data and truth

    plt.figure(figsize=(10, 10))
    plt.suptitle("Model birth")

    plt.subplot(4,1,1)   
    plt.title("Data vs Truth")
    plt.scatter(x, y_data, label="Data")
    plt.plot(x, y_truth, label="Truth")
    plt.legend()

    #Linear fit
    #    Plot data, ground truth self, predictions and loss function for the untrained self.
    plt.subplot(4,1,2)
    plt.title("Untrained model")

    my_model=custom_model()
    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth")
    plt.scatter(x, my_model(x), label="Model", s=10)
    
    print(my_model.mse_loss(my_model(x), y_truth))
    plt.legend()


    #    Train the model using the training_loop function.
    my_model.training_loop(x, y_data)

    plt.subplot(4,1,3)
    plt.title("Trained model")

    plt.scatter(x, y_data,label="Data")
    plt.plot(x, y_truth,label="Truth")
    plt.scatter(x, my_model(x), label="Model", s=10)
    
    plt.legend()

    plt.show()


    #Use Keras
    #    Replace the training loop with Keras self API, check results.
    my_model2 = custom_model()


if __name__ == "__main__":
    main()

