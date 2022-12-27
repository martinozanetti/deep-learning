import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf

# Define a true function true_function(x) = cos(1.5 * pi * x).
def true_function(x):
    return np.cos(1.5 * np.pi * x)


def main():
    N=30
    # Generate N random points in x between [0, 1].
    np.random.seed(0) # <-- to always get the same random numbers
    x = np.sort(np.random.rand(N))

    # Evaluate the target points as: true_function(x) + np.random.rand() * 0.1
    y = true_function(x) + np.random.rand(N) * 0.1
    x_test = np.linspace(0, 1, 100)

    # Implement and perform polynomial fits for degrees 1, 4 and 15. Use as loss function the MSE function.
    plt.figure(figsize=(5, 5))
    degrees = [1, 4, 15]


    # *** Scipy minimize

    # The loss function 
    def loss(p, func):
        ypred = func(list(p), x)
        return tf.reduce_mean(tf.square(ypred - y)).numpy()

    for degree in degrees:
        res = minimize(
            loss, # <-- the function to minimize
            np.zeros(degree+1), # <-- initial guess
            args=(tf.math.polyval), # <-- arguments to pass to the loss function
            method='BFGS') # <-- the optimization method
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_function(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Scipy.minimize")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
