#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import hyperopt
from hyperopt import hp, pyll
import time
from hyperopt import fmin, tpe, rand, STATUS_OK, Trials


# set the seed for replicability (vengono comunque diversi...)
os.environ['HYPEROPT_FMIN_SEED'] = "1"
tf.random.set_seed(0)

# Define an objective function which returns the analytic expression of a 
# 1d polynomial with coefficients 
# f(x) = 0.05 (x^6 - 2 x^5 - 28 x^4 + 28 x^3 + 12 x^2 -26 x + 100).

def func(x):
    return 0.05*(x**6 - 2*x**5 - 28*x**4 + 28*x**3 + 12*x**2 -26*x + 100)

def objective(x):
    return {'loss': func(x), 'status': STATUS_OK, 'eval_time': time.time()}


def main():

    # Plot the previous function using a linear grid of points in x between [-5, 6].
    x = np.linspace(-5, 6, 100)
    y = func(x)

    plt.figure(figsize=(11, 12))

    plt.subplot(3, 2, 1)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('My function')
    plt.grid()

    # Define an uniform search domain space using hyperopt.
    a = hp.uniform('a', -5, 6)
    
    # With hyperopt.pyll.stochastic.sample build an histogram with samples from that space.
    sample = []
    for i in range(1000):
        sample.append(pyll.stochastic.sample(a))

    plt.subplot(3, 2, 2)
    plt.hist(sample, bins=20)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Uniform search space sampling')

    # Perform the objective function minimization using the Tree-structured
    # Parzen Estimator model, 2000 evaluations and store the trials using hyperopt.Trials.
    trials = Trials()
    best = fmin(objective, space=a, algo=tpe.suggest, max_evals=2000, trials=trials)

    # Print to screen the best value of x.
    print('Best value of x: ', best)

    # Show scatter plot with the x-value vs iteration together with the final best value of x.
    plt.subplot(3, 2, 3)
    plt.scatter(trials.idxs_vals[1]['a'], trials.idxs_vals[0]['a'], s=1)
    plt.axvline(best['a'], color='red' , label='Best x: '+str(best['a']))
    plt.ylabel('Iteration')
    plt.xlabel('x')
    plt.title('x vs iteration')
    plt.legend()

    # Show the histogram of x-values extracted during the scan.
    plt.subplot(3, 2, 4)
    plt.hist(trials.idxs_vals[1]['a'], bins=20)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('X-values extracted during the scan (TPE)')

    # Repeat the previous point now using a random search algorithm.
    trials = Trials()
    best = fmin(objective, space=a, algo=rand.suggest, max_evals=2000, trials=trials)

    print('Best value of x: ', best)

    # Show scatter plot with the x-value vs iteration together with the final best value of x.
    plt.subplot(3, 2, 5)
    plt.scatter(trials.idxs_vals[1]['a'], trials.idxs_vals[0]['a'], s=1)
    plt.axvline(best['a'], color='red', label='Best x: '+str(best['a']))
    plt.ylabel('Iteration')
    plt.xlabel('x')
    plt.title('x vs iteration')
    plt.legend()

    # Show the histogram of x-values extracted during the scan.
    plt.subplot(3, 2, 6)
    plt.hist(trials.idxs_vals[1]['a'], bins=20)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('X-values extracted during the scan (Random)')

    plt.show()






if __name__=='__main__':
    main()
