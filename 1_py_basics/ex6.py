from cmath import exp
from random import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self, size):
        raise NotImplementedError()

# Implement an inherited Normal class which
# outputs a list of normal samples [mu=0, sigma=1]
# by overriding the Variable.sample method

class Normal(Variable):
    def __init__(self, name):
        super().__init__(name) # executes MyClass constructor

    def sample(self, size):
        mu = 0
        sigma = 1
        x = np.random.normal(mu, sigma, size)

        return x

    def mysample(self, size):
        norm = []
        for n in range(size):
            s=random()
            t=random()
            x=math.sqrt(-2*math.log(1-s))*math.cos(2.*np.pi*t)
            mu = 0
            sigma = 1
            norm.append(mu + x * sigma)
        return norm

def main():

    dim = 10000
    x = range(dim)

    plt.figure(figsize=(10,7))
    plt.grid(True)

    norm = Normal(Variable).sample(dim)
    plt.hist(norm, 200)

    norm2 = Normal(Variable).mysample(dim)
    plt.hist(norm2, 200)

    plt.show()
    

if __name__ == "__main__":
    main()