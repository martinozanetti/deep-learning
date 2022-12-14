# Write a code which starting from a specific space dimension N allocates:
# a random vector v of real double-precision (numpy.float64) of size N.

N=1000

# Allocate a random vector v of real double-precision (numpy.float64) of size N.

import numpy as np
dtype=np.float64
v = np.random.rand(N).astype(dtype)


# Allocate a random square matrix A with size NxN.

A=np.random.rand(N,N)


# Implement a function which performs the dot product using only python primitives.

def dot_product(v,A):
    result=np.zeros(N)
    for i in range(N):
        for j in range(N):
            result[i]=result[i]+v[j]*A[j,i]
    return result


# Measure the execution time of the previous function and compare with NumPy's dot method.

import time
start_time = time.time()
dot_product(v,A)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
np.dot(v,A)
print("--- %s seconds ---" % (time.time() - start_time))


# Accelerate the python dot product using the Numba library.

from numba import jit
@jit

def dot_product_acc(v,A):
    result=np.zeros(N)
    for i in range(N):
        for j in range(N):
            result[i]=result[i]+v[j]*A[j,i]
    return result

def dot_product_acc2(v,A):
    np.dot(v,A)
    

# Compare the performance results.

start_time = time.time()
dot_product_acc(v,A)
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
dot_product_acc2(v,A)
print("--- %s seconds ---" % (time.time() - start_time))

