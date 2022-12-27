import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

# Define a true function true_function(x) = cos(1.5 * pi * x).
def true_function(x):
    return np.cos(1.5 * np.pi * x)

# ... and 3 polynomial functions of degree 1, 4 and 15.
def poly1(x, a, b):
        return  a * x + b

def poly4(x, a, b, c, d, e):
        return  a * x**4 + b * x**3 + c * x**2 + d * x + e

def poly15(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
        return  a15*x**15+a14*x**14+a13*x**13+a12*x**12+a11*x**11+a10*x**10+a9*x**9+a8*x**8 + \
                a7*x**7+a6*x**6+a5*x**5+a4*x**4+a3*x**3+a2*x**2+a1*x+a0


def main():
    N=30
    # Generate N random points in x between [0, 1].
    np.random.seed(0) # <-- to always get the same random numbers
    x = np.sort(np.random.rand(N))

    # Evaluate the target points as: true_function(x) + np.random.rand() * 0.1
    y = true_function(x) + np.random.rand(N) * 0.1
    x_test = np.linspace(0, 1, 100)

    # Implement and perform polynomial fits for degrees 1, 4 and 15. Use as loss function the MSE function.
    plt.figure(figsize=(14, 5))
    degrees = [1, 4, 15]

    # *** Mode 1 - using least squares
    ax = plt.subplot(1, len(degrees), 1) # <-- firts plot

    for degree in degrees:
        p = np.polyfit(x, y, degree)
        z = np.poly1d(p) # a 1D polinomial of degree "degree" with coefficients p
        plt.plot(x_test, z(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_function(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Polyfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    # *** Mode 2 - curve fitting
    ax = plt.subplot(1, len(degrees), 2) # <-- second plot

    popt1, pcov1 = curve_fit(poly1, x, y)
    popt4, pcov4 = curve_fit(poly4, x, y)
    popt15, pcov15 = curve_fit(poly15, x, y)

    plt.plot(x_test, poly4(x_test, *popt4), label="Poly degree=4")
    plt.plot(x_test, poly1(x_test, *popt1), label="Poly degree=1")
    plt.plot(x_test, poly15(x_test, *popt15), label="Poly degree=15")
    plt.plot(x_test, true_function(x_test), label="True function")

    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Scipy.curve_fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    # *** Mode 3 - scipy minimize
    ax = plt.subplot(1, len(degrees), 3) # <-- third plot

    # The loss function 
    def loss(p, func):
        ypred = func(p)
        return np.mean(np.square(ypred(x) - y))

    for degree in degrees:
        res = minimize(loss, np.zeros(degree+1), args=(np.poly1d), method='BFGS')
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
