import numpy as np
import matplotlib.pyplot as plt

# Define a function f(x) = -sin(x*x)/x + 0.01 * x*x.

def f(x):
    return -np.sin(x*x)/x + 0.01 * x*x

# Generate an array with 100 elements, linearly spaced between [-3, 3].

x = np.linspace(-3, 3, 100)

# Write to a file output.dat the values of x and f(x) line by line.

np.savetxt("output.dat", np.column_stack((x, f(x))))

# Plot all points, with title, axis labels and a line between points, show the equation in the legend.

plt.plot(x, f(x), 'r-', markersize=1, label=r'$f(x) = -sin(x*x)/x + 0.01 * x*x$')
plt.title("Function plot")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# Bound the x-axis between x=[-3,3].

plt.xlim(-3, 3)

# Store plot to disk as output5.png.

plt.savefig("output5.png")
