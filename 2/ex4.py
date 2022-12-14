# data4.dat file contains 2 columns (x,y) of points.
# Load data using numpy and use matplotlib scatter plot for the graphical representation.
# title: "Charged particles", axis titles: "x-coordinate" e "y-coordinate".
# Color points: red.


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data4.dat")
x=data[:,0]
y=data[:,1]

plt.scatter(x,y, s=1, color='red')
plt.title("Charged particles")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')

# Store plot to disk using the filename output.png

plt.savefig("output.png")
