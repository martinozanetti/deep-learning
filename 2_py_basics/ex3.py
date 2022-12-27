# Using Matplotlib, plot the function exp(-x) * cos(2*pi*x) for 100 points in x linearly spaced from 0 to 5

import numpy as np
import matplotlib.pyplot as plt

# allocate array x with 100 points linearly spaced from 0 to 5

N = 100
x = np.linspace(0,5, N)

# allocate array y with the function exp(-x) * cos(2*pi*x)

y = np.exp(-x) * np.cos(2*np.pi*x)


# plot the function and show it

plt.plot(x,y)
plt.show()
