#Construct a class which constructs and evaluates a 1D polynomial model with the following API:

#   - the class constructor must take the polynomial degree as argument.
#   - implement a set_parameters and get_parameters methods to update the parameter list.
#   - provide an execute method to access the polynomial prediction at a specific value in x.

import matplotlib.pyplot as plt


class Polynomial:
    def __init__(self, name, degree, coeff): #take degree and coefficients as argument
        self.name = name
        self.degree = degree
        self.coeff = coeff

    def set_parameters(self, new_degree, new_coeff): #update the parameter list
        self.degree = new_degree
        self.coeff = new_coeff

    def get_parameters(self):
        pars = [self.degree, self.coeff]
        return pars

    def eval(self, x):
        y=0
        for n in range(self.degree+1):
            y+=self.coeff[n]*x**(n)
        return y


def main():

    dim = 10
    x = range(dim)
    y = []
    for n in range(dim):
        y.append(Polynomial('nome?',2, [10,2,3]).eval(n))

    plt.figure(figsize=(10,7))
    plt.grid(True)

    plt.plot(x,y)
    plt.show()
    

if __name__ == "__main__":
    main()