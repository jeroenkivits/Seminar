import numpy as np

def hermite_poly(x, degree):
    if degree == 0:
        return 1
    if degree == 1:
        return x
    if degree > 1:
        return np.dot(x.T, hermite_poly(x, degree - 1)) - degree * hermite_poly(x, degree - 2)

def chebyshev_poly(x, degree):
    if degree == 0:
        return 1
    if degree == 1:
        return x
    if degree > 1:
        return np.dot(2 * x.T, chebyshev_poly(x, degree - 1)) - chebyshev_poly(x, degree - 2)