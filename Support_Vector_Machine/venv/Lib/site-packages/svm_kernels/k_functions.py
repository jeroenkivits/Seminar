import numpy as np
from .polynomials import hermite_poly, chebyshev_poly

def k_hermite(x, z, kwargs):
    degree = kwargs['degree']
    k = 0
    for i in range(0, degree + 1):
        k = k + np.dot(hermite_poly(x, i), hermite_poly(z, i))
    return k / (np.sqrt(np.pi) * np.math.factorial(degree) / (2 ** degree))


def k_chebyshev(x, z, kwargs):
    degree = kwargs['degree']
    k = 0
    for i in range(0, degree + 1):
        k = k + np.dot(chebyshev_poly(x, i), chebyshev_poly(z, i))
    return k / (np.sqrt(len(x) - np.dot(x, z)))