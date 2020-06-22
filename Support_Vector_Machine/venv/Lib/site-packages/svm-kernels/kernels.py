import numpy as np
from k_functions import k_chebyshev, k_hermite



def g(k, **kwargs):
    def kernel(X, Z):
        return np.array([np.array([k(x, z, kwargs) for z in Z]) for x in X])

    return kernel

def hermite(degree=2):
    return g(k_hermite, degree=degree)


def chebyshev(degree=2):
    return g(k_chebyshev, degree=degree)
