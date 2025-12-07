"""
Defines polynomial evaluation. Could be substituted with numpy package.
"""

import numpy as np

def polynomial(x, coeff):
    """
    Creates a polynomial by summing monomials with specified coefficients.

    Args:
    x (float): The independent variable value
    coeff (list): The coefficients in increasing order of monomial.

    Returns:
    float: Polynomial value at each point.
    """
    n = len(coeff)
    terms = []
    for i in range(n):
        term = coeff[i] * (x ** i)
        terms.append(term)
    terms_array = np.array(terms)
    poly = np.sum(terms_array)
    return poly
