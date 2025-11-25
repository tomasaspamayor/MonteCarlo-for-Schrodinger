import math
import numpy as np
from scipy.stats import norm

import two_one.polynomials as poly

def normalized_gaussian(x, sigma=0.2):
    """Gaussian normalized over [-1, 1]"""
    norm_factor = norm.cdf(1, scale=sigma) - norm.cdf(-1, scale=sigma)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2)) / norm_factor

def normalized_exponential(x, lambd=2.0):
    """Exponential decay normalized over [0, 1]"""
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    norm_factor = 1 - np.exp(-lambd * 1)
    return (lambd * np.exp(-lambd * x)) / norm_factor

def wf_pdf(x, n, coeffs):
    """
    Return the wavefunction PDF value at a given position in 1D space.

    Args:
    R (float) - The position in 1D space.
    n (int) - Order of the Hermite polynomial.
    coeffs (list) - List of lists of the Hermite coefficients in increasing order.
    """
    norm_term = np.sqrt(np.pi) * (2 ** n) * math.factorial(n)

    poly_val = poly.polynomial(x, coeffs[n])
    expo_val = np.e ** (- (x ** 2) / 2)
    wavefunction_val = poly_val * expo_val

    density_func = (wavefunction_val ** 2) / norm_term

    return density_func

def wf_3d(*args, n, coeffs):
    """
    Returns the value of the 3D wavefunction density function.
    
    """
    if len(args) == 1:
        x, y, z = args[0]
    else:
        x, y, z = args

    nx, ny, nz = n
    pdf_x = wf_pdf(x, nx, coeffs)
    pdf_y = wf_pdf(y, ny, coeffs)
    pdf_z = wf_pdf(z, nz, coeffs)
    pdf_val = pdf_x * pdf_y * pdf_z

    return pdf_val

