import math
import numpy as np
from scipy.stats import norm

import two_one.polynomials as poly

def normalized_gaussian(x, sigma=0.2):
    """Gaussian normalized over [-1, 1]"""
    norm_factor = norm.cdf(1, scale=sigma) - norm.cdf(-1, scale=sigma)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2)) / norm_factor

def gaussian_3d(xyz, sigma=0.2):
    """3D Gaussian"""
    x, y, z = xyz
    r_squared = x**2 + y**2 + z**2
    return np.exp(-r_squared / (2 * sigma**2))

def normalized_exponential(x, lambd=2.0):
    """Exponential decay normalized over [0, 1]"""
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    norm_factor = 1 - np.exp(-lambd * 1)
    return (lambd * np.exp(-lambd * x)) / norm_factor

def wf_pdf(x, n, coeffs):
    x = np.asarray(x)    
    if x.ndim == 0:
        H_n = poly.polynomial(x, coeffs[n])
    else:  # array
        H_n = np.array([poly.polynomial(xi, coeffs[n]) for xi in x])
    normalization = 1.0 / np.sqrt((2 ** n) * math.factorial(n) * np.sqrt(np.pi))
    wavefunction = H_n * np.exp(-x**2 / 2) * normalization
    return wavefunction ** 2

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
