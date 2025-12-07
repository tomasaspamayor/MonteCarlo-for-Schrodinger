"""
Define wavefunctions and PDFs used throughout the project.
"""

import math
import numpy as np
from scipy.stats import norm

import methods.polynomials as poly

### First we define two sampling functions to try our algorithms over.

def normalized_gaussian(x, sigma=0.2):
    """
    Returns the function values of a Gaussian, normalised over the range [-1, 1].

    Args:
    x (list): The sample points.
    sigma (float): The width of the gaussian.

    Returns:
    np.array: Values of the Gaussian over the sample points.
    """
    norm_factor = norm.cdf(1, scale=sigma) - norm.cdf(-1, scale=sigma)
    vals = np.array((1 / (sigma * np.sqrt(2 * np.pi)))
                    * np.exp(-x**2 / (2 * sigma**2)) / norm_factor)
    return vals

def gaussian_3d(xyz, sigma=0.2):
    """
    Returns the function values of a 3D Gaussian.

    Args:
    x (list): The sample points.
    sigma (float): The width of the gaussian.

    Returns:
    np.array: Values of the Gaussian over the sample points.
    """
    x, y, z = xyz
    r_squared = x**2 + y**2 + z**2
    vals = np.array(np.exp(-r_squared / (2 * sigma**2)))

    return vals

def normalized_exponential(x, lambd=2.0):
    """
    Returns the function values of an exponential, normalised over the range [0, 1].

    Args:
    x (list): The sample points.
    lambd (float): The decay rate of the exponential.

    Returns:
    np.array: Values of the exponential over the sample points.
    """
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    norm_factor = 1 - np.exp(-lambd * 1)
    vals = np.array((lambd * np.exp(-lambd * x)) / norm_factor)

    return vals

### Now we define the different wavefunctions and their PDFs for the different
### systems we work with.

# Quantum Harmonic Oscillator Methods:

def wavefunction_qho(x, coeffs):
    """
    Calculate the QHO wavefunction.

    Args:
    x (list): Points at which to evaluate it.
    coeffs (list): Hermite polynonial coefficients.

    Returns:
    np.array: The wavefunction values at the sample points.
    """
    h_n = np.polynomial.polynomial.polyval(x, coeffs)
    return h_n * np.exp(-x ** 2 / 2)

def wavefunction_qho_pdf(x, n, coeffs):
    """
    Calculate the QHO wavefunction's PDF.

    Args:
    x (list): Points at which to evaluate it.
    n (int): Order of the Hermite polynomials.
    coeffs (list): Hermite polynonial coefficients.

    Returns:
    np.array: The wavefunction values at the sample points.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        h_n = poly.polynomial(x, coeffs[n])
    else:  # array
        h_n = np.array([poly.polynomial(xi, coeffs[n]) for xi in x])
    normalization = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    wavefunction_vals = h_n * np.exp(-x**2 / 2) * normalization
    pdf = wavefunction_vals ** 2
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    pdf = np.array(pdf)
    return pdf

# Hydrogen Atom Methods:

def wavefunction_hydrogen_atom(samples, theta):
    """
    Calculate the hydrogen atom's wavefunction.

    Args:
    samples (list): The points (3D) at which to calculate the value.
    theta (float): The wavefunction parameter.

    Returns:
    np.array: List of wavefunction values at the sample points.
    """
    r = np.linalg.norm(samples, axis=1)
    vals = np.array(np.exp(-theta * r))

    return vals

def wavefunction_hydrogen_atom_pdf(point, theta):
    """
    Calculate the hydrogen atom's wavefunction PDF.

    Args:
    point (list): The points (3D) at which to calculate the value.
    theta (float): The wavefunction parameter.

    Returns:
    np.array: List of wavefunction values at the sample points.
    """
    r = np.linalg.norm(point)
    normalization = theta ** 3 / np.pi
    vals = np.array(normalization * np.exp(-2 * theta * r))

    return vals

# Hydrogen Molecule Methods:

def wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2, bond_length=None):
    """
    Returns the wavefunction values for the two-hydrogen molecule system. Note 
    that this can only be performed for a single point. If many wavefunction
    values must be calculated, one must loop over their arrays calling this
    function for every each individual point.

    Args:
    r1 (list): Position of the first electron (3D).
    r2 (list): Position of the second electron (3D).
    theta (list): Parameters for each dimension (3D).
    q1 (list): Position of the first hydrogen atom (3D).
    q2 (list): Position of the second hydrogen atom (3D).
    bond_length (float): Instead of q1 and q2, atoms get positioned on the z-ax.

    Returns:
    float: Wavefunction value.
    """
    theta_1 = theta[0]
    theta_2 = theta[1]
    theta_3 = theta[2]
    r1 = np.array(r1)
    r2 = np.array(r2)
    r_12 = np.linalg.norm((r1 - r2))
    if bond_length is None:
        q1 = np.array(q1)
        q2 = np.array(q2)
    else:
        q1 = np.array([0, 0, - bond_length / 2])
        q2 = np.array([0, 0, bond_length / 2])

    exp_term = np.e ** (- theta_2 / (1 + theta_3 * r_12))
    first_term = np.e ** (- theta_1 * (np.linalg.norm((r1 - q1)) + np.linalg.norm((r2 - q2))))
    second_term = np.e ** (- theta_1 * (np.linalg.norm((r1 - q2)) + np.linalg.norm((r2 - q1))))

    val = exp_term * (first_term + second_term)

    return val

def wavefunction_hydrogen_molecule_theta_derivative(r1, r2, theta, j, q1, q2):
    """
    Analytic derivative of ln of wavefunction w.r.t. theta at a single point.

    Args:
        r1 (list): Electron 1 position.
        r2 (list): Electron 2 position.
        theta (list): Wavefunction parameter.
        j (int): index of the parameter (0,1,2)
        q1 (list): Nucleus 1 position.
        q2 (list): Nucleus 2 position.

    Returns:
        float: ln of wavefunction's derivative w.r.t. parameter theta.
    """
    r12 = np.linalg.norm(r1 - r2)
    r1q1 = np.linalg.norm(r1 - q1)
    r1q2 = np.linalg.norm(r1 - q2)
    r2q1 = np.linalg.norm(r2 - q1)
    r2q2 = np.linalg.norm(r2 - q2)

    theta1, theta2, theta3 = theta

    if j == 0:
        first_term = -(r1q1 + r2q2)
        second_term = -(r1q2 + r2q1)
        val = (np.exp(-theta1 * (r1q1 + r2q2)) * first_term
              + np.exp(-theta1 * (r1q2 + r2q1)) * second_term) / \
              (np.exp(-theta1 * (r1q1 + r2q2)) + np.exp(-theta1 * (r1q2 + r2q1)))
        return val
    if j == 1:
        return 1.0 / (1 + theta3 * r12)
    if j == 2:
        return -theta2 * r12 / (1 + theta3 * r12) ** 2
