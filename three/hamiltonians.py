"""
Hold the different Hamiltonians used throughout the project.
"""

import numpy as np

import two_one.local_energy as le

def wavefunction(point, theta):
    """
    Returns the wavefunction ansatz value.

    Args:
    r (list) - Position at which to calculate it (3D).
    theta (float) - Parameter
    """
    r = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[3] ** 2)
    return np.e ** (-theta * r)

def dtheta_wavefunction(point, theta):
    """
    Returns the value of the ansatz's derivative with respect to the parameter.

    Args:
    r (list) - Position at which to calculate it (3D).
    theta (float) - Parameter
    """
    r = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[3] ** 2)
    return - r * np.e ** (-theta * r)

def wavefunction_secder(point, theta):
    """
    Returns the Laplacian of the wavefunction.

    Args:
    point (list) - Poistion at which to evaluate
    theta (float) - Parameter
    """
    r = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[3] ** 2)
    return - (theta / r) * np.e**(-theta * r) * (point[0] + point[1] + point[2])

def hydrogen_groundstate(x, theta):
    """
    Calculates the energy expectation values from the local energy

    Args:
    x (list) - Point or array of points where the energy must be calculated.
    theta (float) - Parameter
    """
    n = len(x)
    energy = (1 / n) * np.sum(le.analytical_local_energy_wf(x, wavefunction, wavefunction_secder, theta))
    return energy

def gradtheta_hydrogen_groundstate(x, theta):
    """
    Calculates the gradient of the energy expecation value from the local energy.

    Args:
    x (list) - Point or array of points where the energy must be calculated.
    theta (float) - Parameter
    """
    n = len(x)
    term = (le.analytical_local_energy_wf(x, wavefunction, wavefunction_secder, theta) - hydrogen_groundstate(x, theta)) * (dtheta_wavefunction(x, theta) / wavefunction(x, theta))
    grad_energy = (2 / n) * np.sum(term)

    return grad_energy
