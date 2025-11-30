"""
Hold the different Hamiltonians used throughout the project.
"""

import numpy as np

def energy_expectation(points, theta):
    """
    Energy expectation for hydrogen ground state.

    Args:
    points (list): The 3D points at which to evaluate the energy. 
    theta (float): The parameter for the wavefunction
    """
    r = np.linalg.norm(points, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_energies = -0.5 * (theta ** 2 - 2 * theta / r) - 1.0 / r
        local_energies = np.nan_to_num(local_energies, nan=-0.5*theta**2)

    return np.mean(local_energies)

def energy_expectation_theta_derivative(points, theta):
    """
    Derivative of the energy expectation with respect to theta for hydrogen 
    ground state.

    Args:
    points (list): The 3D points at which to evaluate the energy. 
    theta (float): The parameter for the wavefunction
    """
    r = np.linalg.norm(points, axis=1)

    local_energies = -0.5 * (theta ** 2 - 2 * theta / r) - 1.0 / r
    e_avg = np.mean(local_energies)
    dlnpsi_dtheta = - r
    grad_e = 2 * (np.mean(local_energies * dlnpsi_dtheta) - e_avg * np.mean(dlnpsi_dtheta))

    return grad_e
