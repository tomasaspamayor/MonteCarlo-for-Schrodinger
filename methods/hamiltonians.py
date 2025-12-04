"""
Hold the different Hamiltonians used throughout the project.
"""

import numpy as np
import methods.pdfs as pdfs
import methods.local_energy as le
import methods.differentiators as diff

## Methods for the Hydrogen Atom:

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

# Methods for the Hydrogen Molecule:

def h2_k_term(r1, r2, theta, q1, q2):
    """
    Computes the H_2 molecule's kintetic term with numerical Laplacians.

    Args:
    r1 (list): First electron's position.
    r2 (list): Second electron's position.
    theta (list): Wavefunction parameter.
    q1 (list): First nucleus' position.
    q2 (list): Second nucleus' position.

    Returns:
    float: Kintetic energy of the system.
    """
    wf_vals = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2)

    if abs(wf_vals) < 1e-12:
        return 0.0

    def wavefunction_e1(pos):
        return pdfs.wavefunction_hydrogen_molecule(pos, r2, theta, q1, q2)

    def wavefunction_e2(pos):
        return pdfs.wavefunction_hydrogen_molecule(r1, pos, theta, q1, q2)

    laplacian_1 = diff.cdm_laplacian(wavefunction_e1, r1, step=0.05)
    laplacian_2 = diff.cdm_laplacian(wavefunction_e2, r2, step=0.05)

    kinetic_value = -0.5 * (laplacian_1 + laplacian_2) / wf_vals

    return kinetic_value

def h2_energy_expectation(samples_6d, bond_length, theta):
    """
    Compute the energy expectation of the Hydrogen molecule.

    Args:
    bond_length (float): Interatomic distance.
    theta (list): Wavefunction parameter.
    domain (list): Each dimension's range.
    initial_point (list): Starting point for the sampling algorithm.
    num_samples (int): Number of samples.
    stepsize (float): Stepsize for the sampling algorithm

    Returns:
    float: Energy expectation value.
    """
    q1 = np.array([0, 0, -bond_length / 2])
    q2 = np.array([0, 0, bond_length / 2])

    energy = 0.0
    m = len(samples_6d)
    for i in range(m):
        r1 = samples_6d[i, :3]
        r2 = samples_6d[i, 3:]
        energy_i = le.h2_le_sym(r1, r2, theta, q1, q2)
        energy += energy_i

    return energy / m
