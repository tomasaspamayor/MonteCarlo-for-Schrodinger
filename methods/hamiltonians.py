"""
Hold the different Hamiltonians used throughout the project.
"""

import numpy as np
import methods.pdfs as pdfs
import methods.sampling as samp
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

    laplacian_1 = diff.cdm_laplacian(wavefunction_e1, r1)
    laplacian_2 = diff.cdm_laplacian(wavefunction_e2, r2)

    kinetic_value = -0.5 * (laplacian_1 + laplacian_2) / wf_vals

    return kinetic_value

def h2_k_term_analytical(r1, r2, theta, q1, q2):
    """
    Computes the H_2 molecule's kintetic term with analytically calculated
    Laplacians.

    Args:
    r1 (list): First electron's position.
    r2 (list): Second electron's position.
    theta (list): Wavefunction parameter.
    q1 (list): First nucleus' position.
    q2 (list): Second nucleus' position.

    Returns:
    float: Kintetic energy of the system.
    """
    a, b, c = theta

    r1A_vec = r1 - q1
    r1A = np.linalg.norm(r1A_vec)
    r1B_vec = r1 - q2
    r1B = np.linalg.norm(r1B_vec)
    r2A_vec = r2 - q1
    r2A = np.linalg.norm(r2A_vec)
    r2B_vec = r2 - q2
    r2B = np.linalg.norm(r2B_vec)
    r12_vec = r1 - r2
    r12 = np.linalg.norm(r12_vec)

    epsilon = 1e-12
    r1A = max(r1A, epsilon)
    r1B = max(r1B, epsilon)
    r2A = max(r2A, epsilon)
    r2B = max(r2B, epsilon)
    r12 = max(r12, epsilon)

    grad_f1 = - a * (r1A_vec / r1A) - (b * c / (1 + c * r12) ** 2) * (r12_vec / r12)
    grad_f2 = - a * (r2B_vec / r2B) + (b * c / (1 + c * r12) ** 2) * (r12_vec / r12)

    laplacian_f1 = - 2 * a / r1A - 2 * b * c / (r12 * (1 + c * r12) ** 2) + b * c ** 2 / (1 + c * r12) ** 3
    laplacian_f2 = - 2 * a / r2B - 2 * b * c / (r12 * (1 + c * r12) ** 2) + b * c ** 2 / (1 + c * r12) ** 3

    kinetic = -0.5 * (laplacian_f1 + laplacian_f2 +
                      np.dot(grad_f1, grad_f1) + np.dot(grad_f2, grad_f2))

    return kinetic

def h2_energy_expectation(bond_length, theta, domain=None, initial_point=None, num_samples=10000, stepsize=0.05):
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

    def h2_6d(point_6d):
        r1 = point_6d[:3]
        r2 = point_6d[3:]
        wf_val = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2, bond_length=bond_length)
        return abs(wf_val) ** 2

    if domain is None:
        domain_size = 5.0
        domain = [(-domain_size, domain_size) for _ in range(6)]
    elif np.isscalar(domain):
        domain = [(-domain, domain) for _ in range(6)]

    if initial_point is None:
        initial_point = np.array([0.2, 0, -0.3,
                                  -0.2, 0, 0.3])

    samples_6d = samp.metropolis_hastings_3d(h2_6d, initial_point, domain,
                                             stepsize, num_samples, dimensions=6)
    energy = 0.0
    m = len(samples_6d)
    for i in range(m):
        r1 = samples_6d[i, :3]
        r2 = samples_6d[i, 3:]
        energy_i = le.h2_local_energy(r1, r2, theta, q1, q2)
        energy += energy_i

    return energy / m
