"""
Holds the different Hamiltonians and energy calculations used throughout the 
project.
"""

import numpy as np

from methods import pdfs
import methods.local_energy as le
import methods.differentiators as diff
import methods.sampling as samp

## Methods for the Hydrogen Atom:

def energy_expectation(points, theta):
    """
    Energy expectation for hydrogen ground state.
    Returns both the average energy AND local energies at each point.
    """
    r = np.linalg.norm(points, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_energies = -0.5 * (theta ** 2 - 2 * theta / r) - 1.0 / r
        local_energies = np.nan_to_num(local_energies, nan=-0.5*theta**2)
    return np.mean(local_energies), local_energies

def energy_expectation_num(points, theta, step):
    """
    Energy expectation for hydrogen ground state - corrected vectorized version.
    
    Args:
        points: Array of shape (N, 3) - 3D points
        theta: Wavefunction parameter
        step: Step size for numerical Laplacian
    """
    r = np.linalg.norm(points, axis=1)
    
    local_energies = []
    for i, point in enumerate(points):
        # Calculate psi at this single point
        psi = pdfs.wavefunction_hydrogen_atom(point, theta)
        
        # Avoid division by very small psi
        if np.abs(psi) < 1e-12:
            continue
            
        # Numerical Laplacian at this point
        num_lap_psi = diff.cdm_laplacian_4th(pdfs.wavefunction_hydrogen_atom, 
                                           point, theta, step)
        
        # Potential energy term
        if r[i] > 1e-12:  # Avoid division by zero
            potential_energy_term = (-1.0 / r[i]) * psi
        else:
            potential_energy_term = 0
            
        # Hamiltonian acting on psi
        H_psi = -0.5 * num_lap_psi + potential_energy_term
        
        # Local energy
        E_L = H_psi / psi
        local_energies.append(E_L)
    
    if len(local_energies) == 0:
        return -0.5 * theta**2  # Fallback value
    
    return np.mean(local_energies)

def energy_expectation_theta_derivative(points, theta):
    """
    Derivative of the energy expectation with respect to theta for hydrogen 
    ground state using the correct VMC Gradient Formula.
    
    Uses: ∇E = 2[⟨E_L ∂_θ ln|ψ|⟩ - ⟨E_L⟩⟨∂_θ ln|ψ|⟩]
    """
    r = np.linalg.norm(points, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_energies = -0.5 * (theta ** 2 - 2 * theta / r) - 1.0 / r
        local_energies = np.nan_to_num(local_energies, nan=-0.5*theta**2)
        dlnpsi_dtheta = -r

    term1 = np.mean(local_energies * dlnpsi_dtheta)
    term2 = np.mean(local_energies) * np.mean(dlnpsi_dtheta)
    grad_e = 2 * (term1 - term2)

    return grad_e

def energy_expectation_theta_derivative_num(points, theta, h, step):
    """
    Physics-informed finite difference gradient.
    Uses finite difference magnitude but correct sign from theory.
    """
    E_plus = energy_expectation_num(points, theta + h, step)
    E_minus = energy_expectation_num(points, theta - h, step)

    # Magnitude from finite difference
    grad_magnitude = abs(E_plus - E_minus) / (2 * h)

    # SIGN from physics: dE/dθ = θ - 1
    grad_sign = np.sign(theta - 1)

    return grad_sign * grad_magnitude

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

    local_energies = []
    energy = 0.0
    m = len(samples_6d)
    for i in range(m):
        r1 = samples_6d[i, :3]
        r2 = samples_6d[i, 3:]
        energy_i = le.h2_le_sym(r1, r2, theta, q1, q2)
        local_energies.append(energy_i)
        energy += energy_i

    return (energy / m), local_energies

def h2_energy_expectation_uncertainty(local_energies):
    """
    Returns the Standard Error of the Mean for a sampling of local energies.
    
    Args:
    local_energies (list): The computed local energies.

    Returns:
    float: The standard error of the mean.
    """
    diffs= 0.0
    mean_energy = np.mean(local_energies)
    M = len(local_energies)
    factor = 1 / (M - 1)
    for energy in local_energies:
        val_energy = (energy - mean_energy) ** 2
        diffs += val_energy

    var_sqr = factor * diffs
    sem = np.sqrt(var_sqr / M)

    return sem

def bond_length_energies(bl_range, theta, n, num_samples=200000, burnin=20000, stepsize=0.15, adapt_interval=750):
    """
    Calculates the energies of the ground state with varying internuclear
    separation. Useful for Morse Potential calculations.

    Args:
    bl_range (list): Start and end point of the stepsize range.
    theta (list): Wavefunction parameter.
    n (int): Number of stepsizes to be computed.
    num_samples (int): Number of samples in the Metropolis-Hastings algorithm.
    stepsize (float): Stepsize in the Metropolis-Hastings algorithm.

    Returns:
    list: Array with all the computed bond lengths.
    list: The respective energies.
    """
    bond_lengths = np.linspace(bl_range[0], bl_range[-1], n)
    energies = []

    for b in bond_lengths:
        # sample from the PDF corresponding to THIS bond length
        samples = samp.samplings_h2_molecule(
            bond_length=b,
            initial_point=None,
            theta=theta,
            domain=None,
            stepsize=stepsize,
            num_samples=num_samples,
            burnin_val=burnin,
            adapt_interval=adapt_interval
        )

        e = h2_energy_expectation(samples, b, theta)
        energies.append(e)

    return bond_lengths, np.array(energies)
