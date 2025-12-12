"""
Calculate a variety of different local energies for a QHO system and a
Hydrogen ground state system. Since the Schrödinger equation is different
for the two, they must be used carefully.
"""
import numpy as np

import methods.differentiators as diff
from methods import pdfs
import methods.hamiltonians as ham

## Quantum Harmonic Oscillator Methods:

def local_energy_qho_numerical(x_samples, h, coeffs, level, method=None):
    """
    Compute the local energy for the QHO using numerical derivatives at each
    sample point.
    
    Args:
    x_samples (array): Random samples from the wavefunction PDF
    h (float): Stepsize for finite differences
    coeffs (list): Hermite polynomial coefficients
    level (int): Order of Hermite polynomial
    method (int): Finite difference method (2=2nd order, 4=4th order, etc.)

    Returns:
    np.array: Array with all the calculated local energies
    float: Mean local energy
    """
    local_energies = []

    psi_values = pdfs.wavefunction_qho(x_samples, coeffs[level])

    if method == 2:  # Second order
        psi_double_prime = diff.cdm_step_second(x_samples, pdfs.wavefunction_qho, h, coeffs, level)

    elif method == 4:  # Fourth order
        psi_double_prime = diff.cdm_step_fourth(x_samples, pdfs.wavefunction_qho, h, coeffs, level)

    elif method == 6:  # Sixth order
        psi_double_prime = diff.cdm_step_sixth(x_samples, pdfs.wavefunction_qho, h, coeffs, level)

    elif method == 10:  # Tenth order
        psi_double_prime = diff.cdm_step_tenth(x_samples, pdfs.wavefunction_qho, h, coeffs, level)

    else:  # Eigth order - Predefined.
        psi_double_prime = diff.cdm_step_eighth(x_samples, pdfs.wavefunction_qho, h, coeffs, level)

    s = len(x_samples)
    for i in range(s):
        psi = psi_values[i]
        psi_dd = psi_double_prime[i]
        x = x_samples[i]

        if abs(psi) > 1e-12:
            local_energy_val = -0.5 * psi_dd / psi + 0.5 * x ** 2
            local_energies.append(local_energy_val)
        else:
            epsilon = 1e-12
            if abs(psi) > epsilon:
                safe_psi = psi
            elif psi != 0:
                safe_psi = epsilon * np.sign(psi)
            else:
                safe_psi = epsilon
            local_energy_val = -0.5 * psi_dd / safe_psi + 0.5 * x ** 2
            local_energies.append(local_energy_val)

    local_energies = np.array(local_energies)
    mean_local_energy = np.mean(local_energies)

    return local_energies, mean_local_energy

## Hydrogen Atom Methods:

def local_energy_analytical(x, func, sec_der, theta):
    """
    Defines the analytical calculation for the wavefunction of the Hydrogen
    Atom ground state.

    Args:
    x (list): The points at which to compute the local energy.
    func (callable): The wavefunction.
    sec_der (callable): The wavefunction's second derivative.
    theta (float): The wavefunction's parameter.

    Returns:
    np.array: Values for the local energy at all points.
    """
    vals = -0.5 * (1 / func(x, theta)) * sec_der(x, theta) + 0.5 * x ** 2
    vals = np.array(vals)
    return vals

## Hydrogen Molecule Methods:

def h2_local_energy(r1, r2, theta, q1, q2):
    """
    Calculate the local energy of an H_2 molecule's electrons. Uses CDM to compute
    derivatives.

    Args:
    r1 (list): First electron's position (3D).
    r2 (list): Second electron's position (3D).
    theta (list): Wavefunction parameter.
    q1 (list): First nucleus' position (3D).
    q2 (list): Second nucleus' position (3D).

    Returns:
    float: Local energy value.
    """
    r1A = np.linalg.norm(r1 - q1)
    r1B = np.linalg.norm(r1 - q2)
    r2A = np.linalg.norm(r2 - q1)
    r2B = np.linalg.norm(r2 - q2)
    r12 = np.linalg.norm(r1 - r2)
    R = np.linalg.norm(q1 - q2)

    wf_vals = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2)

    if abs(wf_vals) < 1e-10:
        return 0.0

    k_term = ham.h2_k_term(r1, r2, theta, q1, q2)
    potential_term = -1/r1A - 1/r1B - 1/r2A - 1/r2B + 1/r12 + 1/R

    local_energy = k_term + potential_term
    return local_energy

def h2_le_sym(r1, r2, theta, q1, q2):
    """
    Calculate the local energy of an H_2 molecule. Uses CDM to compute
    derivatives. Introduces symmetry arguments due to indistinguishability of
    fermions.

    Args:
    r1 (list): First electron's position (3D).
    r2 (list): Second electron's position (3D).
    theta (list): Wavefunction parameter.
    q1 (list): First nucleus' position (3D).
    q2 (list): Second nucleus' position (3D).

    Returns:
    float: Local energy value.
    """
    le_even = h2_local_energy(r1=r1, r2=r2, theta=theta, q1=q1,q2=q2)
    le_odd = h2_local_energy(r1=r2, r2=r1, theta=theta, q1=q1,q2=q2)
    return 0.5 * (le_even + le_odd)
