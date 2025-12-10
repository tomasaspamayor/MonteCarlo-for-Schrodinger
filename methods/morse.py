## Define the methods to calculate all the Morse potential information.

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import methods.sampling as samp
import methods.hamiltonians as ham

plt.style.use('seaborn-v0_8-paper')

def bond_length_energies(bl_range, theta, n, num_samples=200000, burnin=20000, stepsize=0.15):
    """
    Compute the energies of the molecule over a range of bond lenghts.
    
    Args:
    bl_range (list): Range of bond lenghts.
    theta (list): Wavefunction parameter.
    n (int): Number of stepsizes to compute.
    num_samples (int): Number of samples in the Metropolis-Hastings algorithm.
    burnin (int): Number of samples to burn in.
    stepsize (float): Metropolis-Hastings stepsize.

    Returns:
    list: Array of bond lengths.
    list: Respective energies.
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
            burnin_val=burnin
        )

        e = ham.h2_energy_expectation(samples, b, theta)
        energies.append(e)
        print(f'Bond length {b} calculated.')

    return bond_lengths, np.array(energies)

def morse_potential(r, d, a, r0, e_single):
    """
    Returns the Morse potential values
    
    Args:
    r (list): Bond lenghts.
    d (float): Function parameter.
    a (float): Function parameter (dissociation energy).
    r0 (float): Equilibrium bond length.
    e_single (float): Ground state energy for a single hydrogen atom.

    Returns:
    float: Morse potential value.
    """
    return d * (1 - np.exp(- a * (r - r0))) ** 2 - d + 2 * e_single

def morse_fitting(bond_lengths, energies, e_single,
                  p0=np.array([0.17, 1.0, 1.4])):
    """
    Fit Morse potential parameters (D, A, r0) to computed energies.

    Args:
    bond_lengths (list) Distances at which the molecular energies were computed.
    energies (list): The associated computed energy values.
    e_single (list): Energy of a single hydrogen atom (used inside morse_potential).
    p0 (list): Initial guess for [D, A, r0].

    Returns

    D_fit (list): Fitted dissociation energy parameter.
    A_fit (list): Fitted width parameter.
    r0_fit (list): Ftitted equilibrium bond length.
    pcov (list): Covariance matrix from the fit.
    """

    def morse_func_fit(r, D, A, r0):
        return morse_potential(r, D, A, r0, e_single)

    # Run the fit
    popt, pcov = curve_fit(
        morse_func_fit,
        bond_lengths,
        energies,
        p0=p0,
        maxfev=20000
    )

    D_fit, A_fit, r0_fit = popt

    print(f"Fit results: r0 = {r0_fit}, D = {D_fit}, A = {A_fit}")
    print("Experimental vals: r0 = 1.14, D = 0.17")

    return D_fit, A_fit, r0_fit, pcov

def morse_plot(d_fit, a_fit, r0_fit, bond_lengths, energies, e_single):
    """
    Plot the fitting's results.
    
    Args:
    D_fit (float): Fitted parameter D
    a_fit (float): Fitted parameter a
    r0_fit (float): Fitted parameter r_0
    bond_lengths (list): Bond lengths at which to compute the potential.
    energies (list): Calculated energies at said bond_lengths
    e_single (float): Energy of a single hydrogen atom.

    Returns:
    plt.plot: Morse potential against distance, with fit.
    """

    morse_values = morse_potential(bond_lengths, d_fit, a_fit, r0_fit, e_single)

    plt.scatter(bond_lengths, energies, label='Generated samples', s=7.5)
    plt.plot(bond_lengths, morse_values, label='Fitted Morse Potential')
    plt.ylabel('Potential')
    plt.xlabel('Bond Length')
    plt.legend()
    plt.show()
