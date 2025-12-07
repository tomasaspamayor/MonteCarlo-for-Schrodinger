## Define the methods to calculate all the Morse potential information.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import methods.sampling as samp
import methods.hamiltonians as ham

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

def morse_fitting(bond_lengths, energies, e_single, p0=np.array([0.17, 1.0, 1.4])):
    """
    Fits the morse data to the model.
    
    Args_
    bond_lengths (list): Bond lengths at which the energies were computed
    energies (list): The computed energy values
    e_single (float): The energy value of a single hydrogen atom.

    Returns:
    float: Fitted 'D' parameter.
    float: Fitted 'A' parameter.
    float: Fitted 'R_0' parameter.
    list: covariance matrix.
    """

    def morse_func_fit(r, d, a, r0):
        return morse_potential(r, d, a, r0, e_single)

    popt, pcov = curve_fit(morse_func_fit, bond_lengths, energies, p0, maxfev=20000)

    d_fit, a_fit, r0_fit = popt

    print(f'The obtained results are: r_0 = {r0_fit} and D = {d_fit}.')
    print('The experimental values are: r_0 = 1.14 and D = 0.17.')
    return d_fit, a_fit, r0_fit, pcov

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

    plt.scatter(bond_lengths, energies, label='Generated samples')
    plt.plot(bond_lengths, morse_values, label='Fitted Morse Potential')
    plt.ylabel('Potential (Ht)')
    plt.xlabel('Bond Length (a.u.)')
    plt.legend()
    plt.show()
