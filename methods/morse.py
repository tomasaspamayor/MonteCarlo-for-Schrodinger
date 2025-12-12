## Define the methods to calculate all the Morse potential information.

import numpy as np

import matplotlib.pyplot as plt
import methods.sampling as samp
import methods.hamiltonians as ham

plt.style.use('seaborn-v0_8-paper')

def bond_length_energies(bl_range, theta, n):
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
    energies_uncertainties = []

    for b in bond_lengths:
        samples = samp.samplings_h2_molecule(
            bond_length=b, 
            initial_point=None,
            theta=theta,
            domain=None,
            stepsize=0.15,
            num_samples=250000,
            burnin_val=20000,
            adapt_interval=40000
        )

        e, energy_locals = ham.h2_energy_expectation(samples, b, theta)
        e_unc = ham.h2_energy_expectation_uncertainty(energy_locals)
        energies.append(e)
        energies_uncertainties.append(e_unc)

        print(f'Bond length {b} calculated.')

    return bond_lengths, np.array(energies), np.array(energies_uncertainties)

def morse(r, D, a, r0):
    E_single = -0.5
    return D * (1 - np.exp(-a * (r - r0))) ** 2 - D + 2 * E_single
