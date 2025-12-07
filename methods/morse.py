## Define the methods to calculate all the Morse potential information.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import methods.sampling as samp
import methods.hamiltonians as ham

def bond_length_energies(bl_range, theta, n, num_samples=200000, burnin=20000, stepsize=0.15):
    """
    Compute the energies of the molecule over a range of bond lenghts.
    
    :param bl_range: Description
    :param theta: Description
    :param n: Description
    :param num_samples: Description
    :param burnin: Description
    :param stepsize: Description
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

        E = ham.h2_energy_expectation(samples, b, theta)
        energies.append(E)

    return bond_lengths, np.array(energies)

def morse_potential(r, D, a, r0, e_single):
    """ Returns the Morse potential values"""
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2 * e_single

def morse_fitting(bond_lengths, energies, e_single, p0=np.array([0.17, 1.0, 1.4])):
    """
    Fits the morse data to the model.
    
    bond_lengths (list): Bond lengths at which the energies were computed
    energies (list): The computed energy values
    e_single (float): The energy value of a single hydrogen atom.
    """

    def morse_func_fit(r, D, a, r0):
        return morse_potential(r, D, a, r0, e_single)

    popt, pcov= curve_fit(morse_func_fit, bond_lengths, energies, p0, maxfev=20000)

    D_fit, a_fit, r0_fit = popt

    print(f'The obtained results are: r_0 = {r0_fit} and D = {D_fit}.')
    print('The experimental values are: r_0 = 1.14 and D = 0.17.')
    return D_fit, a_fit, r0_fit, pco

def morse_plot(D_fit, a_fit, r0_fit, bond_lengths, energies, e_single):
    """
    Plot the fitting's results.
    
    D_fit (float): Fitted parameter D
    a_fit (float): Fitted parameter a
    r0_fit (float): Fitted parameter r_0
    bond_lengths (list): Bond lengths at which to compute the potential.
    energies (list): Calculated energies at said bond_lengths
    e_single (float): Energy of a single hydrogen atom.
    """

    morse_values = morse_potential(bond_lengths, D_fit, a_fit, r0_fit, e_single)

    plt.scatter(bond_lengths, energies, label='Generated samples')
    plt.plot(bond_lengths, morse_values, label='Fitted Morse Potential')
    plt.ylabel('Potential (Ht)')
    plt.xlabel('Bond Length (a.u.)')
    plt.legend()
    plt.show()
