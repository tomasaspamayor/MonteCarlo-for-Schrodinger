## Define the methods to calculate all the Morse potential information.

import numpy as np
from scipy.optimize import curve_fit

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
        # sample from the PDF corresponding to THIS bond length
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

def morse_fitting(bond_lengths, energies, uncertainties, e_single, p0=np.array([0.17, 1.0, 1.4])):
    """
    Fit Morse potential parameters (D, A, r0) using a fixed e_single for the offset.
    
    Args:
    ...
    e_single (float): Ground state energy for a single hydrogen atom (fixed offset).
    """
    # Define a closure that fixes e_single for curve_fit
    def morse_func_fit(r, D, A, r0):
        # Calls the full morse_potential function with the fixed e_single
        return morse_potential(r, D, A, r0, e_single)
        
    popt, pcov = curve_fit(
        morse_func_fit, # This now accepts 3 fitting parameters (D, A, r0)
        bond_lengths,
        energies,
        p0=p0, 
        sigma=uncertainties,
        absolute_sigma=True,
        maxfev=20000
    )
    # Unpack only 3 fitted parameters
    D_fit, A_fit, r0_fit = popt 
    print(f"Fit results: r0 = {r0_fit}, D = {D_fit}, A = {A_fit}, Fixed E_single = {e_single}")
    return D_fit, A_fit, r0_fit, pcov # Returns only 3 params and pcov

def morse_plot(D_fit, A_fit, r0_fit, bond_lengths, energies, E_single_plot):
    """
    Plot the fitting's results.
    """
    # This calls the user's custom morse_potential, expecting the custom E_single value.
    morse_values = morse_potential(bond_lengths, D_fit, A_fit, r0_fit, E_single_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(bond_lengths, energies, label='Generated samples', s=7.5)
    plt.plot(bond_lengths, morse_values, label='Fitted Morse Potential')
    plt.ylabel('Potential (a.u.)', fontsize=12)
    plt.xlabel('Bond Length (a.u.)', fontsize=12)
    plt.title('VMC Data Fitted with Custom Morse Potential', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
