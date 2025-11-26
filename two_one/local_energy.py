import numpy as np
import math
import matplotlib.pyplot as plt
import two_one.differentiators as diff

def wavefunction(x, coeffs):
    """Return the NORMALIZED calculated wavefunction."""
    H_n = np.polyval(coeffs, x)
    
    # Get quantum number n from the polynomial degree
    n = len(coeffs) - 1  # degree of polynomial
    
    # NORMALIZATION CONSTANT for quantum harmonic oscillator
    normalization = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    
    return normalization * H_n * np.exp(-x**2 / 2)

def local_energy(x_vals, stepsize, coeffs, level, method):
    """
    Compute the local energy of a specific wavefunction.

    Args:
    stepsize - (float): Stepsize at which to sample the FDM.
    range_val - (list): Range of x-values at which evaluate the wavefunction.
    level - (int): Order of the Hermite Polynomial [0 -> 4].
    method - (bool): Whether to use fourth order truncation (==1) or second (else).
    """
    coeffs = coeffs[level]
    wavefunction_vals = wavefunction(x_vals, coeffs)

    if method == 1:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_second(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=False)
    elif method == 2:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_fourth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=False)
    elif method == 3:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_sixth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=False)
    elif method == 4:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_tenth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=False)
    else:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_eighth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=False)

    threshold = 1e-6
    mask = np.abs(wf_trimmed) > threshold
    x_filtered = x_val_calc[mask]
    sec_der_filtered = sec_der_vals[mask] 
    wf_filtered = wf_trimmed[mask]
    
    l_energy = - 0.5 * sec_der_filtered / wf_filtered + 0.5 * (x_filtered ** 2)

    return x_vals, l_energy

def plot_local_energies(energy_vals, bins, order, truncation):
    """Plot the local energies obtained."""
    plt.hist(energy_vals, bins=bins, density=True)
    plt.xlim(-3000000, 3000000)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel("E_l")
    plt.title(f'Local Energy (E_l) for the n={order} wavefunction, {truncation}th truncation.')
    plt.show()