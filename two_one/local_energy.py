import math
import numpy as np
import matplotlib.pyplot as plt
import two_one.differentiators as diff

def wavefunction(x, coeffs):
    """ Calculates wavefunction values.
    Args:
    x (float / list) - Points at which to evaluate the wavefunction.
    coeffs (list) - increasing order coefficients of the polynomial.
    """
    # coeffs are assumed lowest-first: [a0, a1, ..., an]
    H_n = np.polynomial.polynomial.polyval(x, coeffs)
    n = len(coeffs) - 1
    normalization = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(math.pi))
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
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_second(x_vals, wavefunction_vals,
                                                     stepsize, coeffs, polynomial=False)
    elif method == 2:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_fourth(x_vals, wavefunction_vals,
                                                     stepsize, coeffs, polynomial=False)
    elif method == 3:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_sixth(x_vals, wavefunction_vals,
                                                     stepsize, coeffs, polynomial=False)
    elif method == 4:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_tenth(x_vals, wavefunction_vals,
                                                     stepsize, coeffs, polynomial=False)
    else:
        x_val_calc, sec_der_vals, wf_trimmed = diff.fd_eighth(x_vals, wavefunction_vals,
                                                     stepsize, coeffs, polynomial=False)

    threshold = 1e-6
    mask = np.abs(wf_trimmed) > threshold
    x_filtered = x_val_calc[mask]
    sec_der_filtered = sec_der_vals[mask] 
    wf_filtered = wf_trimmed[mask]

    l_energy = - 0.5 * sec_der_filtered / wf_filtered + 0.5 * (x_filtered ** 2)

    return x_filtered, l_energy

def analytical_local_energy(x, coeffs, n):
    """"
    Calculate the analytical local energy of the system (solve the Schrödinger Eq.)

    Args:
    x (list) - Array of points at which to evaluate the local energy
    coeffs (list) - Coefficients of the Hermite polynomials
    n (int) - Order of the Hermite Polys
    """
    x = np.array(x)
    coeffs = coeffs[n]
    coeffs_first_der = np.polyder(coeffs[::-1])[::-1]
    coeffs_second_der = np.polyder(np.polyder(coeffs[::-1]))[::-1]

    if len(coeffs_first_der) == 0:
        coeffs_first_der = [0]
    if len(coeffs_second_der) == 0:
        coeffs_second_der = [0]

    hermite_n = np.polynomial.polynomial.polyval(x, coeffs)
    hermite_fd_n = np.polynomial.polynomial.polyval(x, coeffs_first_der)
    hermite_sd_n = np.polynomial.polynomial.polyval(x, coeffs_second_der)

    norm = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))

    wf = norm * hermite_n * np.e**(-(x**2)/2)
    wf_second_der = norm * (hermite_sd_n - 2 * x * hermite_fd_n
                            + (x ** 2 - 1) * hermite_n ) * np.exp(- (x ** 2) / 2)

    l_e = -0.5 * wf_second_der / wf + 0.5 * x**2

    return l_e

def plot_local_energies(energy_vals, bins, order, truncation):
    """Plot the local energy values."""
    plt.hist(energy_vals, bins=bins, density=True, alpha=0.7)

    # Reasonable limits for harmonic oscillator
    expected_energy = order + 0.5
    plt.xlim(expected_energy - 2, expected_energy + 2)
    plt.axvline(expected_energy, color='red', linestyle='--',
                label=f'Expected: {expected_energy}')
    plt.grid()
    plt.xlabel('Local Energy E_l')
    plt.ylabel("Probability Density")
    plt.title(f'Local Energy for n={order}, {truncation}-order FD')
    plt.legend()
    plt.show()
