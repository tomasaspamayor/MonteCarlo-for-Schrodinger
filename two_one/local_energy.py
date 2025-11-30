"""
Calculate a variety of different local energies for a QHO system and a
Hydrogen ground state system. Since the Schrödinger equation is different
for the two, they must be used carefully.
"""
import numpy as np
import matplotlib.pyplot as plt
import two_one.differentiators as diff
import two_two.pdfs as pdfs

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

def plot_local_energies(energy_vals, bins, order, truncation):
    """
    Plot the local energy values in a histogram.
    
    Args:
    energy_vals (list): The energy values obtained.
    bins (int): Number of bins.
    order (int): Order of the Hermite polynomial used.
    truncation (int): Order of the truncation used.

    Returns:
    plt.plot: The resulting histogram.
    """
    plt.hist(energy_vals, bins=bins, density=True, alpha=0.7)

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
