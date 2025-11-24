import numpy as np
import matplotlib.pyplot as plt
import polynomials as poly
import differentiators as diff
import errors as err

def local_energy(stepsize, range_val, samples_num, coeffs, level, method, plot=False):
    """
    Compute the local energy of a specific wavefunction.

    Args:
    stepsize - (float): Stepsize at which to sample the FDM.
    range_val - (list): Range of x-values at which evaluate the wavefunction.
    level - (int): Order of the Hermite Polynomial [0 -> 4].
    method - (bool): Whether to use fourth order truncation (==1) or second (else).
    """
    if method == 1:
        x_vals, y_vals_first, other = diff.fd_fourth(stepsize, range_val, samples_num, coeffs)
    else:
        x_vals, y_vals_first, other = diff.fd_second(stepsize, range_val, samples_num, coeffs)

    y_vals_sec = 0.5 * (x_vals ** 2)
    l_energy = y_vals_first + y_vals_sec

    if plot is True:
        plt.plot(x_vals, l_energy)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("E_l")
        plt.title(f'Local Energy (E_l) for the wavefunction, method {method}')
        plt.show()

    return x_vals, l_energy