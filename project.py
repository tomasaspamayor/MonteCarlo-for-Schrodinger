# COMPUTATIONAL PHYSICS: PROJECT CODE - Tomàs Aspa Mayor
# Variational Monte Carlo Methods to solve the Schröndinger Equation

import numpy as np
import matplotlib.pyplot as plt

## 2.1 - Finite Difference Method to find the Local Energy of SHO.

# Define the first 4 Hermite Polynomials
hermite_coeffs = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [-2, 0, 4, 0, 0], [0, -12, 0, 8, 0],
                  [12, 0, -48, 0, 16]]

def polynomial(x, coeff):
    """
    Creates a polynomial by summing monomials with specified coefficients.

    Args:
    x - (float): The independent variable value
    coeff - (list): The coefficients in increasing order of monomial.
    """
    n = len(coeff)
    terms = []
    for i in range(n):
        term = coeff[i] * (x ** i)
        terms.append(term)
    terms_array = np.array(terms)
    poly = np.sum(terms_array)
    return poly

def finite_difference(stepsize, range_val, level):
    """
    Approximates the second derivative of the Hermitian functions defined
    earlier with the central midpoint difference method. Truncation on the
    quadratic term.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    """
    x_0 = range_val[0]
    x_f = range_val[-1]
    samples = np.arange(x_0, x_f, stepsize)
    n = len(samples)
    func_vals = []

    for i in range(n):
        term = polynomial(samples[i], hermite_coeffs[level])
        func_vals.append(term)
    func_vals = np.array(func_vals)
    sec_der_vals = (func_vals[2:] - 2 * func_vals[1:-1] + func_vals[:-2]) / (stepsize ** 2)
    samples_inner = samples[1:-1]
    func_inner    = func_vals[1:-1]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals

def finite_difference_fourth(stepsize, range_val, level):
    """
    Approximates the second derivative of the Hermitian functions defined
    earlier with the central midpoint difference method. Truncation on the
    quartic term.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    """

    x_0 = range_val[0]
    x_f = range_val[-1]
    samples = np.arange(x_0, x_f, stepsize)
    n = len(samples)
    func_vals = []

    for i in range(n):
        term = polynomial(samples[i], hermite_coeffs[level])
        func_vals.append(term)
    func_vals = np.array(func_vals)

    sec_der_vals = []
    for i in range(2, n - 2):
        fpp = (
            -func_vals[i+2]
            + 16 * func_vals[i+1]
            - 30 * func_vals[i]
            + 16 * func_vals[i-1]
            - func_vals[i-2]
        ) / (12 * stepsize**2)
        sec_der_vals.append(fpp)
    sec_der_vals = np.array(sec_der_vals)

    samples_inner = samples[2:-2]
    func_inner = func_vals[2:-2]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals


def analytical_second_derivative(x, coeffs):
    """
    Returns the values computed from the analytical calculation of the
    second derivative of any polynomial.

    Args:
    x  - (list): The array you wish to calculate the derivative at.
    coeffs - (list): The coefficients in increasing order of monomial.
    """
    n = len(coeffs)
    result = 0
    for k in range(2, n):
        result += k * (k - 1) * coeffs[k] * x**(k-2)
    return result

def err_finite_diff(range_stepsizes, num_stepsizes, range_val, level, coeffs, method):
    """
    Compute and plot the FDM's error with respect to the analytical derivative.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    coeffs - (list): The coefficients in increasing order of monomial.
    method: defines if use fourth trunaction (==1) or not (else).
    """
    stepsizes_array = np.linspace(range_stepsizes[0], range_stepsizes[-1], num_stepsizes)
    n = len(stepsizes_array)
    rms_fdm_list = []

    for i in range(n):
        if method == 1:
            x_vals, other, y_vals = finite_difference_fourth(stepsizes_array[i], range_val, level)
        else:
            x_vals, other, y_vals = finite_difference(stepsizes_array[i], range_val, level)

        exact_der_vals = analytical_second_derivative(x_vals, coeffs[level])
        fdm_err = y_vals - exact_der_vals
        rms_fdm = np.sqrt(np.mean(fdm_err ** 2))
        rms_fdm_list.append(rms_fdm)

    rms_fdm_list = np.array(rms_fdm_list)
    mask = rms_fdm_list > 0
    steps = stepsizes_array[mask]
    errors = rms_fdm_list[mask]
    if level >=2:
        slope, c = np.polyfit(np.log(steps), np.log(errors), 1)
        print(f'The slope of the relation is {np.round(slope, decimals=2)}.')

    plt.loglog(steps, errors)
    plt.grid()
    plt.xlabel('stepsize')
    plt.ylabel("RMS FDM")
    plt.title(f"RMS value of FDM with respect to stepsize (H_{level})")
    plt.show()

    return stepsizes_array, rms_fdm_list

def plot_2ndterm(stepsize, range_val, coeffs):
    """
    Plots the results from the first-term calculation for the SHO S.E. obtained
    from the forwards difference method.

    Args:
    coeffs - (list): list of lists of coefficients for each polynomial.
    """
    num_polys = len(coeffs)
    for i in range(num_polys):
        x_vals_i, y_vals_i, other = finite_difference(stepsize, range_val, i)
        plt.plot(x_vals_i, y_vals_i)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("H''(x)")
        plt.title(f'Hermite polynomial H_{i} second derivative')
        plt.show()

def local_energy(stepsize, range_val, level, method):
    """
    Compute the local energy of a specific wavefunction.

    Args:
    stepsize - (float): Stepsize at which to sample the FDM.
    range_val - (list): Range of x-values at which evaluate the wavefunction.
    level - (int): Order of the Hermite Polynomial [0 -> 4].
    method - (bool): Whether to use fourth order truncation (==1) or second (else).
    """
    if method == 1:
        x_vals, y_vals_first, other = finite_difference_fourth(stepsize, range_val, level)
    else:
        x_vals, y_vals_first, other = finite_difference(stepsize, range_val, level)

    y_vals_sec = 0.5 * (x_vals ** 2)
    l_energy = y_vals_first + y_vals_sec
    return x_vals, l_energy

def plot_local_energy(stepsize, range_val, coeffs, method):
    """
    Plot said local energy.

    Args:
    stepsize - (float): Stepsize at which to sample the FDM.
    range_val - (list): Range of x-values at which evaluate the wavefunction.
    level - (int): Order of the Hermite Polynomial [0 -> 4].
    method - (bool): Whether to use fourth order truncation (==1) or second (else).
    """
    num_polys = len(coeffs)
    for i in range(num_polys):
        x_vals_i, y_vals_i = local_energy(stepsize, range_val, i, method)
        plt.plot(x_vals_i, y_vals_i)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("E_l")
        plt.title(f'Local Energy (E_l) for the {i} wavefunction, method {method}')
        plt.show()

## Scripting the first section, we have:

# Numerical evalutation of the local energies with both 2nd and 4th order truncation:
n_hermite = len(hermite_coeffs)

x_vals_second_le = []
local_energy_second_vals = []
for i in range(n_hermite):
    x_vals_le_second_i, local_energy_vals_second_i = local_energy(0.01, [-0.5, 0.5], i, 1)
    x_vals_second_le.append(x_vals_le_second_i)
    local_energy_second_vals.append(local_energy_vals_second_i)

x_vals_second_le = np.array(x_vals_second_le)
local_energy_second_vals = np.array(local_energy_second_vals)

x_vals_fourth_le = []
local_energy_fourth_vals = []
for i in range(n_hermite):
    x_vals_le_fourth_i, local_energy_vals_fourth_i = local_energy(0.01, [-0.5, 0.5], i, 1)
    x_vals_fourth_le.append(x_vals_le_fourth_i)
    local_energy_fourth_vals.append(local_energy_vals_fourth_i)

x_vals_fourth_le = np.array(x_vals_fourth_le)
local_energy_fourth_vals = np.array(local_energy_fourth_vals)

# Plotting of these local energies:
plot_local_energy(0.01, [-0.5, 0.5], hermite_coeffs, 1)
plot_local_energy(0.01, [-0.5, 0.5], hermite_coeffs, 0)

# Calculating the RMS error on these calculations, for 2nd order and 4th order truncations:

stepsize_vals_second = []
err_vals_second = []
for i in range(n_hermite):
    stepsize_vals_second_i, err_vals_second_i = err_finite_diff([1e-5, 1], 500,
                                                [-0.5, 0.5], i, hermite_coeffs, 0)
    stepsize_vals_second.append(stepsize_vals_second_i)
    err_vals_second.append(err_vals_second_i)

stepsize_vals_second = np.array(stepsize_vals_second)
err_vals_second = np.array(err_vals_second)

stepsize_vals_fourth = []
err_vals_fourth = []
for i in range(n_hermite):
    stepsize_vals_fourth_i, err_vals_fourth_i = err_finite_diff([1e-5, 1], 500,
                                                [-0.5, 0.5], i, hermite_coeffs, 1)
    stepsize_vals_fourth.append(stepsize_vals_fourth_i)
    err_vals_fourth.append(err_vals_fourth_i)

stepsize_vals_fourth = np.array(stepsize_vals_fourth)
err_vals_fourth = np.array(err_vals_fourth)
