#
#%% COMPUTATIONAL PHYSICS: PROJECT CODE - Tomàs Aspa Mayor
# Variational Monte Carlo Methods to solve the Schröndinger Equation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 2.1 - Finite Difference Method to find the Local Energy of SHO.

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

def finite_difference(stepsize, range_val, samples_num, level):
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
    samples = np.linspace(x_0, x_f, samples_num)
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

def finite_difference_fourth(stepsize, range_val, samples_num, level):
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
    samples = np.linspace(x_0, x_f, samples_num)
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


def term_analytical_second_derivative(x, coeffs, level):
    """
    Returns the values computed from the analytical calculation of the
    second derivative of any polynomial.

    Args:
    x  - (list): The array you wish to calculate the derivative at.
    coeffs - (list): The coefficients in increasing order of monomial.
    """
    n = len(coeffs)
    func_vals = np.polyval(coeffs[::-1], x)
    
    sec_der_vals_exact = np.zeros_like(x, dtype=float)
    for i in range(2, n):
        sec_der_vals_exact += i * (i - 1) * coeffs[i] * (x ** (i - 2))
    terms = -0.5 * sec_der_vals_exact / func_vals
    return terms

def plot_analytical_vals(x, coeffs, level):
    vals = term_analytical_second_derivative(x, coeffs, level)

    plt.loglog(x, vals)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel("E_l")
    plt.title(f'Local Energy (E_l) for the {i} wavefunction, analytical method')
    plt.show()

def err_finite_diff(range_stepsizes, num_stepsizes, range_val, samples_num, level, coeffs, method):
    """
    Compute and plot the FDM's error with respect to the analytical derivative.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    coeffs - (list): The coefficients in increasing order of monomial.
    method: defines if use fourth trunaction (==1) or not (else).
    """
    rms_fdm_hermites = []
    h = len(coeffs)
    for i in range(h):
        stepsizes_array = np.linspace(range_stepsizes[0], range_stepsizes[-1], num_stepsizes)
        n = len(stepsizes_array)

        samples = np.linspace(range_val[0], range_val[-1], samples_num)
        current_coeffs = coeffs[i]
        rms_fdm_list = []

        for j in range(n):
            sec_exact = term_analytical_second_derivative(samples, current_coeffs, level)
            if method == 1:
                x_vals, sec_fd, other = finite_difference_fourth(stepsizes_array[j], range_val, samples_num, i)
                sec_exact = sec_exact[2:-2]
            else:
                x_vals, sec_fd, other = finite_difference(stepsizes_array[j], range_val, samples_num, i)
                sec_exact = sec_exact[1:-1]

            rms = np.sqrt(np.mean((sec_fd - sec_exact) ** 2))
            rms_fdm_list.append(rms)

        rms_fdm = np.array(rms_fdm_list)
        rms_fdm_hermites.append(rms_fdm)

    plt.figure(figsize=(10, 6))
    for poly_order in range(h):
        plt.loglog(stepsizes_array, rms_fdm_hermites[poly_order], 
                  label=f'H_{poly_order}', marker='o', markersize=3)
    plt.grid()
    plt.xlabel('stepsize')
    plt.ylabel("RMS FDM")
    plt.title(f"RMS value of FDM with respect to stepsize")
    plt.show()

    return stepsizes_array, rms_fdm_list

def plot_2ndterm(stepsize, range_val, samples_num, coeffs):
    """
    Plots the results from the first-term calculation for the SHO S.E. obtained
    from the forwards difference method.

    Args:
    coeffs - (list): list of lists of coefficients for each polynomial.
    """
    num_polys = len(coeffs)
    for i in range(num_polys):
        x_vals_i, y_vals_i, other = finite_difference(stepsize, range_val, samples_num, i)
        plt.plot(x_vals_i, y_vals_i)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("H''(x)")
        plt.title(f'Hermite polynomial H_{i} second derivative')
        plt.show()

def local_energy(stepsize, range_val, samples_num, level, method):
    """
    Compute the local energy of a specific wavefunction.

    Args:
    stepsize - (float): Stepsize at which to sample the FDM.
    range_val - (list): Range of x-values at which evaluate the wavefunction.
    level - (int): Order of the Hermite Polynomial [0 -> 4].
    method - (bool): Whether to use fourth order truncation (==1) or second (else).
    """
    if method == 1:
        x_vals, y_vals_first, other = finite_difference_fourth(stepsize, range_val, samples_num, level)
    else:
        x_vals, y_vals_first, other = finite_difference(stepsize, range_val, samples_num, level)

    y_vals_sec = 0.5 * (x_vals ** 2)
    l_energy = y_vals_first + y_vals_sec
    return x_vals, l_energy

def plot_local_energy(stepsize, range_val, samples_num, coeffs, method):
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
        x_vals_i, y_vals_i = local_energy(stepsize, range_val, samples_num, i, method)
        plt.plot(x_vals_i, y_vals_i)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("E_l")
        plt.title(f'Local Energy (E_l) for the {i} wavefunction, method {method}')
        plt.show()

#%% Scripting the first section, we have:

# Numerical evalutation of the local energies with both 2nd and 4th order truncation:
n_hermite = len(hermite_coeffs)

x_vals_second_le = []
local_energy_second_vals = []
for i in range(n_hermite):
    x_vals_le_second_i, local_energy_vals_second_i = local_energy(0.01, [-1, 1], 1000, i, 1)
    x_vals_second_le.append(x_vals_le_second_i)
    local_energy_second_vals.append(local_energy_vals_second_i)

x_vals_second_le = np.array(x_vals_second_le)
local_energy_second_vals = np.array(local_energy_second_vals)

x_vals_fourth_le = []
local_energy_fourth_vals = []
for i in range(n_hermite):
    x_vals_le_fourth_i, local_energy_vals_fourth_i = local_energy(0.01, [-1, 1], 1000, i, 1)
    x_vals_fourth_le.append(x_vals_le_fourth_i)
    local_energy_fourth_vals.append(local_energy_vals_fourth_i)

x_vals_fourth_le = np.array(x_vals_fourth_le)
local_energy_fourth_vals = np.array(local_energy_fourth_vals)

# Plotting of these local energies:
plot_local_energy(0.01, [-1, 1], 1000, hermite_coeffs, 1)
plot_local_energy(0.01, [-1, 1], 1000, hermite_coeffs, 0)

# Calculating the RMS error on these calculations, for 2nd order and 4th order truncations:

stepsize_vals_second = []
err_vals_second = []
for i in range(n_hermite):
    stepsize_vals_second_i, err_vals_second_i = err_finite_diff([1e-5, 0.01], 500,
                                                [-1, 1], 1000, i, hermite_coeffs, 0)
    stepsize_vals_second.append(stepsize_vals_second_i)
    err_vals_second.append(err_vals_second_i)

stepsize_vals_second = np.array(stepsize_vals_second)
err_vals_second = np.array(err_vals_second)

stepsize_vals_fourth = []
err_vals_fourth = []
for i in range(n_hermite):
    stepsize_vals_fourth_i, err_vals_fourth_i = err_finite_diff([1e-5, 0.01], 500,
                                                [-1, 1], 1000, i, hermite_coeffs, 1)
    stepsize_vals_fourth.append(stepsize_vals_fourth_i)
    err_vals_fourth.append(err_vals_fourth_i)

stepsize_vals_fourth = np.array(stepsize_vals_fourth)
err_vals_fourth = np.array(err_vals_fourth)

#%% 2.2 - Energy Values Verification

## There's different methods from the lectures we could use. My choice will be the
## Rejection Method and the Metropolis-Hastings Algorithm. We will start with the 
## former.

def rejection_sampling(PDF, CF, start, end, num_samples, N, constant=1, M=10):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    start (float) - Startpoint of the sample array.
    end (float) - Endpoint of the sample array.
    num_samples (int) - The desired number of points in the final sample.
    N (int) - Upper bound for the sampling loop.
    constant (bool) - Option to use a constant CF.
    M (float) - Scaling of the CF. Only used in constant CFs.
    """

    # Calculate M automatically if not provided:
    if constant == 1 and M is None:
        x_trial = np.linspace(start, end, 1000)
        pdf_max = np.max(PDF(x_trial))
        q = 1 / (end - start)
        M = pdf_max / q * 1.1 # Buffering.
        print(f"Calculated M {M}")

    if constant == 1:
        # Generate the simplest proposal function, a constant:
        q = 1 / (end - start)
        constant_cf = M * q

    # Create an empty list for the distribution:
    distribution = []

    for i in range(N):
        x_val = np.random.uniform(start, end)
        pdf_val = PDF(x_val)

        if constant == 1:
            cf_val = constant_cf
        else:
            cf_val = CF(x_val)

        if pdf_val > cf_val:
            raise ValueError('The CF must enclose the PDF for all values (repick M).')

        u = np.random.uniform(0, 1)
        if u < (pdf_val / cf_val):
            distribution.append(x_val)

        if len(distribution) == num_samples:
            break

    samples = np.array(distribution)

    if len(samples) < num_samples:
        raise ValueError('Not all samples were generated.')

    return samples


def rejection_sampling_3D(PDF, CF, start, end, num_samples, N, constant=1, M=10):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    start (list) - Startpoint for each dimension.
    end (float) - Endpoint for each dimension.
    num_samples (int) - The desired number of points in the final sample.
    N (int) - Upper bound for the sampling loop.
    constant (bool) - Option to use a constant CF.
    M (float) - Scaling of the CF. Only used in constant CFs.
    """

    x_start, x_end = start[0], end[0]
    y_start, y_end = start[1], end[1]
    z_start, z_end = start[2], end[2]

    if constant == 1:
        # Generate the simplest proposal function, a constant:
        q = 1 / ((x_end - x_start) * (y_end - y_start) * (z_end - z_start))
        constant_cf = M * q

    # Create an empty list for the distribution:
    distribution = []

    for i in range(N):
        x_val = np.random.uniform(x_start, x_end)
        y_val = np.random.uniform(y_start, y_end)
        z_val = np.random.uniform(z_start, z_end)

        pdf_val = PDF(x_val, y_val, z_val)

        if constant == 1:
            cf_val = constant_cf
        else:
            cf_val = CF(x_val, y_val, z_val)

        if pdf_val > cf_val:
            raise ValueError('The CF must enclose the PDF for all values (repick M)')

        u = np.random.uniform(0, 1)
        if u < (pdf_val / cf_val):
            distribution.append([x_val, y_val, z_val])

        if len(distribution) == num_samples:
            break

    samples = np.array(distribution)

    if len(samples) < num_samples:
        raise ValueError('Not all samples were generated.')

    return samples

# Testing the rejection algorithm:

# Normalized Gaussian on [-1, 1]
def normalized_gaussian(x, sigma=0.2):
    """Gaussian normalized over [-1, 1]"""
    # Compute normalization factor
    norm_factor = norm.cdf(1, scale=sigma) - norm.cdf(-1, scale=sigma)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2)) / norm_factor

# Normalized exponential decay on [0, 1]
def normalized_exponential(x, lambd=2.0):
    """Exponential decay normalized over [0, 1]"""
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    # CORRECTED normalization factor
    norm_factor = 1 - np.exp(-lambd * 1)  # This was the error!
    return (lambd * np.exp(-lambd * x)) / norm_factor

# Verify the normalization
print("Verifying normalization:")
x_test = np.linspace(-1, 1, 10000)
gauss_integral = np.trapezoid(normalized_gaussian(x_test), x_test)
print(f"Gaussian integral over [-1,1]: {gauss_integral}")

x_test_exp = np.linspace(0, 1, 10000)
exp_integral = np.trapezoid(normalized_exponential(x_test_exp), x_test_exp)
print(f"Exponential integral over [0,1]: {exp_integral}")

samples_gaussian = rejection_sampling(normalized_gaussian, 1, -1, 1, 1000, 10000)
samples_exponential = rejection_sampling(normalized_exponential, 1, 0, 1, 1000, 10000)

x_gauss_vals = np.linspace(-1, 1, 1000)
x_expo_vals = np.linspace(0, 1, 1000)
gauss_vals = normalized_gaussian(x_gauss_vals)
expo_vals = normalized_exponential(x_expo_vals)

plt.hist(samples_gaussian, 50, density=True, label='Generated Samples')
plt.plot(x_gauss_vals, gauss_vals, label='PDF')
plt.title('Gaussian PDF Sampling')
plt.legend()
plt.show()

plt.hist(samples_exponential, 50, density=True, label='Generated Samples')
plt.plot(x_expo_vals, expo_vals, label='PDF')
plt.title('Exponential PDF Sampling')
plt.legend()
plt.show()

# %%

















