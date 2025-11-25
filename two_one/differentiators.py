import numpy as np
import matplotlib.pyplot as plt
import two_one.polynomials as poly

def fd_second(stepsize, range_val, samples_num, coeffs):
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
        term = poly.polynomial(samples[i], coeffs)
        func_vals.append(term)
    func_vals = np.array(func_vals)
    sec_der_vals = (func_vals[2:] - 2 * func_vals[1:-1] + func_vals[:-2]) / (stepsize ** 2)
    samples_inner = samples[1:-1]
    func_inner    = func_vals[1:-1]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals

def fd_fourth(stepsize, range_val, samples_num, coeffs):
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
        term = poly.polynomial(samples[i], coeffs)
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

def fd_sixth(stepsize, range_val, samples_num, coeffs):
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
        term = poly.polynomial(samples[i], coeffs)
        func_vals.append(term)
    func_vals = np.array(func_vals)

    sec_der_vals = []
    for i in range(3, n - 3):
        fpp = (
            -2 * func_vals[i+3]
            + 27 * func_vals[i+2]
            - 270 * func_vals[i+1]
            + 490 * func_vals[i]
            - 270 * func_vals[i-1]
            + 27 * func_vals[i-2]
            - 2 * func_vals[i-3]
        ) / (180 * stepsize**2)
        sec_der_vals.append(fpp)
    sec_der_vals = np.array(sec_der_vals)

    samples_inner = samples[3:-3]
    func_inner = func_vals[3:-3]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals

def fd_eighth(stepsize, range_val, samples_num, coeffs):
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
        term = poly.polynomial(samples[i], coeffs)
        func_vals.append(term)
    func_vals = np.array(func_vals)

    sec_der_vals = []
    for i in range(4, n - 4):
        fpp = (
            -9 * func_vals[i+4]
            + 128 * func_vals[i+3]
            - 1008 * func_vals[i+2]
            + 8064 * func_vals[i+1]
            - 14350 * func_vals[i]
            + 8064 * func_vals[i-1]
            - 1008 * func_vals[i-2]
            + 128 * func_vals[i-3]
            - 9 * func_vals[i-4]
        ) / (5040 * stepsize**2)
        sec_der_vals.append(fpp)
    sec_der_vals = np.array(sec_der_vals)

    samples_inner = samples[4:-4]
    func_inner = func_vals[4:-4]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals

def fd_tenth(stepsize, range_val, samples_num, coeffs):
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
        term = poly.polynomial(samples[i], coeffs)
        func_vals.append(term)
    func_vals = np.array(func_vals)

    sec_der_vals = []
    for i in range(5, n - 5):
        fpp = (
            -25 * func_vals[i+5]
            + 405 * func_vals[i+4]
            - 3800 * func_vals[i+3]
            + 33750 * func_vals[i+2]
            - 304005 * func_vals[i+1]
            + 535080 * func_vals[i]
            - 304005 * func_vals[i-1]
            + 33750 * func_vals[i-2]
            - 3800 * func_vals[i-3]
            + 405 * func_vals[i-4]
            - 25 * func_vals[i-5]
        ) / (113400 * stepsize**2)
        sec_der_vals.append(fpp)
    sec_der_vals = np.array(sec_der_vals)

    samples_inner = samples[5:-5]
    func_inner = func_vals[5:-5]
    terms = -0.5 * sec_der_vals / func_inner

    return samples_inner, terms, sec_der_vals


def analytical_second_der(x, coeffs, level, plot):
    """
    Returns the values computed from the analytical calculation of the
    second derivative of any polynomial.

    Args:
    x  - (list): The array you wish to calculate the derivative at.
    coeffs - (list): The coefficients in increasing order of monomial.
    """
    current_coeffs = coeffs[level]
    n = len(current_coeffs)
    func_vals = np.polyval(current_coeffs[::-1], x)
    
    sec_der_vals_exact = np.zeros_like(x, dtype=float)
    for i in range(2, n):
        sec_der_vals_exact += i * (i - 1) * current_coeffs[i] * (x ** (i - 2))
    terms = -0.5 * sec_der_vals_exact / func_vals

    if plot == True:
        plt.loglog(x, terms)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("E_l")
        plt.title(f'Local Energy (E_l) for the {i} wavefunction, analytical method')
        plt.show()

    return terms
