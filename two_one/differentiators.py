"""
Define the finite difference methods (CDM) to calculate the second derivatives
of wavefunctions for the quantum harmonic oscillator and the hydrogen atom.

Two methods are presented: one with uneven sampling, where the function is
called and calculated at each point with a step variation, and one with even
sampling where that is not required. Each method provides truncations up to 10th
order.
"""

import numpy as np
import matplotlib.pyplot as plt

import two_one.polynomials as poly

### The first of these calculate the derivatives by looking at the function
### a given step away from each sample. You therefore need the function to call.

def cdm_step_second(x_vals, wavefunction,  h, coeffs, level):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method. Truncation up to O(h^2).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction (func): The function to be derivated.
    h (float): The step at which to compute the CDM.
    coeffs (list): The coefficients of the polynomial.
    level (list): Order of the Hermite polynomial to be computed.

    Returns:
    np.array: The derivative values at each of the sample points
    """
    func_double_prime = []
    for x in x_vals:
        func_plus = wavefunction(x + h, coeffs[level])
        func_minus = wavefunction(x - h, coeffs[level])
        func = wavefunction(x, coeffs[level])

        func_double_prime_x = (func_plus - 2 * func + func_minus) / (h ** 2)
        func_double_prime.append(func_double_prime_x)

    return np.array(func_double_prime)

def cdm_step_fourth(x_vals, wavefunction,  h, coeffs, level):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method. Truncation up to O(h^4).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction (func): The function to be derivated.
    h (float): The step at which to compute the CDM.
    coeffs (list): The coefficients of the polynomial.
    level (list): Order of the Hermite polynomial to be computed.

    Returns:
    np.array: The derivative values at each of the sample points

    """
    func_double_prime = []
    for x in x_vals:
        func_plus1 = wavefunction(x + h, coeffs[level])
        func_plus2 = wavefunction(x + 2 * h, coeffs[level])
        func_minus1 = wavefunction(x - h, coeffs[level])
        func_minus2 = wavefunction(x - 2 * h, coeffs[level])
        func = wavefunction(x, coeffs[level])

        func_double_prime_x = (-func_plus2 + 16 * func_plus1 - 30 * func +
                               16 * func_minus1 - func_minus2) / (12 * h**2)
        func_double_prime.append(func_double_prime_x)

    return np.array(func_double_prime)

def cdm_step_sixth(x_vals, wavefunction,  h, coeffs, level):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method. Truncation up to O(h^6).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction (func): The function to be derivated.
    h (float): The step at which to compute the CDM.
    coeffs (list): The coefficients of the polynomial.
    level (list): Order of the Hermite polynomial to be computed.

    Returns:
    np.array: The derivative values at each of the sample points
    """
    func_double_prime = []
    for x in x_vals:
        func_plus1 = wavefunction(x + h, coeffs[level])
        func_plus2 = wavefunction(x + 2 * h, coeffs[level])
        func_plus3 = wavefunction(x + 3 * h, coeffs[level])
        func_minus1 = wavefunction(x - h, coeffs[level])
        func_minus2 = wavefunction(x - 2 * h, coeffs[level])
        func_minus3 = wavefunction(x - 3 * h, coeffs[level])
        func = wavefunction(x, coeffs[level])

        func_double_prime_x = (2 * func_plus3 - 27 * func_plus2 + 270 * func_plus1 - 490 * func +
                     270 * func_minus1 - 27 * func_minus2 + 2 * func_minus3) / (180 * h**2)
        func_double_prime.append(func_double_prime_x)

    return np.array(func_double_prime)

def cdm_step_eighth(x_vals, wavefunction,  h, coeffs, level):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method. Truncation up to O(h^8).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction (func): The function to be derivated.
    h (float): The step at which to compute the CDM.
    coeffs (list): The coefficients of the polynomial.
    level (list): Order of the Hermite polynomial to be computed.

    Returns:
    np.array: The derivative values at each of the sample points
    """
    func_double_prime = []
    for x in x_vals:
        func_plus1 = wavefunction(x + h, coeffs[level])
        func_plus2 = wavefunction(x + 2 * h, coeffs[level])
        func_plus3 = wavefunction(x + 3 * h, coeffs[level])
        func_plus4 = wavefunction(x + 4 * h, coeffs[level])
        func_minus1 = wavefunction(x - h, coeffs[level])
        func_minus2 = wavefunction(x - 2 * h, coeffs[level])
        func_minus3 = wavefunction(x - 3 * h, coeffs[level])
        func_minus4 = wavefunction(x - 4 * h, coeffs[level])
        func = wavefunction(x, coeffs[level])

        func_double_prime_x = (-9 * func_plus4 + 128 * func_plus3 - 1008 * func_plus2
                             + 8064 * func_plus1 - 14350 * func + 8064 * func_minus1
                             - 1008 * func_minus2 + 128 * func_minus3
                             - 9 * func_minus4) / (5040 * h**2)

        func_double_prime.append(func_double_prime_x)

    return np.array(func_double_prime)

def cdm_step_tenth(x_vals, wavefunction,  h, coeffs, level):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method. Truncation up to O(h^10).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction (func): The function to be derivated.
    h (float): The step at which to compute the CDM.
    coeffs (list): The coefficients of the polynomial.
    level (list): Order of the Hermite polynomial to be computed.

    Returns:
    np.array: The derivative values at each of the sample points
    """
    func_double_prime = []
    for x in x_vals:
        func_plus1 = wavefunction(x + h, coeffs[level])
        func_plus2 = wavefunction(x + 2 * h, coeffs[level])
        func_plus3 = wavefunction(x + 3 * h, coeffs[level])
        func_plus4 = wavefunction(x + 4 * h, coeffs[level])
        func_plus5 = wavefunction(x + 4 * h, coeffs[level])
        func_minus1 = wavefunction(x - h, coeffs[level])
        func_minus2 = wavefunction(x - 2 * h, coeffs[level])
        func_minus3 = wavefunction(x - 3 * h, coeffs[level])
        func_minus4 = wavefunction(x - 4 * h, coeffs[level])
        func_minus5 = wavefunction(x - 4 * h, coeffs[level])
        func = wavefunction(x, coeffs[level])

        func_double_prime_x = (8 * func_plus5 - 125 * func_plus4 + 1000 * func_plus3
                             - 6000 * func_plus2 + 42000 * func_plus1 - 73766 * func
                             + 42000 * func_minus1 - 6000 * func_minus2 + 1000 * func_minus3
                             - 125 * func_minus4 + 8 * func_minus5) / (25200 * h**2)
        func_double_prime.append(func_double_prime_x)

    return np.array(func_double_prime)

### The second ones take even samplings of any function and evaluate the
### derivatives with respect from one point to its neighbouring ones, so
### there's no need to call the function.

def cdm_samples_second(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=True):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method, with uniform sampling (no need for stepping). Truncation
    up to O(h^2).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction_vals (list): The function evaluated at said points.
    coeffs (list): The coefficients of the polynomial.
    polynonial (bool): Compute the derivative from polynomial values.

    Returns:
    np.array: The final sample points
    np.array: The final function values.
    np.array: The second derivative values at each sample point.
    """
    n = len(x_vals)
    func_vals = []
    if polynomial is True:
        for i in range(n):
            term = poly.polynomial(x_vals[i], coeffs)
            func_vals.append(term)
    else:
        func_vals = wavefunction_vals

    func_vals = np.array(func_vals)

    sec_der_vals = (func_vals[2:] - 2 * func_vals[1:-1] + func_vals[:-2]) / (stepsize ** 2)
    samples_inner = x_vals[1:-1]
    func_inner = func_vals[1:-1]

    return np.array(samples_inner), np.array(func_inner), np.array(sec_der_vals)

def cdm_samples_fourth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=True):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method, with uniform sampling (no need for stepping). Truncation
    up to O(h^4).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction_vals (list): The function evaluated at said points.
    coeffs (list): The coefficients of the polynomial.
    polynonial (bool): Compute the derivative from polynomial values.

    Returns:
    np.array: The final sample points
    np.array: The final function values.
    np.array: The second derivative values at each sample point.
    """
    n = len(x_vals)
    func_vals = []

    if polynomial is True:
        for i in range(n):
            term = poly.polynomial(x_vals[i], coeffs)
            func_vals.append(term)
    else:
        func_vals = wavefunction_vals

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

    samples_inner = x_vals[2:-2]
    func_inner = func_vals[2:-2]

    return np.array(samples_inner), np.array(func_inner), np.array(sec_der_vals)

def cdm_samples_sixth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=True):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method, with uniform sampling (no need for stepping). Truncation
    up to O(h^6).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction_vals (list): The function evaluated at said points.
    coeffs (list): The coefficients of the polynomial.
    polynonial (bool): Compute the derivative from polynomial values.

    Returns:
    np.array: The final sample points
    np.array: The final function values.
    np.array: The second derivative values at each sample point.
    """
    n = len(x_vals)
    func_vals = []

    if polynomial is True:
        for i in range(n):
            term = poly.polynomial(x_vals[i], coeffs)
            func_vals.append(term)
    else:
        func_vals = wavefunction_vals

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

    samples_inner = x_vals[3:-3]
    func_inner = func_vals[3:-3]

    return np.array(samples_inner), np.array(func_inner), np.array(sec_der_vals)

def cdm_samples_eighth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial=True):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method, with uniform sampling (no need for stepping). Truncation
    up to O(h^8).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction_vals (list): The function evaluated at said points.
    coeffs (list): The coefficients of the polynomial.
    polynonial (bool): Compute the derivative from polynomial values.

    Returns:
    np.array: The final sample points
    np.array: The final function values.
    np.array: The second derivative values at each sample point.
    """
    n = len(x_vals)
    func_vals = []

    if polynomial is True:
        for i in range(n):
            term = poly.polynomial(x_vals[i], coeffs)
            func_vals.append(term)
    else:
        func_vals = wavefunction_vals

    func_vals = np.array(func_vals)

    sec_der_vals = []
    for i in range(4, n - 4):
        fpp = (
            -1/560 * func_vals[i+4]
            + 8/315 * func_vals[i+3]
            - 1/5 * func_vals[i+2]
            + 8/5 * func_vals[i+1]
            - 205/72 * func_vals[i]
            + 8/5 * func_vals[i-1]
            - 1/5 * func_vals[i-2]
            + 8/315 * func_vals[i-3]
            - 1/560 * func_vals[i-4]
        ) / (stepsize**2)
        sec_der_vals.append(fpp)
    sec_der_vals = np.array(sec_der_vals)

    samples_inner = x_vals[4:-4]
    func_inner = func_vals[4:-4]

    return np.array(samples_inner), np.array(func_inner), np.array(sec_der_vals)

def cdm_samples_tenth(x_vals, wavefunction_vals, stepsize, coeffs, polynomial):
    """"
    Compute the second derivative of a SHO wavefunction following the Central
    Difference Method, with uniform sampling (no need for stepping). Truncation
    up to O(h^10).

    Args:
    x_vals (list): The points at which to evaluate the function.
    wavefunction_vals (list): The function evaluated at said points.
    coeffs (list): The coefficients of the polynomial.
    polynonial (bool): Compute the derivative from polynomial values.

    Returns:
    np.array: The final sample points
    np.array: The final function values.
    np.array: The second derivative values at each sample point.
    """
    n = len(x_vals)
    func_vals = []

    if polynomial is True:
        for i in range(n):
            term = poly.polynomial(x_vals[i], coeffs)
            func_vals.append(term)
    else:
        func_vals = wavefunction_vals

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

    samples_inner = x_vals[5:-5]
    func_inner = func_vals[5:-5]

    return np.array(samples_inner), np.array(func_inner), np.array(sec_der_vals)

## Finally, there's an analytical method to calculate the derivative of
## any polynomial. Might be useful for Hermite.

def analytical_second_der(x, coeffs, level, plot=False):
    """
    Returns the values computed from the analytical calculation of the
    second derivative of any polynomial.

    Args:
    x (list): The array you wish to calculate the derivative at.
    coeffs (list): The coefficients in increasing order of monomial.
    level (int): The order of the Hermite polynomials.
    plot (bool): Plotting of the results.

    Returns:
    plt.plot: If called, the resulting derivative's plot.
    np.array: The second derivative values at each of the sample points.
    """
    current_coeffs = coeffs[level]
    n = len(current_coeffs)
    func_vals = np.polyval(current_coeffs[::-1], x)

    sec_der_vals_exact = np.zeros_like(x, dtype=float)
    for i in range(2, n):
        sec_der_vals_exact += i * (i - 1) * current_coeffs[i] * (x ** (i - 2))
    terms = -0.5 * sec_der_vals_exact / func_vals

    if plot is True:
        plt.loglog(x, terms)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel("E_l")
        plt.title(f'Local Energy (E_l) for the {i} wavefunction, analytical method')
        plt.show()

    return np.array(terms)

###

if __name__ == "__main__":
    print("Running differentiators.py directly")
