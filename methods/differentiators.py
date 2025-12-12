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

import methods.polynomials as poly
from methods import pdfs

plt.style.use('seaborn-v0_8-paper')

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

## Analytical method to calculate the derivative of any polynomial.

def analytical_second_derivative_qho(n, x):
    """
    Here.
    """
    coeffs = [
    [1],
    [0, 2],
    [-2, 0, 4],
    [0, -12, 0, 8],
    [12, 0, -48, 0, 16],
    [0, 120, 0, -160, 0, 32],
    [-120, 0, 720, 0, -480, 0, 64]  ]

    coeff = coeffs[n]
    H = np.polyval(coeff[::-1], x)
    H_prime = np.polyval(np.polyder(coeff[::-1]), x)
    H_double_prime = np.polyval(np.polyder(coeff[::-1], 2), x)
    exponential = np.exp(-x**2 / 2)
    return exponential * (H_double_prime - 2*x*H_prime + (x**2 - 1)*H)
## CDM to solve a Laplacian.

def cdm_laplacian(func, point, step=0.01):
    """
    2nd-order central difference Laplacian.

    Args:
    func (callable): The function to be differentiated.
    point (list): Point at which to compute the Laplacian (3D).
    step (float): The step at which to compute the finite difference.

    Returns:
    float: Laplacian value.
    """
    x, y, z = point
    f = func(point)

    # x-direction
    f_x1  = func([x + step, y, z])
    f_x_1 = func([x - step, y, z])
    fxx = (f_x1 - 2*f + f_x_1) / (step**2)

    # y-direction
    f_y1  = func([x, y + step, z])
    f_y_1 = func([x, y - step, z])
    fyy = (f_y1 - 2*f + f_y_1) / (step**2)

    # z-direction
    f_z1  = func([x, y, z + step])
    f_z_1 = func([x, y, z - step])
    fzz = (f_z1 - 2*f + f_z_1) / (step**2)

    return fxx + fyy + fzz

def cdm_laplacian_4th(func, point, theta, step=0.01):
    """
    4th-order central difference Laplacian. More stable than 8th-order.

    Args:
    func (callable): The function to be differentiated.
    point (list): Point at which to compute the Laplacian (3D).
    step (float): The step at which to compute the finite difference.

    Returns:
    float: Laplacian value.
    """
    x, y, z = point
    f = func(point, theta)

    # x-direction
    f_x2 = func([x + 2*step, y, z], theta)
    f_x1 = func([x + step, y, z], theta)
    f_x_1 = func([x - step, y, z], theta)
    f_x_2 = func([x - 2*step, y, z], theta)

    fxx = (
        -f_x2 + 16*f_x1 - 30*f + 16*f_x_1 - f_x_2
    ) / (12 * step**2)

    # y-direction
    f_y2 = func([x, y + 2*step, z], theta)
    f_y1 = func([x, y + step, z], theta)
    f_y_1 = func([x, y - step, z], theta)
    f_y_2 = func([x, y - 2*step, z], theta)

    fyy = (
        -f_y2 + 16*f_y1 - 30*f + 16*f_y_1 - f_y_2
    ) / (12 * step**2)

    # z-direction
    f_z2 = func([x, y, z + 2*step], theta)
    f_z1 = func([x, y, z + step], theta)
    f_z_1 = func([x, y, z - step], theta)
    f_z_2 = func([x, y, z - 2*step], theta)

    fzz = (
        -f_z2 + 16*f_z1 - 30*f + 16*f_z_1 - f_z_2
    ) / (12 * step**2)

    return fxx + fyy + fzz

def cdm_laplacian_8th(func, point, theta, step=0.01):
    """
    4th-order central difference Laplacian. More stable than 8th-order.

    Args:
    func (callable): The function to be differentiated.
    point (list): Point at which to compute the Laplacian (3D).
    step (float): The step at which to compute the finite difference.

    Returns:
    float: Laplacian value.
    """
    x, y, z = point
    f = func(point, theta)

    # x-direction
    f_x4 = func([x + 4*step, y, z], theta)
    f_x3 = func([x + 3*step, y, z], theta)
    f_x2 = func([x + 2*step, y, z], theta)
    f_x1 = func([x + step, y, z], theta)
    f_x_1 = func([x - step, y, z], theta)
    f_x_2 = func([x - 2*step, y, z], theta)
    f_x_3 = func([x - 3*step, y, z], theta)
    f_x_4 = func([x - 4*step, y, z], theta)

    fxx = (
        -f_x4 + 32*f_x3 - 168*f_x2 + 672*f_x1 - 1260*f + 
        672*f_x_1 - 168*f_x_2 + 32*f_x_3 - f_x_4
    ) / (840 * step**2)

    # y-direction
    f_y4 = func([x, y + 4*step, z], theta)
    f_y3 = func([x, y + 3*step, z], theta)
    f_y2 = func([x, y + 2*step, z], theta)
    f_y1 = func([x, y + step, z], theta)
    f_y_1 = func([x, y - step, z], theta)
    f_y_2 = func([x, y - 2*step, z], theta)
    f_y_3 = func([x, y - 3*step, z], theta)
    f_y_4 = func([x, y - 4*step, z], theta)

    fyy = (
        -f_y4 + 32*f_y3 - 168*f_y2 + 672*f_y1 - 1260*f + 
        672*f_y_1 - 168*f_y_2 + 32*f_y_3 - f_y_4
    ) / (840 * step**2)

    # z-direction
    f_z4 = func([x, y, z + 4*step], theta)
    f_z3 = func([x, y, z + 3*step], theta)
    f_z2 = func([x, y, z + 2*step], theta)
    f_z1 = func([x, y, z + step], theta)
    f_z_1 = func([x, y, z - step], theta)
    f_z_2 = func([x, y, z - 2*step], theta)
    f_z_3 = func([x, y, z - 3*step], theta)
    f_z_4 = func([x, y, z - 4*step], theta)

    fzz = (
        -f_z4 + 32*f_z3 - 168*f_z2 + 672*f_z1 - 1260*f + 
        672*f_z_1 - 168*f_z_2 + 32*f_z_3 - f_z_4
    ) / (840 * step**2)

    return fxx + fyy + fzz

def laplacian_analytical(theta, point):
    """
    Computes the analytical value of the hydrogen's wavefunction Laplacian.
    """
    r = np.linalg.norm(point)
    
    wavefunction_value = np.exp(-theta * r)
    laplacian_value = wavefunction_value * (theta**2 - 2 * theta / r)
    
    return laplacian_value

def laplacian_comparison(func, theta, point, method=1, step=0.01):
    """
    Comparison of the analytical and numerical Laplacian method.
    """
    if method == 8:
        vals_ap = cdm_laplacian_8th(func, point, theta, step)
    else:
        vals_ap = cdm_laplacian_4th(func, point, theta, step)

    vals_an = laplacian_analytical(theta, point)
    return vals_ap, vals_an

def lap_comp_plot(theta, bounds=np.array([-3,3]), step=0.01, method=1):
    """
    Plot the relation between the numerical and analytical Laplacian.
    """
    check_x_coords = np.linspace(bounds[0], bounds[1], 1000)
    check_points = np.zeros((1000, 3))
    check_points[:, 0] = check_x_coords

    numerical_laplacians = []
    analytical_laplacians = []
    radial_distances = []

    for point in check_points:
        r = np.linalg.norm(point)
        if r == 0:
            continue
        num_lap, analy_lap = laplacian_comparison(
            pdfs.wavefunction_hydrogen_atom, theta, point, method, step)
        numerical_laplacians.append(num_lap)
        analytical_laplacians.append(analy_lap)
        radial_distances.append(r)

    numerical_laplacians = np.array(numerical_laplacians)
    analytical_laplacians = np.array(analytical_laplacians)
    radial_distances = np.array(radial_distances)

    absolute_error = np.abs(numerical_laplacians - analytical_laplacians)

    plt.figure(figsize=(10, 6))
    plt.loglog(radial_distances, absolute_error, 'b.', markersize=5)
    plt.title(f'Absolute Error of Numerical Laplacian vs. Radial Distance (theta={theta}, h={step})', 
            fontsize=14, fontweight='bold')
    plt.xlabel('Radial Distance (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Absolute Error (log scale)', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')
    plt.show()
