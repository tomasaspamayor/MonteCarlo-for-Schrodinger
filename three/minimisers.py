""""
Module to hold all the function minimising methods.
"""
import numpy as np
import two_two.pdfs as pdfs
import two_two.sampling as samp
import three.hamiltonians as ham

def quasi_newton(function, gradient, point, eps=10e-6, n=1000, track=False):
    """
    Find the extrema in a function's domain (1D) by the Quasi-Newton method.

    Args:
    function (func): The function to extremise (1D).
    gradient (func): The function's gradient function (1D).
    point (float): A sensible guess for the algorithm to start from.
    eps (float): Convergence value for the gradient to be considered null.
    n (int): Upper bound for the loop.
    track (bool): Whether or not to print the values at each iteration.

    Return:
    float: The optimised parameter.
    """
    g = 1.0

    for i in range(n):
        grad = gradient(point)

        if track is True and i % 10 == 0:
            f_val = function(point)
            print(f"  QN iter {i}: point = {point:.6f}, f = {f_val:.6f}, grad = {grad:.6f}")

        if abs(grad) < eps:
            break
        p = - grad / g

        alpha = 1.0
        for _ in range(20):
            point_new = point + alpha * p
            if function(point_new) < function(point) + 1e-4 * alpha * grad * p:
                break
            alpha *= 0.5

        point_new = point + alpha * p

        delta = point_new - point
        gamma = gradient(point_new) - grad

        if abs(delta) > 1e-12:
            g = gamma / delta

        point = point_new

    return point

def quasi_newton_3d(function, gradient, point, eps, n):
    """
    Find the extrema in a function's domain (3D) by the Quasi-Newton method.

    Args:
    function (func) - The function to extremise (3D).
    gradient (func) - The function's gradient function (3D).
    point (list) - A sensible guess for the algorithm to start from.
    eps (float) - Convergence value for the gradient to be considered null.
    n (int) - Upper bound for the loop.

    Return:
    float: The optimised parameter.
    """
    g = np.identity(3)

    for _ in range(n):
        grad = gradient(point)
        if np.linalg.norm(grad) < eps:
            break
        p =  - np.linalg.solve(g, grad)
        alpha = backtracking_algorithm(function, gradient, point, p)
        point_new = point + alpha * p

        delta = point_new - point
        gamma = gradient(point_new) - grad

        g_d = g @ delta
        d_g_d = delta @ g_d

        if abs(delta @ gamma) > 1e-10 and d_g_d > 1e-10:
            g = g - np.outer(g_d, g_d) / d_g_d + np.outer(gamma, gamma) / (delta @ gamma)

        point = point_new

    return point

def backtracking_algorithm(
        function, gradient, point, direction, alpha=1.0, c1=1e-4,
        c2=0.9, mult=0.5, n=1000
    ):
    """
    Performs a line search for the given step and returns the optimal distance to be
    travelled in that direction.

    Args:
    function (func): The function to extremise (3D).
    gradient (func): The function's gradient function (3D).
    point (list): A sensible guess for the algorithm to start from.
    direction (float): Defines the direction of travel.
    alpha (float): Defines the length of the travel.
    c1 (float): Defines the tolerance of the first filtering.
    c2 (float): Defines the tolerance of the second filtering.
    mult (float): The change in step if filtering fails.
    n (int): Number of maximum iterations.

    Return:
    float: The optimised parameter.
    """
    f_val = function(point)
    grad_val = gradient(point)
    direc_der = np.dot(grad_val, direction)

    for _ in range(n):
        point_new = point + alpha * direction
        f_new = function(point_new)
        grad_new = gradient(point_new)

        ceiling = f_val + c1 * alpha * direc_der
        first_cond = f_new <= ceiling

        new_direc_der = np.dot(grad_new, direction)
        second_cond = new_direc_der >= c2 * direc_der

        if first_cond and second_cond:
            return alpha

        if not first_cond:
            alpha *= mult
        else:
            alpha *= (1 + mult)

    return alpha

def hydrogen_wavefunction_optimiser(theta, domain=np.array([[-4, 4], [-4, 4], [-4, 4]]), method=False, stepsize=0.05, num_samples=10000, m=50, eps=1e-5):
    """"
    Optimise the wavefunction by variating the parameter theta. Used the Quasi-Newton Algorithm.

    Args:
    theta (float): Wavefunction parameter guess. Defines the startpoint of the algorithm.
    m (int): Maximum number of iterations until the optimal theta is reached.
    eps (float): Convergence requirement.
    domain (list): Range of each dimension, start and end point.
    sampling_method (bool): If True, uses the Rejection, if False uses Metropolis-Hastings.
    num_samples (int): Required number of samples for the respective sampling method.

    Return:
    float: The optimised parameter
    np.array: The list of all parameter iteration values
    np.array: The list of all energy iteration values
    plt.plots: Plots throughout of the progress
    """
    theta_current = theta
    theta_values = []
    energy_values = []

    def current_pdf(x):
        return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_current)

    for l in range(m):
        if method is False:
            x_points = samp.rejection_3d(current_pdf, 0, domain, num_samples,
                                         num_samples*10000, m=None)
        else:
            x_points = samp.metropolis_hastings_3d(current_pdf, 0, domain, stepsize, num_samples)

        samp.plot_3d_samples(x_points, 100, 1)

        current_energy = ham.energy_expectation(x_points, theta_current)
        energy_values.append(current_energy)

        def energy_wrapped(theta):
            def current_pdf(x):
                return pdfs.wavefunction_hydrogen_atom_pdf(x, theta)
            if method is False:
                x_energy_points = samp.rejection_3d(current_pdf, 0, domain, num_samples,
                                                    num_samples*10000, m=None)
            else:
                x_energy_points = samp.metropolis_hastings_3d(current_pdf, 0,
                                                              domain, stepsize, num_samples)

            return ham.energy_expectation(x_energy_points, theta)
        def gradient_wrapped(theta):
            def current_pdf(x):
                return pdfs.wavefunction_hydrogen_atom_pdf(x, theta)
            if method is False:
                x_gradient_points = samp.rejection_3d(current_pdf, 0, domain,
                                                      num_samples, num_samples*10000, m=None)
            else:
                x_gradient_points = samp.metropolis_hastings_3d(current_pdf, 0,
                                                                domain, stepsize, num_samples)

            return ham.energy_expectation_theta_derivative(x_gradient_points, theta)

        theta_opt = quasi_newton(energy_wrapped, gradient_wrapped,
                                            theta_current, 1e-7, 10000, track=True)
        theta_values.append(theta_opt)
        theta_current = theta_opt

        if l > 1 and np.abs(theta_values[-1] - theta_values[-2]) < eps:
            print(f"Final theta = {theta_values[-1]:.6f}, Final energy = {energy_values[-1]:.6f}")
            break

    def final_pdf(x):
        return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_values[-1])

    if method is False:
        x_points = samp.rejection_3d(final_pdf, 0, domain, num_samples,
                                        num_samples*10000, m=None)
    else:
        x_points = samp.metropolis_hastings_3d(final_pdf, 0, domain, stepsize, num_samples)

    samp.plot_3d_samples(x_points, 100, 1)

    return theta_values[-1], theta_values, energy_values
