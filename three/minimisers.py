""""
Module to hold all the function minimising methods.
"""
import numpy as np

def quasi_newton(function, gradient, point, eps=10e-6, n=1000):
    """
    Find the extrema in a function's domain (1D) by the Quasi-Newton method.

    Args:
    function (func) - The function to extremise (1D).
    gradient (func) - The function's gradient function (1D).
    point (float) - A sensible guess for the algorithm to start from.
    eps (float) - Convergence value for the gradient to be considered null.
    n (int) - Upper bound for the loop.

    Return:
    float: The optimised parameter.
    """
    g = 1.0

    for _ in range(n):
        grad = gradient(point)
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
