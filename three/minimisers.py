""""
Module to hold all the function minimising methods.
"""
import numpy as np

def quasi_newton(function, gradient, point, eps, n):
    """
    Find the extrema in a function's domain (3D) by the Quasi-Newton method.

    Args:
    function (func) - The function to extremise (3D).
    gradient (func) - The function's gradient function (3D).
    point (list) - A sensible guess for the algorithm to start from.
    eps (float) - Convergence value for the gradient to be considered null.
    n (int) - Upper bound for the loop.
    """
    G = np.identity(3)

    for _ in range(n):
        grad = gradient(point)
        if np.linalg.norm(grad) < eps:
            break
        p =  - np.linalg.solve(G, grad)
        alpha = backtracking_algorithm(function, gradient, point, p)
        point_new = point + alpha * p

        delta = point_new - point
        gamma = gradient(point_new) - grad

        G_d = G @ delta
        d_G_d = delta @ G_d

        if abs(delta @ gamma) > 1e-10 and d_G_d > 1e-10:
            G = G - np.outer(G_d, G_d) / d_G_d + np.outer(gamma, gamma) / (delta @ gamma)

        point = point_new

    return point

def backtracking_algorithm(function, gradient, point, direction, alpha=1.0, c1=1e-4, c2=0.9, mult=0.5, n=1000):
    """
    Performs a line search for the given step and returns the optimal distance to be
    travelled in that direction.
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
