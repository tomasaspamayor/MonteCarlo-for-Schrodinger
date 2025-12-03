""""
Module to hold all the function minimising methods.
"""
import numpy as np
import two_two.pdfs as pdfs
import two_two.sampling as samp
import three.hamiltonians as ham
import two_one.local_energy as le

## Minimiser Methods:

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

        if track is True:
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

def gradient_descent(gradient, point, alpha=0.01, eps=1e-7, m=1000, track=True):
    """
    Find the extrema of a function in its domain (3D) by the Gradient Descent
    method.

    Args:
    gradient (callable): Gradient function.
    point (list): The starting point.
    alpha (float): Learning rate.
    eps (float): Convergence factor.
    m (int): Cap for iterations.
    track (bool): Whether to print out the statements.

    """
    points_history = []
    gradients_history = []
    converged = False
    for i in range(m):
        grad = gradient(point)
        points_history.append(point)
        gradients_history.append(grad)
        if track is True:
            print(f'Iteration {i}: point = {point}, grad = {grad}')
        if abs(grad) < eps:
            print(f'Method has converged at iteration {i}, point = {point}, grad = {grad}')
            converged = True
            break
        point = point - alpha * grad

    if not converged and track:
        print(f'Did not converge after {m} iterations')

    return point, grad, converged, points_history, gradients_history

def gradient_descent_3d(gradient, point, alpha=0.01, eps=1e-7, m=1000, track=True):
    """
    Find the extrema of a function in its domain (3D) by the Gradient Descent
    method.

    Args:
    gradient (callable): Gradient function.
    point (list): The starting point.
    alpha (float): Learning rate.
    eps (float): Convergence factor.
    m (int): Cap for iterations.
    track (bool): Whether to print out the statements.

    """
    point = np.array(point, dtype=float)

    points_history = []
    gradients_history = []
    converged = False

    for i in range(m):
        grad = np.array(gradient(point), dtype=float)
        grad_norm = np.linalg.norm(grad)
        points_history.append(point)
        gradients_history.append(grad)

        if track is True:
            print(f'Iter. {i}: point = {point}, grad = {grad}, grad_norm = {grad_norm}')
        if grad_norm < eps:
            print(f'Method has converged at iteration {i}, point = {point}, grad = {grad}')
            converged = True
            break
        point = point - alpha * grad

    if not converged:
        print(f'Did not converge after {m} iterations')

    return point, grad, converged, points_history, gradients_history

## Optimising Methods:

# Hydrogen Atom Methods:

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

def hydrogen_wavefunction_optimiser_gd(theta, domain=np.array([[-4, 4], [-4, 4], [-4, 4]]),
                                       method=False, stepsize=0.05, num_samples=10000,
                                       m=50, eps=1e-5, learning_rate=0.1, track_progress=True):
    """
    Optimise Hydrogen wavefunction parameter theta using Gradient Descent.
    
    Args:
        theta: Initial wavefunction parameter
        domain: 3D sampling domain
        method: False=Rejection, True=Metropolis-Hastings
        stepsize: Step size for Metropolis-Hastings
        num_samples: Number of samples per iteration
        m: Maximum iterations
        eps: Convergence tolerance
        learning_rate: Gradient descent learning rate
        track_progress: Print progress information
        
    Returns:
        tuple: (optimized_theta, theta_history, energy_history)
    """
    theta_current = theta
    theta_values = [theta]
    energy_values = []
    
    for iteration in range(m):
        def current_pdf(x):
            return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_current)

        if method is False:
            x_points = samp.rejection_3d(current_pdf, 0, domain, num_samples,
                                         num_samples*10000, m=None)
        else:
            x_points = samp.metropolis_hastings_3d(current_pdf, 0, domain, 
                                                   stepsize, num_samples)

        if track_progress and iteration % 10 == 0:
            samp.plot_3d_samples(x_points, 50, 1)
        
        current_energy = ham.energy_expectation(x_points, theta_current)
        current_gradient = ham.energy_expectation_theta_derivative(x_points, theta_current)

        energy_values.append(current_energy)

        if track_progress:
            print(f"Iter {iteration}: θ = {theta_current:.6f}, "
                  f"E = {current_energy:.6f}, ∇ = {current_gradient:.6f}")

        # Gradient descent update
        theta_new = theta_current - learning_rate * current_gradient
        theta_new = max(0.1, min(2.0, theta_new))
        theta_values.append(theta_new)
        theta_current = theta_new

        # Check convergence
        if iteration > 0 and abs(theta_values[-1] - theta_values[-2]) < eps:
            if track_progress:
                print(f"Converged after {iteration} iterations")
                print(f"Final θ = {theta_current:.6f}, E = {current_energy:.6f}")
            break

    # Final sampling with optimized theta
    def final_pdf(x):
        return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_values[-1])
    
    if method is False:
        final_points = samp.rejection_3d(final_pdf, 0, domain, num_samples,
                                         num_samples*10000, m=None)
    else:
        final_points = samp.metropolis_hastings_3d(final_pdf, 0, domain, 
                                                   stepsize, num_samples)

    samp.plot_3d_samples(final_points, 100, 1)
    final_energy = ham.energy_expectation(final_points, theta_values[-1])

    if track_progress:
        print("Optimization complete!")
        print(f"Final θ = {theta_values[-1]:.6f}")
        print(f"Final E = {final_energy:.6f}")
        print("Expected: θ ≈ 1.0, E ≈ -0.5")

    return theta_values[-1], theta_values, energy_values

# Hydrogen Molecule Methods:

def h2_optimiser_gd(theta, start, stepsize,  bond_length, delta, num_samples, alpha, m, eps, domain=np.array([[-3, 3], [-3, 3], [-3, 3]])):
    """
    Optimise the hydrogen molecule theta by the gradient descent method.
    """
    theta_current = np.array(theta, dtype=float)
    theta_vals = [theta_current]
    e_vals= []

    q1 = np.array([0, 0, - bond_length / 2])
    q2 = np.array([0, 0, bond_length / 2])

    domain_list = domain.tolist()
    dom_6d = domain_list + domain_list

    for i in range(m):

        def current_pdf(r):
            r1 = r[:3]
            r2 = r[3:]
            wf_vals = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta_current, q1, q2)
            return wf_vals ** 2

        samples = samp.metropolis_hastings_3d(current_pdf, start, dom_6d, stepsize, num_samples, dimensions=6)

        if i % 10 == 0:
            samp.plot_6d_samples(samples, bond_length)

        energies = []
        for sample in samples:
            r1 = sample[:3]
            r2 = sample[3:]
            E_local = le.h2_local_energy(r1, r2, theta_current, q1, q2)
            energies.append(E_local)
        
        current_energy = np.mean(energies)
        e_vals.append(current_energy)
        theta_vals.append(theta_current)

        grad = np.zeros_like(theta_current)
        d = len(theta_current)

        for j in range(d):
            theta_plus = theta_current.copy()
            theta_minus = theta_current.copy()
            theta_plus[j] += delta
            theta_minus[j] -= delta

            energies_plus = []
            energies_minus = []
            for sample in samples:
                r1 = sample[:3]
                r2 = sample[3:]
                e_perturbed_plus = le.h2_local_energy(r1, r2, theta_plus,
                                                        q1, q2)
                e_perturbed_minus = le.h2_local_energy(r1, r2, theta_minus,
                                                        q1, q2)
                energies_plus.append(e_perturbed_plus)
                energies_minus.append(e_perturbed_minus)

            energy_plus = np.mean(energies_plus)
            energy_minus = np.mean(energies_minus)

            grad[j] = (energy_plus - energy_minus) / (2 * delta)

        grad_norm = np.linalg.norm(grad)

        print(f"Iteration {i}: R = {theta_current}, E = {current_energy}, grad = {grad}, grad norm = {grad_norm}")

        if grad_norm < eps:
            print(f'Convergence has been reached: Theta values: {theta}, Gradient = {grad}, Gradient Norm = {grad_norm}')
            break

        theta_current = theta_current - alpha * grad

    minimum_index = np.argmin(e_vals)
    theta_optimised = theta_vals[minimum_index]
    e_optimised = e_vals[minimum_index]

    print(f'The optimal theta is {theta_optimised}, with energy {e_optimised}.')

    return theta_optimised, e_optimised, theta_vals, e_vals
