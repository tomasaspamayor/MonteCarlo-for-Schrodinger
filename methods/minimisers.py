""""
Module to hold all the function minimising methods.
"""
import numpy as np
import matplotlib.pyplot as plt

import methods.pdfs as pdfs
import methods.sampling as samp
import methods.hamiltonians as ham
import methods.local_energy as le

plt.style.use('seaborn-v0_8-paper')

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
    """
    theta_current = theta
    theta_values = [theta]
    energy_values = []
    grad_values = []
    
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
        
        # Get energy and gradient
        current_energy, _ = ham.energy_expectation(x_points, theta_current)  # Unpack tuple
        current_gradient = ham.energy_expectation_theta_derivative(x_points, theta_current)

        energy_values.append(current_energy)
        grad_values.append(current_gradient)

        if track_progress:
            print(f"Iter {iteration}: θ = {theta_current:.6f}, "
                  f"E = {current_energy:.6f}, ∇ = {current_gradient:.6f}")

        # Gradient descent update
        theta_new = theta_current - learning_rate * current_gradient
        # Optional: clip theta to reasonable range
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
    final_energy, _ = ham.energy_expectation(final_points, theta_values[-1])

    if track_progress:
        print("Optimization complete!")
        print(f"Final θ = {theta_values[-1]:.6f}")
        print(f"Final E = {final_energy:.6f}")
        print("Expected: θ ≈ 1.0, E ≈ -0.5")
    
    iterations = np.arange(len(energy_values))

    return iterations, theta_values[-1], theta_values, grad_values, energy_values

def hydrogen_wavefunction_optimiser_gd_num(theta, step, h, domain=np.array([[-4, 4], [-4, 4], [-4, 4]]),
                                           method=False, stepsize=0.05, num_samples=100000,
                                           m=50, eps=1e-5, learning_rate=0.1, track_progress=True):
    """
    Optimise Hydrogen wavefunction parameter theta using Gradient Descent.
    """
    theta_current = theta
    theta_values = [theta]
    energy_values = []
    grad_values = []
    
    # Use the SAME samples for both energy and gradient calculations
    # to reduce noise in finite differences
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
        
        # Use the SAME samples for all calculations to reduce noise
        current_energy = ham.energy_expectation_num(x_points, theta_current, step)
        
        # Use the SAME x_points for gradient calculation (CRITICAL!)
        current_gradient = ham.energy_expectation_theta_derivative_num(
            x_points, theta_current, h, step
        )

        energy_values.append(current_energy)
        grad_values.append(current_gradient)

        if track_progress:
            print(f"Iter {iteration}: θ = {theta_current:.6f}, "
                  f"E = {current_energy:.6f}, ∇ = {current_gradient:.6f}")

        # Gradient descent update
        theta_new = theta_current - learning_rate * current_gradient
        theta_new = max(0.1, min(2.0, theta_new))
        theta_values.append(theta_new)
        theta_current = theta_new

        # Check convergence - consider both theta change and gradient magnitude
        if iteration > 0:
            theta_change = abs(theta_values[-1] - theta_values[-2])
            if theta_change < eps:
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
    # Use numerical energy for consistency
    final_energy = ham.energy_expectation_num(final_points, theta_values[-1], step)

    if track_progress:
        print("Optimization complete!")
        print(f"Final θ = {theta_values[-1]:.6f}")
        print(f"Final E = {final_energy:.6f}")
        print("Expected: θ ≈ 1.0, E ≈ -0.5")
    
    iterations = np.arange(len(energy_values))

    return iterations, theta_values[-1], theta_values, grad_values, energy_values

def h_optimiser_plot(iterations, e_history, grad_history, theta_history):
    """
    Visualise the results and process produced by the H_2 optimiser.
    Generates a plot for the energy values throughout the iterations, and one
    where the three parameter components' evolution is shown.
    
    Args:
    iterations (list): Iterations indexes.
    e_history (list): Respective energies.
    theta_history (list): Respective theta values.

    Returns:
    plt.plot: The energy evolution.
    plt.plot: The parameter evolution.
    """
    # Plot energy convergence.
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, e_history, linewidth=2)
    plt.grid(alpha=0.3)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Molecule Energy", fontsize=12)
    plt.title("Energy Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot gradient value.
    plt.figure(figsize=(7,4))
    plt.plot(iterations, grad_history, linewidth=2)
    plt.grid(alpha=0.3)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Molecule Energy Gradient", fontsize=12)
    plt.title("Energy Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot parameter convergence.
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, theta_history, linewidth=2)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Theta Value", fontsize=12)
    plt.title("Parameter Evolution", fontsize=14)
    plt.tight_layout()
    plt.show()

# Hydrogen Molecule Methods: ## Add normal gradient descent.

def h2_optimiser_vmc(theta, start, bond_length, stepsize=0.5, num_samples=50000,
                      alpha=0.05, m=100, eps=1e-3, burnin_val=5000):
    """
    Variational Monte Carlo optimiser for H2 molecule using stochastic gradient descent.

    Args:
    theta (list): Initial parameter array [theta1, theta2, theta3].
    start (list): Initial 6D electron coordinates.
    bond_length (float): Inter-nuclear distance.
    stepsize (float): Metropolis-Hastings step size.
    num_samples (int): Number of Monte Carlo samples per iteration.
    alpha (float): Learning rate.
    m (int): Maximum number of iterations.
    eps (float): Gradient norm convergence criterion.
    burnin_val (int): Number of burn-in steps for sampling.

    Returns:
    theta_opt (list): Optimized parameters.
    energy_opt (list): Energy at optimized parameters.
    theta_history (list): List of theta values.
    energy_history (list): List of energies.
    """
    theta_current = np.array(theta, dtype=float)
    theta_history = []
    energy_history = []
    grad_norm_history = []

    q1 = np.array([0, 0, - bond_length / 2])
    q2 = np.array([0, 0, bond_length / 2])

    for it in range(m):
        samples_6d = samp.samplings_h2_molecule(
            bond_length, start, theta_current,
            domain=None, stepsize=stepsize,
            num_samples=num_samples, burnin_val=burnin_val
        )

        el_samples = np.array([
            le.h2_le_sym(samples_6d[i,:3], samples_6d[i,3:], theta_current, q1, q2)
            for i in range(len(samples_6d))
        ])
        e_mean = el_samples.mean()

        grad = np.zeros_like(theta_current)
        for j in range(len(theta_current)):
            lnpsi_derivative = np.array([
                pdfs.wavefunction_hydrogen_molecule_theta_derivative(samples_6d[i,:3],
                                    samples_6d[i,3:], theta_current, j, q1, q2)
                for i in range(len(samples_6d))
            ])
            grad[j] = 2.0 * np.mean((el_samples - e_mean) * lnpsi_derivative)

        theta_current -= alpha * grad
        start = samples_6d[-1]

        theta_history.append(theta_current.copy())
        energy_history.append(e_mean)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        print(f"It. {it}: Theta={theta_current}, Energy={e_mean:.6f}, Grad. Norm={grad_norm:.6f}")

        if grad_norm < eps:
            print(f"Converged: Grad. Norm={grad_norm:.6f} < {eps}")
            break

    i_min = np.argmin(energy_history)
    theta_opt = theta_history[i_min]
    energy_opt = energy_history[i_min]
    iterations = np.arange(len(energy_history))

    print(f"Optimised theta = {theta_opt}, energy = {energy_opt:.6f}")
    return iterations, theta_opt, energy_opt, theta_history, grad_norm_history, energy_history

def h2_optimiser_plot(iterations, e_history, grad_history, theta_history):
    """
    Visualise the results and process produced by the H_2 optimiser.
    Generates a plot for the energy values throughout the iterations, and one
    where the three parameter components' evolution is shown.
    
    Args:
    iterations (list): Iterations indexes.
    e_history (list): Respective energies.
    theta_history (list): Respective theta values.

    Returns:
    plt.plot: The energy evolution.
    plt.plot: The parameter evolution.
    """
    if not isinstance(theta_history, np.ndarray):
        theta_history = np.array(theta_history)

    if not isinstance(grad_history, np.ndarray):
        grad_history = np.array(grad_history)

    # Plot energy convergence.
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, e_history, linewidth=2)
    plt.grid(alpha=0.3)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Molecule Energy", fontsize=12)
    plt.title("Energy Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot gradient value.
    plt.figure(figsize=(7,4))
    plt.plot(iterations, grad_history, linewidth=2)
    plt.grid(alpha=0.3)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Molecule Energy Gradient", fontsize=12)
    plt.title("Energy Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot parameter convergence.
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, theta_history[:, 0], label='Theta 1', linewidth=2)
    plt.plot(iterations, theta_history[:, 1], label='Theta 2', linewidth=2)
    plt.plot(iterations, theta_history[:, 2], label='Theta 3', linewidth=2)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Theta Values", fontsize=12)
    plt.title("Parameter Evolution", fontsize=14)
    plt.tight_layout()
    plt.show()
