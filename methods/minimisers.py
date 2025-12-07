""""
Module to hold all the function minimising methods.
"""
import numpy as np
import methods.pdfs as pdfs
import methods.sampling as samp
import methods.hamiltonians as ham
import methods.local_energy as le

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
    Optimise Hydrogen wavefunction parameter theta. Used Gradient Descent.

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

def h2_optimiser_gd(theta, start, stepsize, bond_length, delta, num_samples, alpha, m, eps, burnin_val):
    """
    Optimise the hydrogen molecule theta by the gradient descent method.
    """
    theta_current = np.array(theta, dtype=float)
    theta_vals = []
    e_vals= []
    grad_vals = []
    grad_norms_vals = []

    for i in range(m):
        samples_6d = samp.samplings_h2_molecule(bond_length, start, theta_current, domain=None, stepsize=stepsize, num_samples=num_samples, burnin_val=burnin_val)
        
        current_energy = ham.h2_energy_expectation(samples_6d, bond_length, theta_current)
        grad = np.zeros_like(theta_current)
        d = len(theta_current)

        for j in range(d):
            theta_plus = theta_current.copy()
            theta_plus[j] += delta
            samples_6d_plus = samp.samplings_h2_molecule(bond_length, start, theta_plus, domain=None, stepsize=stepsize, num_samples=num_samples, burnin_val=burnin_val)
            energy_plus = ham.h2_energy_expectation(samples_6d_plus, bond_length, theta_plus)
            grad[j] = (energy_plus - current_energy) / delta

        e_vals.append(current_energy)
        theta_vals.append(theta_current.copy())
        grad_norm = np.linalg.norm(grad)
        grad_vals.append(grad)
        grad_norms_vals.append(grad_norm)

        print(f"Iteration {i}: R = {theta_current}, E = {current_energy}, grad = {grad}, grad norm = {grad_norm}")

        if i > 0 and np.linalg.norm(theta_current - theta_vals[-1]) < eps:
            print(f'Convergence has been reached: Theta values: {theta_current}, Gradient = {grad}, Gradient Norm = {grad_norm}')
            break

        start = samples_6d[-1]
        theta_current = theta_current - alpha * grad
        theta_current = np.clip(theta_current, 0.1, 2.0)

    minimum_index = np.argmin(e_vals)
    theta_optimised = theta_vals[minimum_index]
    e_optimised = e_vals[minimum_index]

    print(f'The optimal theta is {theta_optimised}, with energy {e_optimised}.')

    return theta_optimised, e_optimised, theta_vals, e_vals

def h2_optimiser_gd_2(theta, start, stepsize, bond_length, delta, num_samples, alpha, m, eps, burnin_val):
    """
    Optimise the hydrogen molecule theta by the gradient descent method.
    """
    theta_current = np.array(theta, dtype=float)
    alpha_current = alpha
    theta_vals = []
    e_vals= []
    grad_vals = []
    grad_norms_vals = []

    for i in range(m):
        samples_6d = samp.samplings_h2_molecule(bond_length, start, theta_current,
                                            domain=None, stepsize=stepsize,
                                            num_samples=num_samples,
                                            burnin_val=burnin_val)
    
        current_energy = ham.h2_energy_expectation(samples_6d, bond_length, theta_current)

        grad = np.zeros_like(theta_current)
    
        for j in range(len(theta_current)):
            theta_plus = theta_current.copy()
            theta_plus[j] += delta
            theta_minus = theta_current.copy()
            theta_minus[j] -= delta

            samples_6d_plus = samp.samplings_h2_molecule(
                bond_length, start, theta_plus,
                domain=None, stepsize=stepsize,
                num_samples=num_samples,
                burnin_val=burnin_val
            )
            samples_6d_minus = samp.samplings_h2_molecule(
                bond_length, start, theta_minus,
                domain=None, stepsize=stepsize,
                num_samples=num_samples,
                burnin_val=burnin_val
            )

            energy_plus  = ham.h2_energy_expectation(samples_6d_plus,  bond_length, theta_plus)
            energy_minus = ham.h2_energy_expectation(samples_6d_minus, bond_length, theta_minus)

            grad[j] = (energy_plus - energy_minus) / (2 * delta)

        grad_norm = np.linalg.norm(grad)    
        e_vals.append(current_energy)
        theta_vals.append(theta_current.copy())
        grad_vals.append(grad.copy())
        grad_norms_vals.append(grad_norm)

        print(f"Iteration {i}: R = {theta_current}, E = {current_energy}, grad = {grad}, grad norm = {grad_norm}")

        start = samples_6d[-1]
        theta_current = theta_current - alpha_current * grad
        alpha_current *= 0.99

        if grad_norm < eps:
            print(f'Converged: gradient norm {grad_norm:.6f} < {eps}')
            break

    minimum_index = np.argmin(e_vals)
    theta_optimised = theta_vals[minimum_index]
    e_optimised = e_vals[minimum_index]

    print(f'The optimal theta is {theta_optimised}, with energy {e_optimised}.')

    return theta_optimised, e_optimised, theta_vals, e_vals

def h2_optimiser_gd_chat(theta, start, stepsize, bond_length, delta,
                         num_samples, alpha, m, eps, burnin_val):
    """
    Optimise the hydrogen molecule theta by gradient descent using
    correlated finite differences and MCMC sampling.
    """

    theta_current = np.array(theta, dtype=float)
    alpha_current = alpha

    theta_vals = []
    e_vals = []
    grad_vals = []
    grad_norms_vals = []

    # Precompute nuclear positions once
    q1 = np.array([0, 0, -bond_length / 2])
    q2 = np.array([0, 0,  bond_length / 2])

    # Local fast alias to the local-energy function
    le_func = le.h2_le_sym

    for it in range(m):

        # --- SAMPLE ----------------------------------------------------------
        samples = samp.samplings_h2_molecule(
            bond_length=bond_length,
            initial_point=start,
            theta=theta_current,
            domain=None,
            stepsize=stepsize,
            num_samples=num_samples,
            burnin_val=burnin_val
        )

        # Pre-split coordinates ONCE — huge performance improvement
        coords = [(s[:3], s[3:]) for s in samples]

        # --- CURRENT ENERGY --------------------------------------------------
        E_current_samples = np.array(
            [le_func(r1, r2, theta_current, q1, q2) for (r1, r2) in coords]
        )
        E_current = E_current_samples.mean()

        # --- GRADIENT (FINITE DIFFERENCE WITH SHARED SAMPLES) ---------------
        n_params = len(theta_current)
        grad = np.zeros(n_params)

        for j in range(n_params):
            # Prepare ± parameter vectors
            theta_plus  = theta_current.copy()
            theta_minus = theta_current.copy()
            theta_plus[j]  += delta
            theta_minus[j] -= delta

            # Compute local energies for +δθ_j
            E_plus_samples = np.array(
                [le_func(r1, r2, theta_plus, q1, q2) for (r1, r2) in coords]
            )

            # Compute local energies for -δθ_j
            E_minus_samples = np.array(
                [le_func(r1, r2, theta_minus, q1, q2) for (r1, r2) in coords]
            )

            # Central finite difference derivative
            grad[j] = (E_plus_samples.mean() - E_minus_samples.mean()) / (2 * delta)

        grad_norm = np.linalg.norm(grad)

        # Record diagnostic values
        e_vals.append(E_current)
        theta_vals.append(theta_current.copy())
        grad_vals.append(grad.copy())
        grad_norms_vals.append(grad_norm)

        print(f"Iteration {it}: θ={theta_current}, "
              f"E={E_current:.6f}, |grad|={grad_norm:.6f}")

        # Update for next MCMC chain
        start = samples[-1]

        # Gradient descent update
        theta_current = theta_current - alpha_current * grad
        alpha_current *= 0.99  # learning-rate decay

        if grad_norm < eps:
            print(f"Converged: gradient norm {grad_norm:.6f} < {eps}")
            break

    # Pick the lowest-energy result ever reached
    idx_min = np.argmin(e_vals)
    theta_opt = theta_vals[idx_min]
    e_opt = e_vals[idx_min]

    print(f"\nOptimised θ = {theta_opt}, E = {e_opt}")

    return theta_opt, e_opt, theta_vals, e_vals

def h2_optimiser_vmc(theta, start, bond_length, stepsize=0.5, num_samples=50000,
                      alpha=0.05, m=100, eps=1e-3, burnin_val=5000):
    """
    Variational Monte Carlo optimiser for H2 molecule using stochastic gradient descent.
    
    Args:
        theta: initial parameter array [theta1, theta2, theta3]
        start: initial 6D electron coordinates
        bond_length: inter-nuclear distance
        stepsize: Metropolis-Hastings step size
        num_samples: number of Monte Carlo samples per iteration
        alpha: learning rate
        m: maximum number of iterations
        eps: gradient norm convergence criterion
        burnin_val: number of burn-in steps for sampling
    
    Returns:
        theta_opt: optimized parameters
        energy_opt: energy at optimized parameters
        theta_history: list of theta values
        energy_history: list of energies
    """
    theta_current = np.array(theta, dtype=float)
    theta_history = []
    energy_history = []

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
                pdfs.h2_wavefunction_theta_derivative(samples_6d[i,:3],
                                    samples_6d[i,3:], theta_current, j, q1, q2)
                for i in range(len(samples_6d))
            ])
            grad[j] = 2.0 * np.mean((el_samples - e_mean) * lnpsi_derivative)

        theta_current -= alpha * grad
        start = samples_6d[-1]

        theta_history.append(theta_current.copy())
        energy_history.append(e_mean)
        grad_norm = np.linalg.norm(grad)
        print(f"Iteration {it}: Theta = {theta_current}, Energy = {e_mean:.6f}, Gradient Norm = {grad_norm:.6f}")

        if grad_norm < eps:
            print(f"Converged: |grad| = {grad_norm:.6f} < {eps}")
            break

    i_min = np.argmin(energy_history)
    theta_opt = theta_history[i_min]
    energy_opt = energy_history[i_min]

    print(f"Optimised theta = {theta_opt}, energy = {energy_opt:.6f}")
    return theta_opt, energy_opt, theta_history, energy_history
