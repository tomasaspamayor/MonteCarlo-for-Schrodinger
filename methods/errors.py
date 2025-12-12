import numpy as np
import matplotlib.pyplot as plt
import methods.differentiators as diff
import methods.pdfs as pdfs

plt.style.use('seaborn-v0_8-paper')

def error_calculation(wavefunction=None, coeffs=None, n=2):
    """
    Calculate and plot the errors in the finite difference method.

    Args:
    wavefunction (callable): The wavefunction to be studied.
    coeffs (list): The Hermite coefficients.
    n (int): Order of the wavefunction.

    Returns:
    int: Optimal truncation order.
    float: Optimal stepsize.
    float: Error of said optimal step.
    plt.plot: Plot of all errors against stepsize.
    """
    if wavefunction is None:
        wavefunction = pdfs.wavefunction_qho

    if coeffs is None:
        coeffs = [
        [1],
        [0, 2],
        [-2, 0, 4],
        [0, -12, 0, 8],
        [12, 0, -48, 0, 16],
        [0, 120, 0, -160, 0, 32],
        [-120, 0, 720, 0, -480, 0, 64]  ]

    x = np.linspace(-3, 3, 1000)
    h_values = np.logspace(-12, 1, 120)
    psi_exact = diff.analytical_second_derivative_qho(n, x)

    methods = {
        '2nd order': diff.cdm_step_second,
        '4th order': diff.cdm_step_fourth,
        '6th order': diff.cdm_step_sixth,
        '8th order': diff.cdm_step_eighth,
        '10th order': diff.cdm_step_tenth
    }

    errors_vals = {}
    optimal_h = {}
    min_errors = {}

    # Calculate errors for each method
    print("Calculating errors for each method:")
    for name, method in methods.items():
        method_errors = []
        for h in h_values:
            psi_numerical = method(x, wavefunction, h, coeffs, n)
            error = np.sqrt(np.mean((psi_numerical - psi_exact)**2))
            method_errors.append(error)

        errors_vals[name] = method_errors
        min_idx = np.argmin(method_errors)
        optimal_h[name] = h_values[min_idx]
        min_errors[name] = method_errors[min_idx]

        print(f"{name}: Min error = {method_errors[min_idx]:.2e} at h = {h_values[min_idx]:.2e}")

    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 'o', 'o', 'o', 'o']

    for (name, method_errors), color, marker in zip(errors_vals.items(), colors, markers):
        plt.loglog(h_values, method_errors, marker=marker, markersize=4,
                linestyle='-', linewidth=1.5, color=color, label=name)

        min_idx = np.argmin(method_errors)
        plt.plot(h_values[min_idx], method_errors[min_idx],
                marker='*', markersize=10, color=color)

    plt.xlabel('Step size h (log scale)', fontsize=12)
    plt.ylabel('RMS Error (log scale)', fontsize=12)
    plt.title(f'RMS Error vs. Step Size for Finite Difference Schemes (n={n} QHO)', fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    best_method = min(min_errors, key=min_errors.get)
    best_stepsize = optimal_h[best_method]
    best_error = min_errors[best_method]

    return best_method, best_stepsize, best_error

def theta_uncertainty(theta_history, A):
    """
    Compute the uncertainty on the convergeing thetas.
    
    :param theta_history: Description
    :param A: Description
    """
    converged_thetas = theta_history[-A:]
    theta_std = np.std(converged_thetas)
    theta_uncertainty_val = theta_std / np.sqrt(len(converged_thetas))
    return theta_uncertainty_val

def energy_uncertainty(energy_history, A):
    """
    Compute the uncertainty on the convergeing energies.
    
    theta_history: Description
    A: Description
    """
    converged_energies = energy_history[-A:]
    energy_std = np.std(converged_energies)
    energy_uncertainty_val = energy_std / np.sqrt(len(converged_energies))
    return energy_uncertainty_val
