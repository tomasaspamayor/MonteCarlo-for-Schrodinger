"""
See the script to solve the different questions with the methods defined
in other files.
"""
#%% Imports and constants
from functools import partial
import numpy as np

import two_one.errors as err
import two_one.local_energy as le
import two_two.sampling as samp
import two_two.pdfs as pdfs
import three.minimisers as minimisers
import three.hamiltonians as ham

hermite_coeffs = [
    [1],
    [0, 2],
    [-2, 0, 4],
    [0, -12, 0, 8],
    [12, 0, -48, 0, 16],
    [0, 120, 0, -160, 0, 32],
    [-120, 0, 720, 0, -480, 0, 64]  ]

P = len(hermite_coeffs)

#%% 2.1 - Local Energy

## To change: update code to serve all truncations. Fix error calculation.
H_N = len(hermite_coeffs)
x_example = np.linspace(-4, 4, 1000)

x_vals_le = []
local_energy_vals = []

for i in range(H_N):
    for j in range(5):
        x_vals_le_second_i, local_energy_vals_second_i = le.local_energy(x_example, 0.01,
                                                        hermite_coeffs, i, 0)
        x_vals_le.append(x_vals_le_second_i)
        local_energy_vals.append(local_energy_vals_second_i)

x_vals_le = np.array(x_vals_le)
local_energy_vals = np.array(local_energy_vals)

# Calculating the RMS error on these calculations:

stepsize_vals, err_vals_second = err.err_finite_diff([1e-4, 0.01], 500,
                                            [-1, 1], 1000, hermite_coeffs, 0)
_, err_vals_fourth = err.err_finite_diff([1e-4, 0.01], 500,
                                            [-1, 1], 1000, hermite_coeffs, 1)
_, err_vals_sixth = err.err_finite_diff([1e-4, 0.01], 500,
                                            [-1, 1], 1000, hermite_coeffs, 2)
_, err_vals_eighth = err.err_finite_diff([1e-4, 0.01], 500,
                                            [-1, 1], 1000, hermite_coeffs, 3)
_, err_vals_tenth = err.err_finite_diff([1e-4, 0.01], 500,
                                            [-1, 1], 1000, hermite_coeffs, 4)

err.plot_err_methods([1e-4, 0.01], 500, [-1, 1], 1000, hermite_coeffs)

#%% 2.2 - Sampling & Eingenvalues

## We plot the algorithms' samples for different example PDFs.

samples_gaussian_r = samp.rejection(pdfs.normalized_gaussian, 1, -1, 1, 100000, 1e8)
samples_exponential_r = samp.rejection(pdfs.normalized_exponential, 1, 0, 1, 100000, 1e8)

samp.plot_samples(pdfs.normalized_gaussian, [-1, 1], samples_gaussian_r, 50, 0)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_r, 50, 0)

samples_gaussian_mh = samp.metropolis_hastings(pdfs.normalized_gaussian, 0, [-1, 1], 0.5, 100000)
samples_exponential_mh = samp.metropolis_hastings(pdfs.normalized_exponential,
                                                     0.5, [0, 1], 0.5, 100000)
samp.plot_samples(pdfs.normalized_gaussian, [-1, 1], samples_gaussian_mh, 50, 1)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_mh, 50, 1)

samples_gaussian_r_3d = samp.rejection_3d(pdfs.gaussian_3d, 0, [[-0.5,0.5], [-0.5,0.5],
                                                [-0.5,0.5]], 1000000, 10000000, m=None)
samples_gaussian_mh_3d = samp.metropolis_hastings_3d(pdfs.gaussian_3d, [0, 0, 0], [[-0.5, 0.5],
                                                       [-0.5, 0.5], [-0.5, 0.5]], 0.5, 1000000)

samp.plot_3d_samples(samples_gaussian_r_3d, 40, 0)
samp.plot_3d_samples(samples_gaussian_mh_3d, 40, 1)

for k in range(P):
    pdf_qo = partial(pdfs.wavefunction_qho_pdf, n=k, coeffs=hermite_coeffs)
    samples_wf_r = samp.rejection(pdf_qo, 0, -4.6, 4.5, 500000, 5000000, m=None)
    samp.plot_samples(pdf_qo, [-4.6, 4.5], samples_wf_r, 200, 0)
    local_energies_r = le.local_energy_qho_numerical(samples_wf_r, h=1e-4,
                                            coeffs=hermite_coeffs, level=k, method=None)
    energy_k_r = np.mean(local_energies_r[0])
    print(f'Local energy (order {k}): {energy_k_r} (Rejection sampling)')

for k in range(P):
    pdf_qo = partial(pdfs.wavefunction_qho_pdf, n=k, coeffs=hermite_coeffs)
    samples_wf_mh = samp.metropolis_hastings(pdf_qo, 0, [-4.5, 4.5], 0.05, 500000, 250000)
    samp.plot_samples(pdf_qo, [-4.5, 4.5], samples_wf_mh, 200, 1)
    local_energies_mh = le.local_energy_qho_numerical(samples_wf_mh, h=1e-4,
                                             coeffs=hermite_coeffs, level=k, method=None)
    energy_k_mh = np.mean(local_energies_mh[0])
    print(f'Local energy (order {k}): {energy_k_mh} (Metropolis-Hastings algorithm)')

#%% 3 - Hydrogen Ground State

def wf_optimiser(theta, m=1000, eps=1e-5, domain=np.array([[-4, 4], [-4, 4], [-4, 4]]), num_samples=10000):
    """"
    Optimise the wavefunction by variating the parameter theta. Used Quasi-Newton Algorithm.
    """
    theta_current = theta
    theta_values = []
    energy_values = []

    def current_pdf(x):
        return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_current)

    for i in range(m):
        x_points = samp.rejection_3d(current_pdf, 0, domain, num_samples, num_samples*10000, m=None)
        samp.plot_3d_samples(x_points, 100, 1)

        current_energy = ham.energy_expectation(x_points, theta_current)
        energy_values.append(current_energy)
        print(f"Iteration {i}: theta = {theta_current:.6f}, energy = {current_energy:.6f}")

        def energy_wrapped(theta):
            def current_pdf(x):
                return pdfs.wavefunction_hydrogen_atom_pdf(x, theta)
            x_points = samp.rejection_3d(current_pdf, 0, domain, num_samples,
                                         num_samples*10000, m=None)
            return ham.energy_expectation(x_points, theta)

        def gradient_wrapped(theta):
            def current_pdf(x):
                return pdfs.wavefunction_hydrogen_atom_pdf(x, theta)
            x_points = samp.rejection_3d(current_pdf, 0, domain, num_samples,
                                         num_samples*10000, m=None)
            return ham.energy_expectation_theta_derivative(x_points, theta)

        theta_opt = minimisers.quasi_newton(energy_wrapped, gradient_wrapped,
                                            theta_current, 1e-7, 10000)
        theta_values.append(theta_opt)
        theta_current = theta_opt

        if i > 1 and np.abs(theta_values[-1] - theta_values[-2]) < eps:
            print(f"Final theta = {theta_values[-1]:.6f}, Final energy = {energy_values[-1]:.6f}")
            break

    def final_pdf(x):
        return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_values[-1])
    x_points = samp.rejection_3d(final_pdf, 0, domain, num_samples, num_samples*10000, m=None)
    samp.plot_3d_samples(x_points, 100, 1)

    return theta_values[-1], theta_values, energy_values

# %%
