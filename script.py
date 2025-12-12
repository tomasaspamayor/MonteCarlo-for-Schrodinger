"""
See the script to solve the different questions with the methods defined
in other files.
"""
#%% Imports and constants

from functools import partial
import numpy as np

import methods.errors as err
import methods.local_energy as le
import methods.sampling as samp
import methods.differentiators as diff
from methods import pdfs
from methods import minimisers
from methods import morse
#import data

hermite_coeffs = [
    [1],
    [0, 2],
    [-2, 0, 4],
    [0, -12, 0, 8],
    [12, 0, -48, 0, 16],
    [0, 120, 0, -160, 0, 32],
    [-120, 0, 720, 0, -480, 0, 64]  ]

P = len(hermite_coeffs)

#%% 2.1 - QHO Local Energy Error Calculations

x_example = np.linspace(-4, 4, 1000)

x_vals_le = []
local_energy_vals = []

for i in range(P):
    for j in range(5):
        x_vals_le_second_i, local_energy_vals_second_i = le.local_energy_qho_numerical(
                                                x_example, 0.01, hermite_coeffs, i, 0)
        x_vals_le.append(x_vals_le_second_i)
        local_energy_vals.append(local_energy_vals_second_i)

x_vals_le = np.array(x_vals_le)
local_energy_vals = np.array(local_energy_vals)

# Calculating the RMS error on these calculations:
optimal_methods = []
optimal_stepsizes = []
stepsize_errors = []
for i in range(P):
    optimal_method_i, optimal_stepsize_i, stepsize_error_i = err.error_calculation(n=i)
    optimal_methods.append(optimal_method_i)
    optimal_stepsizes.append(optimal_stepsize_i)
    stepsize_errors.append(stepsize_error_i)

print(optimal_methods)
print(optimal_stepsizes)
print(stepsize_errors)

optimal_stepsize = optimal_stepsizes[0]

#%% 2.2 - QHO Sampling & Eingenvalues

## We plot the algorithms' samples for different example PDFs.

samples_gaussian_r = samp.rejection(pdfs.normalized_gaussian, 1, -1, 1, 250000, 1e8)
samples_exponential_r = samp.rejection(pdfs.normalized_exponential, 1, 0, 1, 25000, 1e8)

samp.plot_samples(pdfs.normalized_gaussian, [-1, 1], samples_gaussian_r, 75, 0, 0)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_r, 75, 0, 0)

samples_gaussian_mh = samp.metropolis_hastings_opt(
    pdfs.normalized_gaussian, 
    start=0.0,
    stepsize=0.05,
    num_samples=500000,
    burnin_val=250000,
    adapt_interval=500
)
samples_exponential_mh = samp.metropolis_hastings_opt(
    pdfs.normalized_exponential,
    start=0.0,
    stepsize=0.05,
    num_samples=250000,
    burnin_val=20000,
    adapt_interval=500
)

samp.plot_samples(pdfs.normalized_gaussian, [-1,1], samples_gaussian_mh, 75, 1, 0)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_mh, 50, 1, 0)
P = 6
optimal_stepsize = 0.030718143012686966

for k in range(P):
    pdf_qo = partial(pdfs.wavefunction_qho_pdf, n=k, coeffs=hermite_coeffs)
    samples_wf_r = samp.rejection(pdf_qo, 0, -4.6, 4.5, 1000000, 1e8, m=None)
    samp.plot_samples(pdf_qo, [-4.6, 4.5], samples_wf_r, 200, 0, k)
    local_energies_r = le.local_energy_qho_numerical(samples_wf_r, optimal_stepsize,
                                            coeffs=hermite_coeffs, level=k, method=None)
    energy_k_r = np.mean(local_energies_r[0])
    print(f'Local energy (order {k}): {energy_k_r} (Rejection sampling)')

for k in range(P):
    pdf_qo = partial(pdfs.wavefunction_qho_pdf, n=k, coeffs=hermite_coeffs)
    samples_wf_mh = samp.metropolis_hastings_opt(
        pdf_qo,
        start=0.0,
        stepsize=0.3,
        num_samples=1000000,
        burnin_val=500000,
        adapt_interval=500,
    )
    samp.plot_samples(pdf_qo, [-4.5, 4.5], samples_wf_mh, 200, 1, k)
    local_energies_mh = le.local_energy_qho_numerical(samples_wf_mh, optimal_stepsize,
                                             coeffs=hermite_coeffs, level=k, method=None)
    energy_k_mh = np.mean(local_energies_mh[0])
    print(f'Local energy (order {k}): {energy_k_mh} (Metropolis-Hastings algorithm)')

#%% 3 - Hydrogen Ground State Optimising

optimal_stepsize = 0.030718143012686966
# We check the numerical Laplacian works well:
diff.lap_comp_plot(1, step=1e-4) # 4th
diff.lap_comp_plot(1, method=8, step=optimal_stepsize) # 8th

theta_guess = 0.7
A = 40

def intial_pdf_h(x):
    return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_guess)

initial_samples_h = samp.metropolis_hastings_3d_opt(
                intial_pdf_h, 
                [0.0, 0.0, 0.0], 
                domain=np.array([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]]), 
                stepsize=0.05, 
                num_samples=5000000, 
                burnin_val=500000,
                dimensions=3,
                adapt_interval=750
            )

samp.plot_3d_samples(initial_samples_h, 100, 1)

iterations, theta_optimal, theta_history, grad_history, energy_history = minimisers.hydrogen_wavefunction_optimiser_gd(theta_guess, m=100, stepsize=0.05, eps=1e-8, learning_rate=0.1, num_samples=1000000, burnin_val=200000)
theta_unc = err.theta_uncertainty(theta_history, A)
energy_unc = err.energy_uncertainty(energy_history, A)
print(f'The optimal theta value is: {theta_optimal} +- {theta_unc}, with energy: {energy_history[-1]} +- {energy_unc} for Gradient Descent.')
minimisers.h_optimiser_plot(iterations, energy_history, grad_history, theta_history[1:])

theta_guess = 0.7
iterations_num, theta_optimal_num, theta_history_num, grad_history_num, energy_history_num = minimisers.hydrogen_wavefunction_optimiser_gd_num(theta_guess, step=1e-2, h=0.01, m=125, stepsize=0.05, eps=1e-8, num_samples=10000, burnin_val=2000, learning_rate=0.5)
theta_unc_num = err.theta_uncertainty(theta_history_num, A)
energy_unc_num = err.energy_uncertainty(energy_history_num, A)
print(f'The optimal theta value is: {theta_optimal} +- {theta_unc_num}, with energy: {energy_history[-1]} +- {energy_unc_num} for Numerical.')
minimisers.h_optimiser_plot(iterations_num, energy_history_num, grad_history_num, theta_history_num[1:])

def opt_pdf_h(x):
    return pdfs.wavefunction_hydrogen_atom_pdf(x, theta_optimal)

opt_samples_h = samp.metropolis_hastings_3d_opt(
                opt_pdf_h, 
                [0.0, 0.0, 0.0], 
                domain=np.array([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]]), 
                stepsize=0.05, 
                num_samples=5000000, 
                burnin_val=500000,
                dimensions=3,
                adapt_interval=750
            )
samp.plot_3d_samples(opt_samples_h, 100, 1)

e_single = energy_history[-1]

#%% 4 - Hydrogen Molecule Optimisation

# First we plot out the original sampling:
# Define some needed constants:
theta = np.array([1.0, 1.0, 1.0])
bond_length = 2.0
num_samples = 5000000
q1 = np.array([0, 0, -bond_length/2])
q2 = np.array([0, 0, bond_length/2])
start_pos = [0.1, 0, -0.7, -0.1, 0, 0.7]
domain_6d = [[-3, 3], [-3, 3], [-3, 3],
             [-3, 3], [-3, 3], [-3, 3]]

def h2_pdf(pos_6d):
    """PDF for Hydrogen Molecule"""
    r1 = pos_6d[:3]
    r2 = pos_6d[3:]
    wf = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2)
    return wf ** 2

samples = samp.metropolis_hastings_3d_opt(
    pdf=h2_pdf,
    start=start_pos,
    domain=domain_6d,
    stepsize=0.1,
    num_samples=num_samples,
    adapt_interval=2000,
    burnin_val=int(num_samples*0.1),
    dimensions=6
)

samp.plot_6d_samples(samples, bins=100)

iterations, theta_opt, e_opt, th_history, grad_norm_history, e_history, t_unc, e_unc = minimisers.h2_optimiser_vmc(
    theta=[0.5, 0.5, 0.5],
    start=[0.0, 0.0, -0.5, 0.0, 0.0, 0.5],
    bond_length=2.0,
    stepsize=0.5,
    num_samples=200000,
    alpha=0.05,
    m=40,
    eps=1e-3,
    burnin_val=10000
)

minimisers.h2_optimiser_plot(iterations, e_history, grad_norm_history, th_history)

def h2_pdf_opt(pos_6d):
    """PDF for Hydrogen Molecule"""
    r1 = pos_6d[:3]
    r2 = pos_6d[3:]
    wf = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta_opt, q1, q2)
    return wf ** 2

samples_opt = samp.metropolis_hastings_3d_opt(
    pdf=h2_pdf_opt,
    start=start_pos,
    domain=domain_6d,
    stepsize=0.1,
    num_samples=num_samples,
    adapt_interval=2000,
    burnin_val=int(num_samples*0.1),
    dimensions=6
)

samp.plot_6d_samples(samples_opt, bins=100)

theta_morse = theta_opt
bond_length_vals, energy_vals, energy_uncertainites = morse.bond_length_energies([0.5, 3],
                                theta_morse, 200, num_samples=1000000)

#bond_length_vals = data.bond_length_vals
#N = len(bond_length_vals)
#energy_vals = data.energy_vals
#energy_uncertainites = np.ones(N)

D_val, a_val, r0_val, pcov = morse.morse_fitting(bond_length_vals, energy_vals, energy_uncertainites, e_single)
morse.morse_plot(D_val, a_val, r0_val, bond_length_vals, energy_vals, 1.4)

D_unc = np.sqrt(pcov[0, 0])
A_unc = np.sqrt(pcov[1, 1])
r0_unc = np.sqrt(pcov[2, 2])

print(f'The fitted bond length is {r0_val} +- {r0_unc}, and the dissociation energy {D_val} +- {D_unc}.')

# %%
