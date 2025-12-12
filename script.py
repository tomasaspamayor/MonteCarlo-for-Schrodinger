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

iterations_h2, theta_opt_h2, e_opt, th_history_h2, grad_norm_history_h2, e_history_h2, t_unc_h2, e_unc_h2 = minimisers.h2_optimiser_vmc(
    theta=[0.5, 0.5, 0.5],
    start=[0.0, 0.0, -0.5, 0.0, 0.0, 0.5],
    bond_length=2.0,
    stepsize=0.1,
    num_samples=200000,
    alpha=0.05,
    m=50,
    eps=1e-4,
    burnin_val=50000
)

minimisers.h_optimiser_plot(iterations_h2, e_history_h2, grad_norm_history_h2, th_history_h2)

def h2_pdf_opt(pos_6d):
    """PDF for Hydrogen Molecule"""
    r1 = pos_6d[:3]
    r2 = pos_6d[3:]
    wf = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta_opt_h2, q1, q2)
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

bond_lengths = [0.5, 1, 2, 3]
q1 = np.array([0, 0, -bond_length/2])
q2 = np.array([0, 0, bond_length/2])
start_pos = [0.1, 0, -0.7, -0.1, 0, 0.7]
domain_6d = [[-3, 3], [-3, 3], [-3, 3],
             [-3, 3], [-3, 3], [-3, 3]]

for i in range(4):
    def h2_pdf_opt_bl(pos_6d):
        """PDF for Hydrogen Molecule"""
        r1 = pos_6d[:3]
        r2 = pos_6d[3:]
        wf = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta_opt_h2, [0, 0, -bond_lengths[i]/2], [0, 0, bond_lengths[i]/2])
        return wf ** 2

    samples_opt_calc = samp.metropolis_hastings_3d_opt(
        pdf=h2_pdf_opt_bl,
        start=start_pos,
        domain=domain_6d,
        stepsize=0.1,
        num_samples=2500000,
        adapt_interval=2000,
        burnin_val=250000,
        dimensions=6
    )
    samp.plot_6d_samples(samples_opt_calc, bins=100)

#%%
theta_morse = np.array([1.07244149, 0.5457916 , 0.49039177])
bond_length_vals, energy_vals, energy_uncertainites = morse.bond_length_energies([0.5, 3], theta_morse, 20)
#%%

bond_lengths_e = np.array([0.5       , 0.63157895, 0.76315789, 0.89473684, 1.02631579,
       1.15789474, 1.28947368, 1.42105263, 1.55263158, 1.68421053,
       1.81578947, 1.94736842, 2.07894737, 2.21052632, 2.34210526,
       2.47368421, 2.60526316, 2.73684211, 2.86842105, 3.        ])

energy_value_e = np.array([-0.28601037, -0.63709579, -0.85099871, -0.96930524, -1.03372281,
       -1.09789324, -1.10875586, -1.12440668, -1.1207862 , -1.1302163 ,
       -1.12050434, -1.11035835, -1.10685506, -1.08452129, -1.07111344,
       -1.06707832, -1.05840597, -1.05473801, -1.04197196, -1.03518121])

energy_uncertainties_e = np.array([0.00162731, 0.00160187, 0.00142811, 0.00111893, 0.00100208,
       0.00084931, 0.00092753, 0.00087459, 0.00069726, 0.0005744 ,
       0.00054837, 0.00049509, 0.0004743 , 0.00047429, 0.00047113,
       0.00048683, 0.0003927 , 0.00035979, 0.00036422, 0.00035378])

D_val, a_val, r0_val, pcov = morse.morse_fitting(
    bond_lengths_e, 
    energy_value_e, 
    energy_uncertainties_e, e_single=-0.5,
    p0=np.array([0.17, 1.0, 1.4, -0.5])
)

# The fixed value for the single hydrogen atom energy in a.u.
FIXED_E_SINGLE = -0.5 

# Call morse_fitting, providing the required FIXED_E_SINGLE as the 4th positional argument.
# IMPORTANT: Since you are now fitting with a fixed offset (E_single), 
# the function should only return D, A, r0, and pcov (4 values, not 5). 
# We remove the E_limit_val unpacking.

D_val, a_val, r0_val, pcov = morse.morse_fitting( 
    bond_lengths_e,      
    energy_value_e,      
    energy_uncertainties_e,      
    e_single=FIXED_E_SINGLE, # <--- PASS THE REQUIRED ARGUMENT
    p0=np.array([0.17, 1.0, 1.4]) # <--- GUESS IS NOW ONLY 3 PARAMETERS (D, A, r0)
)
# 2. Call the plotting function with the fixed E_SINGLE value
morse.morse_plot(D_val, a_val, r0_val, bond_lengths_e, energy_value_e, FIXED_E_SINGLE)

# 3. Calculate and print uncertainties
D_unc = np.sqrt(pcov[0, 0])
A_unc = np.sqrt(pcov[1, 1])
r0_unc = np.sqrt(pcov[2, 2])
print(f'\nThe fitted bond length is {r0_val:.4f} \u00B1 {r0_unc:.4e}.')

# %%
