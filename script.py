"""
See the script to solve the different questions with the methods defined
in other files.
"""
# Imports and constants

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import methods.errors as err
import methods.local_energy as le
import methods.sampling as samp
import methods.differentiators as diff
from methods import pdfs
from methods import minimisers
from methods import morse

hermite_coeffs = [
    [1],
    [0, 2],
    [-2, 0, 4],
    [0, -12, 0, 8],
    [12, 0, -48, 0, 16],
    [0, 120, 0, -160, 0, 32],
    [-120, 0, 720, 0, -480, 0, 64]  ]

P = len(hermite_coeffs)

# 2.1 - QHO Local Energy Error Calculations

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

# 2.2 - QHO Sampling & Eingenvalues

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

# 3 - Hydrogen Ground State Optimising

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

theta_guess = 0.7

iterations, theta_optimal, theta_history, grad_history, energy_history = minimisers.hydrogen_wavefunction_optimiser_gd(theta_guess, m=100, stepsize=0.05, eps=1e-8, learning_rate=0.1, num_samples=1000000, burnin_val=200000)
theta_unc = err.theta_uncertainty(theta_history, A)
energy_unc = err.energy_uncertainty(energy_history, A)
print(f'The optimal theta value is: {theta_optimal} +- {theta_unc}, with energy: {energy_history[-1]} +- {energy_unc} for Gradient Descent.')
minimisers.h_optimiser_plot(iterations, energy_history, grad_history, theta_history[1:])

iterations_num, theta_optimal_num, theta_history_num, grad_history_num, energy_history_num = minimisers.hydrogen_wavefunction_optimiser_gd_num(theta_guess, step=1e-2, h=0.01, m=100, stepsize=0.05, eps=1e-8, num_samples=1000000, burnin_val=200000, learning_rate=0.1)
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

# 4 - Hydrogen Molecule Optimisation

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

theta_morse = theta_opt_h2
bond_length_vals, energy_vals, energy_uncertainties = morse.bond_length_energies([0.5, 3], theta_morse, 20)

def morse_f(r, D, a, r0):
    E_single = e_single
    return D * (1 - np.exp(-a * (r - r0))) ** 2 - D + 2 * E_single

p0 = [0.2, 1.0, 1.0]
popt, pcov = curve_fit(morse_f, bond_length_vals, energy_vals, p0=p0, 
                      sigma=energy_uncertainties, absolute_sigma=True)

D_fit, a_fit, r0_fit = popt

D_err = np.sqrt(pcov[0, 0])
a_err = np.sqrt(pcov[1, 1])
r0_err = np.sqrt(pcov[2, 2])

print("Fit results:")
print(f"D  = {D_fit:.4f} ± {D_err:.4f}")
print(f"a  = {a_fit:.4f} ± {a_err:.4f}")
print(f"r0 = {r0_fit:.4f} ± {r0_err:.4f}")

r_smooth = np.linspace(0.5, 3.0, 200)
energies_smooth = morse_f(r_smooth, D_fit, a_fit, r0_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(bond_length_vals, energy_vals, yerr=energy_uncertainties,
             fmt='o', color='blue', markersize=6, capsize=3,
             label='VMC Data', alpha=0.8)
plt.plot(r_smooth, energies_smooth, 'r-', linewidth=2.5, label='Fitted Morse Potential')
plt.axvline(x=r0_fit, color='green', linestyle='--', alpha=0.7,
            label=f'Equilibrium length: r = {r0_fit:.3f}')
plt.xlabel('Bond Length', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Morse Potential Fit to Hydrogen Molecule Bonding Energy', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
