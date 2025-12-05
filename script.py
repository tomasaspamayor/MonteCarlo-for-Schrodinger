"""
See the script to solve the different questions with the methods defined
in other files. ## Use optimal stepsize. Fix Metropolis Hastings. Fix convergence for molecule. Revise potential. Errors?
"""
#%% Imports and constants

from functools import partial
import numpy as np

import methods.errors as err
import methods.local_energy as le
import methods.sampling as samp
import methods.pdfs as pdfs
import methods.minimisers as minimisers

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

H_N = len(hermite_coeffs)
x_example = np.linspace(-4, 4, 1000)

x_vals_le = []
local_energy_vals = []

for i in range(H_N):
    for j in range(5):
        x_vals_le_second_i, local_energy_vals_second_i = le.local_energy_qho_numerical(
                                                x_example, 0.01, hermite_coeffs, i, 0)
        x_vals_le.append(x_vals_le_second_i)
        local_energy_vals.append(local_energy_vals_second_i)

x_vals_le = np.array(x_vals_le)
local_energy_vals = np.array(local_energy_vals)

# Calculating the RMS error on these calculations:

optimal_method, optimal_stepsize, stepsize_error = err.error_calculation()

#%% 2.2 - QHO Sampling & Eingenvalues

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

#%% 3 - Hydrogen Ground State Optimising

# We minimise first with a simple gradient descent and after with a
# Quasi-Newton method. The latter takes a very long time.
theta_guess = 0.90

theta_optimal, theta_history, energy_history = minimisers.hydrogen_wavefunction_optimiser_gd(theta_guess, m=200, eps=1e-8)
print(f'The optimal theta value is: {theta_optimal}, with energy: {energy_history[-1]} for Quasi-Newton.')

#%% 4 - Hydrogen Molecule Optimising

## First we plot out some of the samplings:

theta = np.array([1.0, 1.0, 1.0])
bond_length = 2
num_samples = 500000
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

samples = samp.metropolis_hastings_3d(
    pdf=h2_pdf,
    start=start_pos,
    domain=domain_6d,
    stepsize=0.1,
    num_samples=num_samples,
    burnin_val=int(num_samples*0.1),
    dimensions=6
)
print(f"Generated {len(samples)} samples")
samp.plot_6d_samples(samples, bins=125)

#%% Now we optimise the theta values to minimise the energy:

theta_opt, e_opt, th_history, e_history = minimisers.h2_optimiser_gd_2(
    theta=[0.75, 0.7, 0.75],
    stepsize=0.15,
    bond_length=2,
    start=[0.1, 0, -0.7, -0.1, 0, 0.7],
    delta=0.02,
    num_samples=100000,
    alpha=0.1,
    m=100,            # ← Increase to allow more iterations
    eps=1e-2,         # ← Stricter gradient threshold (not parameter change)
    burnin_val=10000
)

# %%

# Test with different starting points
test_thetas = [
    [0.5, 0.5, 0.5],
    [1.2, 0.3, 0.8],
    [0.7, 0.6, 0.6],
    [1.0, 0.2, 1.0],
]

for theta_init in test_thetas:
    print(f"\nTesting from theta = {theta_init}")

    theta_opt, e_opt, th_history, e_history = minimisers.h2_optimiser_gd(
        theta=theta_init,
        stepsize=0.15,
        bond_length=2,
        start=[0.1, 0, -0.7, -0.1, 0, 0.7],
        delta=0.02,
        num_samples=200000,
        alpha=0.01,
        m=30,
        eps=1e-5,
        burnin_val=20000
    )

    print(f"  Final: theta={theta_opt}, E={e_opt:.6f}")
# %%
