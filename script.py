#%%
## See the script to solve the different questions with the methods defined
## in other files.

import math as math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import two_one.polynomials as poly
import two_one.errors as err
import two_one.local_energy as le
import two_two.sampling as samp
import two_two.pdfs as pdfs

hermite_coeffs = [
    [1],                       # n=0
    [0, 2],                    # n=1  
    [-2, 0, 4],                # n=2
    [0, -12, 0, 8],            # n=3
    [12, 0, -48, 0, 16],       # n=4
    [0, 120, 0, -160, 0, 32],  # n=5
    [-120, 0, 720, 0, -480, 0, 64]  ] # n=6
polys = len(hermite_coeffs)

#%% 2.1

# Numerical evalutation of the local energies with both 2nd and 4th order truncation:
num_hermite = len(hermite_coeffs)
x_example = np.linspace(-4, 4, 1000)

x_vals_le = []
local_energy_vals = []

for i in range(num_hermite):
    for j in range(5):
        x_vals_le_second_i, local_energy_vals_second_i = le.local_energy(x_example, 0.01,
                                                        hermite_coeffs, i, i, plot=True)
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

#%% 2.2

samples_gaussian_r = samp.rejection(pdfs.normalized_gaussian, 1, -1, 1, 100000, 1e8)
samples_exponential_r = samp.rejection(pdfs.normalized_exponential, 1, 0, 1, 100000, 1e8)

samp.plot_samples(pdfs.normalized_gaussian, [-1, 1], samples_gaussian_r, 50, 0)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_r, 50, 0)

samples_gaussian_mh = samp.metropolis_hastings(pdfs.normalized_gaussian, 0, [-1, 1], 0.5, 100000)
samples_exponential_mh = samp.metropolis_hastings(pdfs.normalized_exponential, 0.5, [0, 1], 0.5, 100000)
samp.plot_samples(pdfs.normalized_gaussian, [-1, 1], samples_gaussian_mh, 50, 1)
samp.plot_samples(pdfs.normalized_exponential, [0, 1], samples_exponential_mh, 50, 1)

samples_gaussian_r_3d = samp.rejection_3d(pdfs.gaussian_3d, 0, [[-0.5,0.5], [-0.5,0.5],
                                                [-0.5,0.5]], 1000000, 10000000, m=None)
samples_gaussian_mh_3d = samp.metropolis_hastings_3d(pdfs.gaussian_3d, [0, 0, 0], [[-0.5, 0.5],
                                                       [-0.5, 0.5], [-0.5, 0.5]], 0.5, 1000000)

samp.plot_3d_samples(samples_gaussian_r_3d, 40, 0)
samp.plot_3d_samples(samples_gaussian_mh_3d, 40, 1)


#%% Passing through the local energy function:
wf_samples = []
for n in range(polys):
    pdf_qo = partial(pdfs.wf_pdf, n=n, coeffs=hermite_coeffs)
    samples_wf_r = samp.rejection(pdf_qo, 0, -3, 3.1, 10000, 1000000, m=None)
    #samp.plot_samples(pdf_qo, [-5,5.1], samples_wf_r, 150, 0)
    wf_samples.append(samples_wf_r)

energies = []
for i in range(polys):
    x_values, local_energies = le.local_energy(wf_samples[i], 0.01, hermite_coeffs, i, 10)
    le.plot_local_energies(local_energies, 200, i, 8)
    energy_i = np.sum(local_energies) / len(local_energies)
    energies.append(energy_i)


# %%
