#%%
## See the script to solve the different questions with the methods defined
## in other files.

import numpy as np
from functools import partial

import two_one.errors as err
import two_one.local_energy as le
import two_two.sampling as samp
import two_two.pdfs as pdfs

hermite_coeffs = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [-2, 0, 4, 0, 0], [0, -12, 0, 8, 0],
                  [12, 0, -48, 0, 16]]

#%% 2.1


# Numerical evalutation of the local energies with both 2nd and 4th order truncation:
num_hermite = len(hermite_coeffs)

x_vals_second_le = []
local_energy_second_vals = []
x_vals_fourth_le = []
local_energy_fourth_vals = []

for i in range(num_hermite):
    x_vals_le_second_i, local_energy_vals_second_i = le.local_energy(0.01,
                                [-1, 1], 1000, hermite_coeffs, i, 0, plot=True)
    x_vals_second_le.append(x_vals_le_second_i)
    local_energy_second_vals.append(local_energy_vals_second_i)

for i in range(num_hermite):
    x_vals_le_fourth_i, local_energy_vals_fourth_i = le.local_energy(0.01,
                                [-1, 1], 1000, hermite_coeffs, i, 1, plot=True)
    x_vals_fourth_le.append(x_vals_le_fourth_i)
    local_energy_fourth_vals.append(local_energy_vals_fourth_i)

x_vals_second_le = np.array(x_vals_second_le)
local_energy_second_vals = np.array(local_energy_second_vals)
x_vals_fourth_le = np.array(x_vals_fourth_le)
local_energy_fourth_vals = np.array(local_energy_fourth_vals)

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

n = [0, 0, 0]
coeffs = hermite_coeffs
wf_3d_wrapped = partial(pdfs.wf_3d, n=n, coeffs=coeffs)

wf_samples_r = samp.rejection_3d(wf_3d_wrapped, 0, [0, 0, 0], [1, 1, 1], 1000, 10000000)
wf_samples_mh = samp.metropolis_hastings_3d(wf_3d_wrapped, [0, 0, 0], [[0, 1], [0, 1], [0, 1]], 0.5, 1000)

# %%
