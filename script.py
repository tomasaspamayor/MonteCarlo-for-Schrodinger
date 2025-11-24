#%%
## See the script to solve the different questions with the methods defined
## in other files.

import numpy as np
import matplotlib.pyplot as plt

import two_one.errors as err
import two_one.local_energy as le
import two_two.sampling as samp

#%% 2.1
hermite_coeffs = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [-2, 0, 4, 0, 0], [0, -12, 0, 8, 0],
                  [12, 0, -48, 0, 16]]

# Numerical evalutation of the local energies with both 2nd and 4th order truncation:
n_hermite = len(hermite_coeffs)

x_vals_second_le = []
local_energy_second_vals = []
for i in range(n_hermite):
    x_vals_le_second_i, local_energy_vals_second_i = le.local_energy(0.01, [-1, 1], 1000, hermite_coeffs, i, 0, plot=True)
    x_vals_second_le.append(x_vals_le_second_i)
    local_energy_second_vals.append(local_energy_vals_second_i)

x_vals_second_le = np.array(x_vals_second_le)
local_energy_second_vals = np.array(local_energy_second_vals)

x_vals_fourth_le = []
local_energy_fourth_vals = []
for i in range(n_hermite):
    x_vals_le_fourth_i, local_energy_vals_fourth_i = le.local_energy(0.01, [-1, 1], 1000, hermite_coeffs, i, 1, plot=True)
    x_vals_fourth_le.append(x_vals_le_fourth_i)
    local_energy_fourth_vals.append(local_energy_vals_fourth_i)

x_vals_fourth_le = np.array(x_vals_fourth_le)
local_energy_fourth_vals = np.array(local_energy_fourth_vals)

# Calculating the RMS error on these calculations, for 2nd order and 4th order truncations:

stepsize_vals_second = []
err_vals_second = []

stepsize_vals_second_i, err_vals_second_i = err.err_finite_diff([1e-5, 0.01], 500,
                                            [-1, 1], 1000, 0, hermite_coeffs, 0)
stepsize_vals_second.append(stepsize_vals_second_i)
err_vals_second.append(err_vals_second_i)

stepsize_vals_second = np.array(stepsize_vals_second)
err_vals_second = np.array(err_vals_second)

stepsize_vals_fourth = []
err_vals_fourth = []

stepsize_vals_fourth_i, err_vals_fourth_i = err.err_finite_diff([1e-5, 0.01], 500,
                                            [-1, 1], 1000, 0, hermite_coeffs, 1)
stepsize_vals_fourth.append(stepsize_vals_fourth_i)
err_vals_fourth.append(err_vals_fourth_i)

stepsize_vals_fourth = np.array(stepsize_vals_fourth)
err_vals_fourth = np.array(err_vals_fourth)

#%% 2.2

samples_gaussian = samp.rejection_sampling(samp.normalized_gaussian, 1, -1, 1, 100000, 1e8)
samples_exponential = samp.rejection_sampling(samp.normalized_exponential, 1, 0, 1, 100000, 1e8)

x_gauss_vals = np.linspace(-1, 1, 1000)
x_expo_vals = np.linspace(0, 1, 1000)
gauss_vals = samp.normalized_gaussian(x_gauss_vals)
expo_vals = samp.normalized_exponential(x_expo_vals)

plt.hist(samples_gaussian, 50, density=True, label='Generated Samples')
plt.plot(x_gauss_vals, gauss_vals, label='PDF')
plt.title('Gaussian PDF Sampling')
plt.legend()
plt.show()

plt.hist(samples_exponential, 50, density=True, label='Generated Samples')
plt.plot(x_expo_vals, expo_vals, label='PDF')
plt.title('Exponential PDF Sampling')
plt.legend()
plt.show()

# %%
