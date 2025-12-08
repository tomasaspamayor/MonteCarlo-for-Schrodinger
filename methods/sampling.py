"""
Define the two sampling algorithms used and allow their implementations both
in 1D and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt

import methods.pdfs as pdfs

## Two different sampling methods. The Metropolis-Hastings in 3D allows virtually
## any dimension of samples. Must be specified in args.

def rejection(pdf, cf, start, end, num_samples, max_iterations, constant=1, m=10):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func): A callable PDF function (1D).
    start (float): Startpoint of the sample array.
    end (float): Endpoint of the sample array.
    num_samples (int): The desired number of points in the final sample.
    N (int): Upper bound for the sampling loop.
    constant (bool): Option to use a constant CF.
    M (float): Scaling of the CF. Only used in constant CFs.

    Returns:
    np.array: The calculated samples following the PDF.
    """

    if constant == 1 and m is None:
        x_trial = np.linspace(start, end, 1000)
        pdf_max = np.max(pdf(x_trial))
        q = 1 / (end - start)
        m = pdf_max / q * 1.1 # Buffering.

    if constant == 1:
        q = 1 / (end - start)
        constant_cf = m * q

    distribution = []

    iterations = 0
    while len(distribution) < num_samples and iterations < max_iterations:
        x_val = np.random.uniform(start, end)
        pdf_val = pdf(x_val)

        if constant == 1:
            cf_val = constant_cf
        else:
            cf_val = cf(x_val)

        if pdf_val > cf_val:
            raise ValueError('The CF must enclose the PDF for all values (repick M).')

        u = np.random.uniform(0, 1)
        if u < (pdf_val / cf_val):
            distribution.append(x_val)

        iterations += 1

    samples = np.array(distribution)

    return samples

def rejection_3d(pdf, cf, boundaries, num_samples, max_iterations, constant=1, m=None):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    boundaries (list) - List of the boundaries for each dimension
    num_samples (int) - The desired number of points in the final sample.
    max_iterations (int) - Maximum number of iterations.
    constant (bool) - Option to use a constant CF.
    M (float) - Scaling of the CF. Only used in constant CFs.

    Returns:
    np.array: The calculated samples following the PDF.
    """

    x_start = boundaries[0][0]
    x_end = boundaries[0][1]
    y_start = boundaries[1][0]
    y_end = boundaries[1][1]
    z_start = boundaries[2][0]
    z_end = boundaries[2][1]

    points_grid = 15
    x_test = np.linspace(x_start, x_end, points_grid)
    y_test = np.linspace(y_start, y_end, points_grid)
    z_test = np.linspace(z_start, z_end, points_grid)

    if constant == 1 and m is None:
        pdf_max = 0
        for x in x_test:
            for y in y_test:
                for z in z_test:
                    pdf_val = pdf([x, y, z])
                    pdf_max = max(pdf_max, pdf_val)

        q = 1 / ( (x_end - x_start) * (y_end - y_start) * (z_end - z_start) )
        m = pdf_max / q * 1.1 # Buffering.
        constant_cf = m * q
    else:
        q = 1 / ( (x_end - x_start) * (y_end - y_start) * (z_end - z_start) )
        constant_cf = m * q

    distribution = []

    iterations = 0
    while len(distribution) < num_samples and iterations < max_iterations:
        x_val = np.random.uniform(x_start, x_end)
        y_val = np.random.uniform(y_start, y_end)
        z_val = np.random.uniform(z_start, z_end)
        point_val = [x_val, y_val, z_val]
        pdf_val = pdf(point_val)

        if constant == 1:
            cf_val = constant_cf
        else:
            cf_val = cf([x_val, y_val, z_val])

        if pdf_val > cf_val:
            raise ValueError('The CF must enclose the PDF for all values (repick M).')

        u = np.random.uniform(0, 1)
        if u < (pdf_val / cf_val):
            distribution.append(point_val)

        iterations += 1

    samples = np.array(distribution)

    return samples

def metropolis_hastings(pdf, start, domain, stepsize, num_samples, burnin_val=1000):
    """Generate an array of sample points which follow a PDF of choice. This is computed
    following the Metropolis-Hastings (MCMC) algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    start (float) - Startpoint in the range.
    stepsize (float) - Value of move by iteration.
    num_samples (int) - The desired number of points in the final sample.
    burnin_val (int) - The needed number of rejected samples by the algorithm.

    Returns:
    np.array: The calculated samples following the PDF.
    """
    state = start
    samples = []

    def proposal(current_state):
        candidate = current_state + np.random.normal(0, stepsize)
        return candidate

    iterations = num_samples + burnin_val
    for i in range(iterations):
        candidate = proposal(state)
        ratio = pdf(candidate) / pdf(state)
        alpha = np.min([1, ratio])
        u = np.random.random()

        if u <= alpha:
            state = candidate
        if i >= burnin_val:
            samples.append(state)

    samples = np.array(samples)

    return samples

def metropolis_hastings_3d(pdf, start, domain, stepsize, num_samples, burnin_val=1000, dimensions=3):
    """Generate an array of sample points which follow a PDF of choice. This is computed
    following the Metropolis-Hastings (MCMC) algorithm.

    Args:
    PDF (func) - A callable PDF function (3D).
    start (list) - Startpoint in the range (for each dimension).
    stepsize (list) - Value of move by iteration (for each dimension).
    num_samples (int) - The desired number of points in the final sample.
    burnin_val (int) - The needed number of rejected samples by the algorithm.

    Returns:
    np.array: The calculated samples following the PDF.
    """
    state = np.array(start)
    samples = []

    if np.isscalar(stepsize):
        stepsize = np.full(dimensions, stepsize)
    else:
        stepsize = np.array(stepsize)

    def proposal(current_state):
        candidate = current_state + np.random.normal(0, stepsize, dimensions)
        return candidate

    iterations = num_samples + burnin_val
    for i in range(iterations):
        candidate = proposal(state)

        valid = True
        for d in range(dimensions):
            bottom, top = domain[d]
            if candidate[d] < bottom or candidate[d] > top:
                valid = False
                break

        if not valid:
            if i >= burnin_val:
                samples.append(state.copy())
            continue

        ratio = pdf(candidate) / pdf(state)
        alpha = np.min([1, ratio])
        u = np.random.random()

        if u <= alpha:
            state = candidate.copy()
        if i >= burnin_val:
            samples.append(state)

    samples = np.array(samples)

    return samples

def samplings_h2_molecule(bond_length, initial_point, theta, domain, stepsize, num_samples, burnin_val):
    """
    Sampling the H_2 molecule's PDF.
    
    Args:
    bond_length (float): Description
    initial_point (list): Description
    theta (list): Description
    domain (list): Description
    stepsize (float): Description
    num_samples (int): Description
    burnin_val (int): Description

    Returns:
    list: The samples.
    """
    q1 = np.array([0, 0, -bond_length / 2])
    q2 = np.array([0, 0, bond_length / 2])

    def h2_6d(point_6d):
        r1 = point_6d[:3]
        r2 = point_6d[3:]
        wf_val = pdfs.wavefunction_hydrogen_molecule(r1, r2, theta, q1, q2,
                                                     bond_length=bond_length)
        return abs(wf_val) ** 2

    if domain is None:
        domain_size = 5.0
        domain = [(-domain_size, domain_size) for _ in range(6)]
    elif np.isscalar(domain):
        domain = [(-domain, domain) for _ in range(6)]

    if initial_point is None:
        initial_point = np.array([0.2, 0, -0.3,
                                  -0.2, 0, 0.3])

    samples_6d = metropolis_hastings_3d(h2_6d, initial_point, domain,
                      stepsize, num_samples,
                      burnin_val=burnin_val,
                      dimensions=6)
    return samples_6d

## Plotting methods for the encountered cases.

def plot_samples(pdf, x_vals, samples, bins, method_num):
    """
    Plot the sampled array in a histogram with the original PDF.

    Args:
    PDF (func) - The PDF function which was sampled (1D).
    x_vals (list) - The start and end point of the sampling.
    bins (int) - Number of bins in the histogram.
    method_num (bool) - Gives the label for the correct method:
                        0 --> Rejection Method
                        1 --> Metropolis-Hastings Algorithm

    Returns:
    plt.plot: The generated plot.
    """
    x_array = np.linspace(x_vals[0], x_vals[1], 1000)
    y_array = pdf(x_array)

    if method_num == 0:
        method = "Rejection Method"
    else:
        method = "Metropolis-Hastings Algorithm"

    plt.hist(samples, bins, density=True, label='Samples')
    plt.plot(x_array, y_array, label='PDF')
    plt.title(f'Sampling of PDF ({method})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_3d_samples(samples, bins, method_num):
    """
    Plot 2D projections of 3D samples.
    
    Args:
    samples: sampled values.
    bins: number of bins for histograms
    method_num: 0 for Rejection, 1 for Metropolis-Hastings

    Returns:
    plt.plot: The x-y projection of the samples (tight).
    plt.plot: The x-z projection of the samples (tight).
    plt.plot: The y-z projection of the samples (tight).
    """

    if method_num == 0:
        method = "Rejection Method"
    else:
        method = "Metropolis-Hastings Algorithm"

    samples_t = samples.T

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.hist2d(samples_t[0], samples_t[1], bins=bins, density=True)
    plt.colorbar(label='Density')
    plt.xlabel('X samples')
    plt.ylabel('Y samples')
    plt.title(f'X-Y Projection - {method}')

    plt.subplot(132)
    plt.hist2d(samples_t[0], samples_t[2], bins=bins, density=True)
    plt.colorbar(label='Density')
    plt.xlabel('X samples')
    plt.ylabel('Z samples')
    plt.title(f'X-Z Projection - {method}')

    plt.subplot(133)
    plt.hist2d(samples_t[1], samples_t[2], bins=bins, density=True)
    plt.colorbar(label='Density')
    plt.xlabel('Y samples')
    plt.ylabel('Z samples')
    plt.title(f'Y-Z Projection - {method}')

    plt.tight_layout()
    plt.show()

def plot_6d_samples(samples, bins=70):
    """
    Plot only the combined ZX histogram (bottom left plot from original).
    Removed nuclei markings and other plots.

    Args:
    samples: Sampled values.
    bins: Number of bins for histograms.

    Returns:
    plt.plot: The z-x projection of the samples.
    """
    bins = int(bins)

    electron1_xz = samples[:, [0, 2]]
    electron2_xz = samples[:, [3, 5]]
    plt.figure(figsize=(8, 6))

    all_x = np.concatenate([electron1_xz[:, 0], electron2_xz[:, 0]])
    all_z = np.concatenate([electron1_xz[:, 1], electron2_xz[:, 1]])

    h3 = plt.hist2d(all_z, all_x, bins=bins, cmap='viridis', density=True)

    plt.colorbar(h3[3], label='Probability Density')

    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Electron Wavefunction - H2 Molecule')

    plt.tight_layout()
    plt.show()
