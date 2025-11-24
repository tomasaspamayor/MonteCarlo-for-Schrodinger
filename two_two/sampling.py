import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def rejection_sampling(pdf, cf, start, end, num_samples, max_iterations, constant=1, m=10):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    start (float) - Startpoint of the sample array.
    end (float) - Endpoint of the sample array.
    num_samples (int) - The desired number of points in the final sample.
    N (int) - Upper bound for the sampling loop.
    constant (bool) - Option to use a constant CF.
    M (float) - Scaling of the CF. Only used in constant CFs.
    """

    # Calculate M automatically if not provided:
    if constant == 1 and m is None:
        x_trial = np.linspace(start, end, 1000)
        pdf_max = np.max(pdf(x_trial))
        q = 1 / (end - start)
        m = pdf_max / q * 1.1 # Buffering.
        print(f"Calculated m {m}")

    if constant == 1:
        # Generate the simplest proposal function, a constant:
        q = 1 / (end - start)
        constant_cf = m * q

    # Create an empty list for the distribution:
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

    if len(samples) < num_samples:
        raise ValueError(f'Only generated {len(samples)} / {num_samples} after {iterations} iterations.')

    return samples

def rejection_sampling_3d(pdf, cf, start, end, num_samples, n, constant=1, m=10):
    """"
    Generate an array of sample points which follow a PDF of choice. This is computed
    following the acceptance-rejection algorithm.

    Args:
    PDF (func) - A callable PDF function (1D).
    start (list) - Startpoint for each dimension.
    end (float) - Endpoint for each dimension.
    num_samples (int) - The desired number of points in the final sample.
    N (int) - Upper bound for the sampling loop.
    constant (bool) - Option to use a constant CF.
    M (float) - Scaling of the CF. Only used in constant CFs.
    """

    x_start, x_end = start[0], end[0]
    y_start, y_end = start[1], end[1]
    z_start, z_end = start[2], end[2]

    if constant == 1:
        # Generate the simplest proposal function, a constant:
        q = 1 / ((x_end - x_start) * (y_end - y_start) * (z_end - z_start))
        constant_cf = m * q

    # Create an empty list for the distribution:
    distribution = []

    for i in range(n):
        x_val = np.random.uniform(x_start, x_end)
        y_val = np.random.uniform(y_start, y_end)
        z_val = np.random.uniform(z_start, z_end)

        pdf_val = pdf(x_val, y_val, z_val)

        if constant == 1:
            cf_val = constant_cf
        else:
            cf_val = cf(x_val, y_val, z_val)

        if pdf_val > cf_val:
            raise ValueError('The CF must enclose the PDF for all values (repick M)')

        u = np.random.uniform(0, 1)
        if u < (pdf_val / cf_val):
            distribution.append([x_val, y_val, z_val])

        if len(distribution) == num_samples:
            break

    samples = np.array(distribution)

    if len(samples) < num_samples:
        raise ValueError('Not all samples were generated.')

    return samples

def plot_samples(pdf, x_vals, samples, bins):
    """
    Plot the sampled array in a histogram with the original PDF.

    Args:
    PDF (func) - The PDF function which was sampled.
    x_vals (list) - The start and end point of the sampling.
    bins (int) - Number of bins in the histogram.
    """
    x_array = np.linspace(x_vals[0], x_vals[1], 1000)
    y_array = pdf(x_array)

    plt.hist(samples, bins, density=True, label='Samples')
    plt.plot(x_array, y_array, label='PDF')
    plt.title('Sampling of PDF (Rejection Method)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

## I will also define some normalised PDFs to check the methods:

def normalized_gaussian(x, sigma=0.2):
    """Gaussian normalized over [-1, 1]"""
    # Compute normalization factor
    norm_factor = norm.cdf(1, scale=sigma) - norm.cdf(-1, scale=sigma)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2)) / norm_factor

def normalized_exponential(x, lambd=2.0):
    """Exponential decay normalized over [0, 1]"""
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    # CORRECTED normalization factor
    norm_factor = 1 - np.exp(-lambd * 1)  # This was the error!
    return (lambd * np.exp(-lambd * x)) / norm_factor
