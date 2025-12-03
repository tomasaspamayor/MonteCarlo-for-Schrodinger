import numpy as np
import matplotlib.pyplot as plt
import methods.differentiators as diff

def err_finite_diff(range_stepsizes, num_stepsizes, range_val, samples_num, coeffs, method, plot=True):
    """
    Compute and plot the FDM's error with respect to the analytical derivative.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    coeffs - (list): The coefficients in increasing order of monomial.
    method: Defines the truncation used. In increasing integer order, they are:
            0 -> Second Order Truncation
            1 -> Fourth Order Truncation
            2 -> Sixth Order Truncation
            else -> Eighth Order Truncation
            4 -> Tenth Order Truncation
    """
    rms_fdm_hermites = []
    h = len(coeffs)
    for poly_order in range(h):
        stepsizes_array = np.linspace(range_stepsizes[0], range_stepsizes[-1], num_stepsizes)
        n = len(stepsizes_array)

        current_coeffs = coeffs[poly_order]
        rms_fdm_list = []

        for step_idx in range(n):

            if method == 0:
                samples_inner, sec_fd, _ = diff.cdm_samples_second(stepsizes_array[step_idx],
                                        range_val, samples_num, current_coeffs)
                sec_exact = diff.analytical_second_der(samples_inner, coeffs, poly_order, plot=False)

            elif method == 1:
                samples_inner, sec_fd, _ = diff.cdm_samples_fourth(stepsizes_array[step_idx],
                                        range_val, samples_num, current_coeffs)
                sec_exact = diff.analytical_second_der(samples_inner, coeffs, poly_order, plot=False)

            elif method == 2:
                samples_inner, sec_fd, _ = diff.cdm_samples_sixth(stepsizes_array[step_idx],
                                        range_val, samples_num, current_coeffs)
                sec_exact = diff.analytical_second_der(samples_inner, coeffs, poly_order, plot=False)

            elif method == 4:
                samples_inner, sec_fd, _ = diff.cdm_samples_tenth(stepsizes_array[step_idx],
                                    range_val, samples_num, current_coeffs, polynomial=1)
                sec_exact = diff.analytical_second_der(samples_inner, coeffs, poly_order, plot=False)

            else: # Keeping the 8th as the default as it is the optimal.
                samples_inner, sec_fd, _ = diff.cdm_samples_eighth(stepsizes_array[step_idx],
                                    range_val, samples_num, current_coeffs)
                sec_exact = diff.analytical_second_der(samples_inner, coeffs, poly_order, plot=False)

            rms = np.sqrt(np.mean((sec_fd - sec_exact) ** 2))
            rms_fdm_list.append(rms)

        rms_fdm = np.array(rms_fdm_list)
        rms_fdm_hermites.append(rms_fdm)

    if plot is True:
        plt.figure(figsize=(10, 6))
        for poly_order in range(h):
            plt.loglog(stepsizes_array, rms_fdm_hermites[poly_order],
                    label=f'H_{poly_order}', marker='o', markersize=3)
        plt.grid()
        plt.xlabel('Step size')
        plt.ylabel("RMS Error")
        plt.title(f"RMS Error vs Step Size - {['Second', 'Fourth', 'Sixth', 'Eighth', 'Tenth'][method]} Order FD")
        plt.legend()
        plt.show()

    return stepsizes_array, rms_fdm_hermites

def plot_err_methods(range_stepsizes, num_stepsizes, range_val, samples_num, coeffs):
    """
    Plot the errors in the different finite difference method truncations
    for a same function, log-log.

    Args:
    stepsizes (list) - Array of the arrays for the stepsizes of decresing order
                       of truncation.
    errors (list) - As above, for the errors.
    poly_order (int) - Order of the Hermite polynomial being analised.
    """
    tenth_steps, tenth_err = err_finite_diff(range_stepsizes, num_stepsizes,
                         range_val, samples_num, coeffs, 4,  plot=False)
    eigth_steps, eigth_err = err_finite_diff(range_stepsizes, num_stepsizes,
                         range_val, samples_num, coeffs, 3,  plot=False)
    sixth_steps, sixth_err = err_finite_diff(range_stepsizes, num_stepsizes,
                          range_val, samples_num, coeffs, 2, plot=False)
    fourth_steps, fourth_err = err_finite_diff(range_stepsizes, num_stepsizes,
                          range_val, samples_num, coeffs, 1, plot=False)
    second_steps, second_err = err_finite_diff(range_stepsizes, num_stepsizes,
                          range_val, samples_num, coeffs, 0, plot=False)

    n = len(tenth_err)

    for i in range(n):
        plt.loglog(tenth_steps, tenth_err[i], label='10th order')
        plt.loglog(eigth_steps, eigth_err[i], label='8th order')
        plt.loglog(sixth_steps, sixth_err[i], label='6th order')
        plt.loglog(fourth_steps, fourth_err[i], label='4th order')
        plt.loglog(second_steps, second_err[i], label='2nd order')

        plt.grid()
        plt.xlabel('stepsize')
        plt.ylabel("RMS FDM")
        plt.title(f"RMS value of FDM with respect to stepsize, H_{i}")
        plt.legend()
        plt.show()
