import numpy as np
import matplotlib.pyplot as plt
import differentiators as diff
import polynomials as poly

def err_finite_diff(range_stepsizes, num_stepsizes, range_val, samples_num, level, coeffs, method):
    """
    Compute and plot the FDM's error with respect to the analytical derivative.

    Args:
    stepsize - (float): The stepsize in the FD method.
    range_val - (list): The beggining and end points of the independent variable.
    level - (int): The order of the Hermite polynomial to be differentiated.
    coeffs - (list): The coefficients in increasing order of monomial.
    method: defines if use fourth trunaction (==1) or not (else).
    """
    rms_fdm_hermites = []
    h = len(coeffs)
    for i in range(h):
        stepsizes_array = np.linspace(range_stepsizes[0], range_stepsizes[-1], num_stepsizes)
        n = len(stepsizes_array)

        samples = np.linspace(range_val[0], range_val[-1], samples_num)
        current_coeffs = coeffs[i]
        rms_fdm_list = []

        for j in range(n):
            sec_exact = diff.analytical_second_der(samples, current_coeffs, plot=False)
            if method == 1:
                x_vals, sec_fd, other = diff.fd_fourth(stepsizes_array[j], range_val, samples_num, current_coeffs)
                sec_exact = sec_exact[2:-2]
            else:
                x_vals, sec_fd, other = diff.fd_second(stepsizes_array[j], range_val, samples_num, current_coeffs)
                sec_exact = sec_exact[1:-1]

            rms = np.sqrt(np.mean((sec_fd - sec_exact) ** 2))
            rms_fdm_list.append(rms)

        rms_fdm = np.array(rms_fdm_list)
        rms_fdm_hermites.append(rms_fdm)

    plt.figure(figsize=(10, 6))
    for poly_order in range(h):
        plt.loglog(stepsizes_array, rms_fdm_hermites[poly_order], 
                  label=f'H_{poly_order}', marker='o', markersize=3)
    plt.grid()
    plt.xlabel('stepsize')
    plt.ylabel("RMS FDM")
    plt.title(f"RMS value of FDM with respect to stepsize")
    plt.show()

    return stepsizes_array, rms_fdm_list