import numpy as np

's-polarized light'
def rs(n1, n2, theta_i, theta_t):
    return (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (
        n1 * np.cos(theta_i) + n2 * np.cos(theta_t)
    )


def ts(n1, n2, theta_i, theta_t):
    return (2 * n1 * np.cos(theta_i)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))

'p-polarized light'
def rp(n1, n2, theta_i, theta_t):
    return (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (
        n2 * np.cos(theta_i) + n1 * np.cos(theta_t)
    )


def tp(n1, n2, theta_i, theta_t):
    return (2 * n1 * np.cos(theta_i)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
