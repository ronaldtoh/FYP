import numpy as np
from mpmath import gammainc
import scipy.integrate as integrate
from data_another_style import data


def abs_scatter(frequency, theta, tau, thickness):
    delta_n = (3) ** 0.5 - 1
    wavelength = 3e8 / frequency
    scatter = (
        delta_n * 4 * np.pi * tau * np.cos(np.radians(theta)) / wavelength
    ) ** 2 / thickness
    return scatter


def get_tav_THZ_old(theta: float, n: np.array, alpha: np.array, wave: np.array):
    """Get average transmission (When absorption is non-negligible)

    Args:
        theta (float): Degree of incidence (in degrees)
        n (np.array): Refractive Index array
        alpha (np.array): Absorption Coefficient array
        wave (np.array): Wavelength array
    """
    theta_r = np.radians(theta)
    kappa = wave * alpha / (4 * np.pi)
    cos_theta = np.cos(theta_r)
    sin_theta = np.sin(theta_r)

    if theta == 0:
        r = ((n - 1) ** 2 + kappa**2) / ((n + 1) ** 2 + kappa**2)
        return 1 - r

    # Common Expressions for psi expressions
    p1 = n**2 - kappa**2 - (sin_theta) ** 2
    p2 = 4 * n**2 * kappa**2

    # Calculating psi
    psi1 = (0.5 * (p1 + (p1**2 + p2) ** 0.5)) ** 0.5
    psi2 = (0.5 * (-p1 + (p1**2 + p2) ** 0.5)) ** 0.5

    # Calculate r_perpendicular
    r_per1 = (cos_theta - psi1) ** 2 + psi2**2
    r_per2 = (cos_theta + psi1) ** 2 + psi2**2
    r_per = r_per1 / r_per2

    # Calculate r_parallel
    r_par1 = ((n**2 - kappa**2) * cos_theta - psi1) ** 2
    r_par2 = (2 * n * kappa * cos_theta - psi2) ** 2
    r_par3 = ((n**2 - kappa**2) * cos_theta + psi1) ** 2
    r_par4 = (2 * n * kappa * cos_theta + psi2) ** 2
    r_par = (r_par1 + r_par2) / (r_par3 + r_par4)

    t_per = 1 - r_per
    t_par = 1 - r_par

    return 0.5 * (t_per + t_par)


def get_tav_THZ(alpha: float, ref_arr: np.array, absorption: np.array, wave: np.array):
    """Get average transmission (When absorption is non-negligible)

    Args:
        alpha (float): Degree of incidence (in degrees)
        n (np.array): Refractive Index array
        absorption (np.array): Absorption Coefficient array
        wave (np.array): Wavelength array
    """
    kappa_i = wave * absorption / (4 * np.pi)
    # print(kappa)

    if alpha == 0:
        r = ((ref_arr - 1) ** 2 + kappa_i**2) / ((ref_arr + 1) ** 2 + kappa_i**2)
        return 1 - r

    def txnn(theta, n, kappa):
        # Common Expressions for psi expressions
        theta_r = np.radians(theta)
        cos_theta = np.cos(theta_r)
        sin_theta = np.sin(theta_r)
        p1 = n**2 - kappa**2 - (sin_theta) ** 2
        p2 = 4 * n**2 * kappa**2

        # Calculating psi
        psi1 = (0.5 * (p1 + (p1**2 + p2) ** 0.5)) ** 0.5
        psi2 = (0.5 * (-p1 + (p1**2 + p2) ** 0.5)) ** 0.5

        # Calculate r_perpendicular
        r_per1 = (cos_theta - psi1) ** 2 + psi2**2
        r_per2 = (cos_theta + psi1) ** 2 + psi2**2
        r_per = r_per1 / r_per2

        # Calculate r_parallel
        r_par1 = ((n**2 - kappa**2) * cos_theta - psi1) ** 2
        r_par2 = (2 * n * kappa * cos_theta - psi2) ** 2
        r_par3 = ((n**2 - kappa**2) * cos_theta + psi1) ** 2
        r_par4 = (2 * n * kappa * cos_theta + psi2) ** 2
        r_par = (r_par1 + r_par2) / (r_par3 + r_par4)

        t_per = 1 - r_per
        t_par = 1 - r_par

        return 0.5 * (t_per + t_par)

    output_arr = []

    for i, j in zip(ref_arr, kappa_i):
        tav_num = integrate.quad(txnn, 0, np.sin(alpha) ** 2, args=(i, j))
        tav = tav_num[0] / np.sin(alpha) ** 2
        output_arr.append(tav)

    return np.array(output_arr)


def get_tav(theta, refractive_index_arr: np.array):

    s = len(refractive_index_arr)
    theta_r = np.radians(theta)
    r2 = refractive_index_arr**2
    rp = r2 + 1
    rm = r2 - 1

    a = (refractive_index_arr + 1) ** 2 / 2
    k = -(rm**2) / 4
    ds = np.sin(theta_r)

    if theta_r == 0:
        return 4 * np.divide(refractive_index_arr, a * 2)
    elif theta_r == np.pi / 2:
        b1 = np.zeros(s)
    else:
        b1 = np.sqrt((ds**2 - rp / 2) ** 2 + k)
    b2 = ds**2 - rp / 2
    b = b1 - b2
    ts = ((k**2) / (6 * b**3) + k / b - b / 2) - (
        ((k**2) / (6 * a**3)) + k / a - a / 2
    )

    tp1 = (-2 * r2 * (b - a)) / (rp**2)
    tp2 = (-2 * r2 * rp) * np.log(b / a) / (rm**2)
    tp3 = r2 * ((b**-1) - (a**-1)) / 2
    tp4 = (
        16
        * r2**2
        * (r2**2 + 1)
        * np.log((2 * rp * b - rm**2) / (2 * rp * a - rm**2))
        / (rp**3 * rm**2)
    )
    tp5 = (
        16
        * r2**3
        * ((2 * rp * b - rm**2) ** -1 - (2 * rp * a - rm**2) ** -1)
        / (rp**3)
    )

    tp = tp1 + tp2 + tp3 + tp4 + tp5

    return (ts + tp) / (2 * ds**2)


def transfer_matrix(d, n, k, w):
    """
    Calculates the transfer matrix for a plate with given thickness, refractive index, and absorption coefficient
    at a given frequency.

    Arguments:
    d -- thickness of the plate in meters
    n -- refractive index of the plate
    k -- absorption coefficient of the plate in 1/meters
    w -- frequency of the incident light in radians/second

    Returns:
    The transfer matrix for the plate
    """
    c = 3e8  # speed of light in m/s
    alpha = k
    beta = n * np.sqrt(1 - (k / n) ** 2)
    gamma = np.exp(-2j * alpha * d)
    delta = np.cos(beta * d) - 1j * (alpha / beta) * np.sin(beta * d)
    A = np.array(
        [[np.exp(1j * beta * d), 0], [0, np.exp(-1j * beta * d)]], dtype=np.complex
    )
    B = np.array(
        [
            [np.cos(beta * d), 1j * (beta / alpha) * np.sin(beta * d)],
            [1j * (alpha / beta) * np.sin(beta * d), np.cos(beta * d)],
        ],
        dtype=np.complex,
    )
    C = np.array([[gamma, 0], [0, gamma]], dtype=np.complex)
    D = np.array(
        [
            [delta, 1j * (beta / alpha) * np.sin(beta * d)],
            [1j * (alpha / beta) * np.sin(beta * d), delta],
        ],
        dtype=np.complex,
    )
    T = np.matmul(np.matmul(A, B), np.matmul(C, D))
    return T


def reflectance_transmittance(d, n, k, w, theta):
    """
    Calculates the reflectance and transmittance of a single plate with given thickness, refractive index,
    and absorption coefficient at a given frequency and angle of incidence using the transfer matrix method.

    Arguments:
    d -- thickness of the plate in meters
    n -- refractive index of the plate
    k -- absorption coefficient of the plate in 1/meters
    w -- frequency of the incident light in radians/second
    theta -- angle of incidence in radians

    Returns:
    The reflectance and transmittance of the single plate
    """
    T = transfer_matrix(d, n, k, w)
    r = T[1, 0] / T[0, 0]
    t = 1 / T[0, 0]
    R = np.abs(r) ** 2

    T = np.abs(t) ** 2

    return R, T


def reflectance_transmittance_2plates(d1, d2, n1, n2, k1, k2, w):
    """
    Calculates the reflectance and transmittance of two stacked plates with given thickness, refractive index,
    and absorption coefficient at a given frequency using the transfer matrix method.

    Arguments:
    d1 -- thickness of the first plate in meters
    d2 -- thickness of the second plate in meters
    n1 -- refractive index of the first plate
    n2 -- refractive index of the second plate
    k1 -- absorption coefficient of the first plate in 1/meters
    k2 -- absorption coefficient of the second plate in 1/meters
    w -- frequency of the incident light in radians/second

    Returns:
    The reflectance and transmittance of the two stacked plates
    """
    T1 = transfer_matrix(d1, n1, k1, w)
    T2 = transfer_matrix(d2, n2, k2, w)
    T = np.matmul(T2, T1)
    r = T[1, 0] / T[0, 0]
    t = 1 / T[0, 0]
    R = np.abs(r) ** 2
    T = np.abs(t) ** 2 * (
        n2
        * np.cos(w * np.sqrt(n2**2 - k2**2) * d2)
        / (n1 * np.cos(w * np.sqrt(n1**2 - k1**2) * d1))
    )
    return R, T


def get_tav_test(alpha, ref_idx: np.array):
    def txn(theta, n):
        theta_r = np.radians(theta)
        s2 = np.sin(theta_r) ** 2

        part1 = (1 - s2) ** 0.5
        part2 = (n**2 - s2) ** 0.5
        t_per_num = 4 * part1 * part2
        t_per_denom = (part1 + part2) ** 2
        t_per = t_per_num / t_per_denom

        t_par_num = 4 * n**2 * part1 * part2
        t_par_denom = (n**2 * part1 + part2) ** 2
        t_par = t_par_num / t_par_denom

        return 0.5 * (t_per + t_par)

    output_arr = []
    for i in ref_idx:
        tav_num = integrate.quad(txn, 0, np.sin(alpha) ** 2, args=(i))
        tav = tav_num[0] / np.sin(alpha) ** 2
        output_arr.append(tav)

    return np.array(output_arr)


def get_integral(k):

    if np.all(k <= 4):
        x = 0.5 * k - 1
        y = -3.60311230482612224e-13 * x + 3.46348526554087424e-12
        y = y * x - 2.99627399604128973e-11
        y = y * x + 2.57747807106988589e-10
        y = y * x - 2.09330568435488303e-9
        y = y * x + 1.59501329936987818e-8
        y = y * x - 1.13717900285428895e-7
        y = y * x + 7.55292885309152956e-7
        y = y * x - 4.64980751480619431e-6
        y = y * x + 2.63830365675408129e-5
        y = y * x - 1.37089870978830576e-4
        y = y * x + 6.47686503728103400e-4
        y = y * x - 2.76060141343627983e-3
        y = y * x + 1.05306034687449505e-2
        y = y * x - 3.57191348753631956e-2
        y = y * x + 1.07774527938978692e-1
        y = y * x - 2.96997075145080963e-1
        y = y * x + 8.64664716763387311e-1
        y = y * x + 7.42047691268006429e-1
        return y - np.log(k)

    elif np.all(k >= 85):
        return 0

    else:
        x = 14.5 / (k + 3.25) - 1
        y = -1.62806570868460749e-12 * x - 8.95400579318284288e-13
        y = y * x - 4.08352702838151578e-12
        y = y * x - 1.45132988248537498e-11
        y = y * x - 8.35086918940757852e-11
        y = y * x - 2.13638678953766289e-10
        y = y * x - 1.10302431467069770e-9
        y = y * x - 3.67128915633455484e-9
        y = y * x - 1.66980544304104726e-8
        y = y * x - 6.11774386401295125e-8
        y = y * x - 2.70306163610271497e-7
        y = y * x - 1.05565006992891261e-6
        y = y * x - 4.72090467203711484e-6
        y = y * x - 1.95076375089955937e-5
        y = y * x - 9.16450482931221453e-5
        y = y * x - 4.05892130452128677e-4
        y = y * x - 2.14213055000334718e-3
        y = y * x - 1.06374875116569657e-2
        y = y * x - 8.50699154984571871e-2
        y = y * x + 9.23755307807784058e-1
        return np.exp(-k) * y / k


# test = np.array([[3], [4], [5], [7], [8], [7]])

# test = np.array([3, 4, 5, 7, 8, 7])


if __name__ == "__main__":
    # test = [2, 15, 1, 3.33, 0.0005]
    # for i in test:
    #     print(f"i: {i}")
    #     print(get_integral(i))
    #     print(gammainc(0, i))
    data2 = np.array(data)
    n = data2[:, 1]

    # ans = get_tav(10, n)
    # ans2 = get_tav_THZ_v2(10, n)

    # print(ans2)
