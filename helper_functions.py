import numpy as np
from mpmath import gammainc


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
        b1 = np.zeros(
            s
        )  ####################################probably wrong syntax#################
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
    test = [2, 15, 1, 3.33, 0.0005]
    for i in test:
        print(f"i: {i}")
        print(get_integral(i))
        print(gammainc(0, i))
