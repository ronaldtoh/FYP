import numpy as np
from helper_functions import get_tav
import matplotlib.pyplot as plt
from scipy.special import exp1
from data_another_style import data

# import pandas as pd

"""
Plant leaf reflectance and transmittance are calculated from 400 nm to
2500 nm with the following parameters:
      - N   = leaf structure parameter
      - Cab = chlorophyll a+b content in µg/cm²
      - Cw  = equivalent water thickness in g/cm² or cm
      - Cm  = dry matter content in g/cm²
"""


class MultiPlateModel:
    def __init__(self, N: int, Cab, Cw, Cm):
        "Leaf Structure Parameter"
        self.N = N
        data2 = np.array(data)
        self.wavelength = data2[:, 0]
        refractive_index_array = data2[:, 1]
        k = (Cab * data2[:, 3] + Cw * data2[:, 4] + Cm * data2[:, 5]) / N + data2[:, 2]

        if np.all(k <= 0):
            k = 1
        else:
            k = (1 - k) * np.exp(-k) + (k**2) * exp1(k)

        t1 = get_tav(90, refractive_index_array)
        t2 = get_tav(40, refractive_index_array)

        x1 = 1 - t1
        x2 = (t1**2) * (k**2) * (refractive_index_array**2 - t1)
        x3 = (t1**2) * k * (refractive_index_array**2)
        x4 = (refractive_index_array**4) - (k**2) * (
            (refractive_index_array**2 - t1) ** 2
        )
        x5 = t2 / t1
        x6 = x5 * (t1 - 1) + 1 - t2

        "r_90/t_90: Reflectance/Transmitance for remaining plates since light is isotropic in leaf"
        self.r_90 = x1 + x2 / x4
        self.t_90 = x3 / x4
        "r_alpha/t_aplha: Reflectance/Transmitance for first plate"
        self.r_alpha = x5 * self.r_90 + x6
        self.t_alpha = x5 * self.t_90
        # print(f"r_90: {self.r_90}")
        # print(f"t_90: {self.t_90}")
        # print(f"r_alpha: {self.r_alpha}")
        # print(f"t_alpha: {self.t_alpha}")

        self.delta = self._delta()
        self.alpha = self._alpha()
        self.beta = self._beta()

        # print(f"beta: {self.beta}")
        # print(f"delta: {self.delta}")
        # print(f"alpha: {self.alpha}")
        self.b = self._b()

        self.s1 = self.r_alpha * (
            self.alpha * self.b ** (self.N - 1)
            - (self.alpha**-1) * (self.b ** (1 - self.N))
        ) + (self.t_alpha * self.t_90 - self.r_alpha * self.r_90) * (
            self.b ** (self.N - 1) - self.b ** (1 - self.N)
        )

        self.s2 = self.t_alpha * (self.alpha - self.alpha**-1)

        self.s3 = (
            self.alpha * (self.b ** (self.N - 1))
            - (self.alpha**-1) * (self.b ** (1 - self.N))
            - self.r_90 * (self.b ** (self.N - 1) - self.b ** (1 - self.N))
        )

    def _b(self):
        return (
            (self.beta * (self.alpha - self.r_90))
            / (self.alpha * (self.beta - self.r_90))
        ) ** 0.5

    def _alpha(self):
        return (1 + self.r_90**2 - self.t_90**2 + self.delta**0.5) / (
            2 * self.r_90
        )

    def _beta(self):
        return (1 + self.r_90**2 - self.t_90**2 - self.delta**0.5) / (
            2 * self.r_90
        )

    def _delta(self):
        return (self.t_90**2 - self.r_90**2 - 1) ** 2 - (4 * self.r_90**2)

    def reflectance(self):
        return self.s1 / self.s3

    def transmittance(self):
        return self.s2 / self.s3

    def output(self):
        return [self.wavelength, 1 - self.reflectance(), self.transmittance()]


if __name__ == "__main__":
    model = MultiPlateModel(1.518, 58, 0.0131, 0.003662)
    # model = PROSPECT1990(2.698, 70.8, 0.000117, 0.009327)
    # df = pd.DataFrame(model.output())
    output = model.output()
    # print(f"reflectance: {output[1]}")
    # print(f"transmittance: {output[2]}")
    plot, axs = plt.subplots()
    axs.plot(output[0], output[1], label="Reflectance")
    axs.plot(output[0], output[2], label="Transmittance")
    total = output[1] + output[2]

    axs.plot(output[0], total, label="R+T")
    axs.legend()
    plot.savefig("test.jpg")
