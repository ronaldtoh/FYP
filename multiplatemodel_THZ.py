import numpy as np
from helper_functions import get_tav, get_tav_THZ
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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


class MultiPlateModel_THZ:
    def __init__(self, N: int, Cab, Cw, Cm):
        "Leaf Structure Parameter"
        self.N = N
        self.Cab = Cab
        self.Cw = Cw
        self.Cm = Cm
        self.calculate()

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

    def _reflectance(self):
        return self.s1 / self.s3

    def _transmittance(self):
        return self.s2 / self.s3

    def calculate(self):
        self.frequency = np.linspace(0.2, 1.6, 100)
        self.wavelength = 3e8 / (self.frequency * 10e12) * 1000

        self.refractive_index_array = np.linspace(2.1, 1.7, 100)
        self.dry_matter_absorption_coeff = np.linspace(5, 140, 100)
        self.water_absorption_coeff = np.linspace(100, 300, 100)
        self.albino_absorption_coeff = np.linspace(50, 200, 100)
        # self.scatter = abs_scatter(self.frequency, 40 ,)
        # This a_k is the absorption coefficient.
        # It is derived from adding:
        # [(Chlorophyll (a+b) content of leaf) * (Specific absorption coefficient of leaf) +
        # (Water content of leaf) * (Specific absorption coefficient of leaf) +
        # (Dry Matter content of leaf) * (Specific absorption coefficient of leaf)] / Leaf Structure Parameter
        # + Absorption coefficient of albino elementary layer
        # + Scattering-induced absorption
        a_k = (
            self.Cw * self.water_absorption_coeff
            + self.Cm * self.dry_matter_absorption_coeff
        ) / self.N

        # Getting the transmission coefficient k
        # Should never hit the if statement
        if np.all(a_k <= 0):
            k = 1
        else:
            k = (1 - a_k) * np.exp(-a_k) + (a_k**2) * exp1(a_k)
        t1 = get_tav_THZ(90, self.refractive_index_array, a_k, self.wavelength)
        t2 = get_tav_THZ(40, self.refractive_index_array, a_k, self.wavelength)

        x1 = 1 - t1
        x2 = (t1) ** 2 * (k**2) * (self.refractive_index_array**2 - t1)  # * t2
        x3 = (t1) ** 2 * k * (self.refractive_index_array**2)  # * t2
        x4 = (self.refractive_index_array**4) - (k**2) * (
            (self.refractive_index_array**2 - t1) ** 2
        )
        x5 = t2 / t1
        x6 = x5 * (t1 - 1) + 1 - t2

        "r_90/t_90: Reflectance/Transmitance for remaining plates since light is isotropic in leaf"
        self.r_90 = x1 + x2 / x4
        self.t_90 = x3 / x4
        "r_alpha/t_aplha: Reflectance/Transmitance for first plate"
        self.r_alpha = x5 * self.r_90 + x6
        self.t_alpha = x5 * self.t_90

        self.delta = self._delta()
        self.alpha = self._alpha()
        self.beta = self._beta()

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

    def output(self):
        return [self.frequency, self._reflectance(), self._transmittance()]

    def summary(self, title="test"):
        plot, axs = plt.subplots()
        output = self.output()
        axs.plot(output[0], output[1], label="Reflectance")
        axs.plot(output[0], output[2], label="Transmittance")

        total = output[1] + output[2]
        absorbed = 1 - total
        axs.plot(output[0], absorbed, label="Absorbed")

        axs.set_title(
            f"Reflectance, Transmittance,Absorbance of light through a {title} leaf"
        )
        axs.set_xlabel("frequency/Thz")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc=(1.04, 0.5))
        plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "output " + title + ".jpg"
        plot.savefig(path)


if __name__ == "__main__":
    # model = MultiPlateModel_THZ(1.518, 58, 0.00131, 0.003662)
    # model = MultiPlateModel_THZ(1.2, 30, 0.0015, 0.01)
    # model = MultiPlateModel_THZ(2.2, 30, 0.0015, 0.0005)
    # model = PROSPECT1990(2.698, 70.8, 0.000117, 0.009327)
    # output = model.output()
    # model = MultiPlateModel_THZ(2.107, 35.2, 0.000244, 0.00225)  # Dry Lettuce
    model = MultiPlateModel_THZ(2.275, 23.7, 0.0075, 0.005811)  # Fresh Rice
    model.summary("fresh rice")
