import numpy as np
from helper_functions import (
    reflectance_transmittance,
    reflectance_transmittance_2plates,
    get_tav_THZ,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.special import exp1
from data_another_style import data
from typing import List

from THZ_PROSPECT import THZ_PROSPECT

# import pandas as pd


class MultiPlateModel_THZ_disease:
    def __init__(self, N, Cw, Cm):
        self.N = N
        self.Cw = Cw
        self.Cm = Cm

        self.absorption_disease = np.linspace(1.5, 8, 100) / (2.303 * 1000000)
        self.refractive_index_disease = np.linspace(1.5, 1.1, 100)
        self.thickness_disease = 0.05
        self.frequency = np.linspace(0.2, 1.6, 100)

        self.wavelength = 3e8 / (self.frequency * 10e12)

        self.refractive_index_leaf = np.linspace(2.1, 1.7, 100)
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
        self.absorption_leaf = (
            (
                self.Cw * self.water_absorption_coeff
                + self.Cm * self.dry_matter_absorption_coeff
            )
            / self.N
            / 100
        )
        # print(f"absorption disease: {self.absorption_disease}")
        self.calculate()

    def _b(self):
        """Calculate intermediate value/matrix b

        Returns:
            ndarray(float): b
        """
        return (
            (self.beta * (self.alpha - self.r_90))
            / (self.alpha * (self.beta - self.r_90))
        ) ** 0.5

    def _alpha(self):
        """Calculate intermediate value/matrix alpha

        Returns:
            ndarray(float): alpha
        """
        return (1 + self.r_90**2 - self.t_90**2 + self.delta**0.5) / (
            2 * self.r_90
        )

    def _beta(self):
        """Calculate intermediate value/matrix beta

        Returns:
            ndarray(float): beta
        """
        return (1 + self.r_90**2 - self.t_90**2 - self.delta**0.5) / (
            2 * self.r_90
        )

    def _delta(self):
        """Calculate intermediate value/matrix delta

        Returns:
            ndarray(float): delta
        """
        return (self.t_90**2 - self.r_90**2 - 1) ** 2 - (4 * self.r_90**2)

    def _reflectance(self):
        """Calculate reflectance of leaf

        Returns:
            ndarray(float): reflectance
        """
        return self.s1 / self.s3

    def _transmittance(self):
        """Calculate transmittance of leaf

        Returns:
            ndarray(float): transmittance
        """
        return self.s2 / self.s3

    def calculate(self):
        """Calculate intermediate values of model"""
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
        self.a_k = (
            self.Cw * self.water_absorption_coeff
            + self.Cm * self.dry_matter_absorption_coeff
        ) / self.N
        # print(a_k)

        # Getting the transmission coefficient k
        # Should never hit the if statement
        if np.all(self.a_k <= 0):
            k = 1
        else:
            k = (1 - self.a_k) * np.exp(-self.a_k) + (self.a_k**2) * exp1(self.a_k)

        # Calculate complex refractive index
        leaf_cri = self.refractive_index_array + 1j * self.a_k
        disease_cri = self.refractive_index_disease  # + 1j * self.absorption_disease
        # print(f"disease_cri: {disease_cri}")
        # print(f"leaf_cri: {leaf_cri}")
        # print(f"dise_cri: {disease_cri}")

        # Calculate phase shifts at each interface
        disease_delta = 2 * np.pi * disease_cri * 0.05 / self.wavelength
        leaf_delta = 2 * np.pi * leaf_cri * 1 / self.wavelength

        # Calculate reflection coefficients at disease-leaf interface
        # r02 = -r20
        rdl = (disease_cri - leaf_cri) / (disease_cri + leaf_cri)

        # Calculate reflection coefficients at air-disease interface
        # r10 = -r01
        rad = (1 - disease_cri) / (1 + disease_cri)

        # Calculate reflection coefficients at leaf-air interface
        # r21 = -r12
        rla = (leaf_cri - 1) / (1 + leaf_cri)

        # Calculate transmission coefficients at disease-leaf interface
        # t02 = t20
        tdl = 2 * disease_cri / (disease_cri + leaf_cri)

        # Calculate transmission coefficients at air-disease interface
        # t10 = t01
        tad = 2 / (disease_cri + 1)

        # Calculate reflection coefficients at leaf-air interface
        # t21 = t12
        tla = 2 / (1 + leaf_cri)

        # Calculate rdla, tdla
        rdla = rdl + (tdl**2 * rla * k**2) / (1 - (-1) * rdl * rla * k**2)
        tdla = (tdl * -1 * rdl * k**2) / (1 - (-1) * rdl * rla * k**2)
        disease_delta = -2 * np.pi * disease_cri * 0.05 / self.wavelength
        # Calculate total reflection and transmittance of first plate
        self.r_alpha = rad + (tad**2 * rdla) / (1 + rad * rdla)
        self.t_alpha = (tad * tdla) / (1 + rad * rdla)
        self.r_alpha = self.r_alpha**2
        self.t_alpha = self.t_alpha**2
        # print(self.r_alpha)
        # Calculate total reflection and transmittance of other plates
        self.r_90 = -rla + (tla**2 * rla * k**2) / (1 - rla**2 * k**2)
        self.t_90 = (tla**2) / (1 - rla**2 * k**2)

        self.r_90 = self.r_90**2
        self.t_90 = self.t_90**2
        # t1 = get_tav_THZ(90, self.refractive_index_array, self.a_k, self.wavelength)
        # t2 = get_tav_THZ(40, self.refractive_index_array, self.a_k, self.wavelength)
        # # print(f"t1: {t1}")
        # # print(f"t2: {t2}")
        # x1 = 1 - t1
        # x2 = (t1) ** 2 * (k**2) * (self.refractive_index_array**2 - t1)  # * t2
        # x3 = (t1) ** 2 * k * (self.refractive_index_array**2)  # * t2
        # x4 = (self.refractive_index_array**4) - (k**2) * (
        #     (self.refractive_index_array**2 - t1) ** 2
        # )
        # x5 = t2 / t1
        # x6 = x5 * (t1 - 1) + 1 - t2

        # "r_alpha/t_aplha: Reflectance/Transmitance for first plate"
        # self.r_alpha, self.t_alpha = rl1, tl1
        # # print(f"self.r_alpha: {self.r_alpha}")
        # # print(f"self.t_alpha: {self.t_alpha}")

        # "r_90/t_90: Reflectance/Transmitance for remaining plates since light is isotropic in leaf"
        # self.r_90 = x1 + x2 / x4
        # self.t_90 = x3 / x4

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

    def get_r_alpha(self):
        return self.r_alpha

    def get_t_alpha(self):
        return self.t_alpha

    def plot_first_layer(self):
        plot, axs = plt.subplots()
        output = self.output()
        axs.plot(output[0], self.r_alpha, label="Reflectance 1st layer")
        axs.plot(output[0], 1 - self.t_alpha, label="Transmittance 1st layer")

        # total = output[1] + output[2]
        # absorbed = 1 - total
        # axs.plot(output[0], absorbed, label="Absorption")

        axs.set_title(f"Optical spectrum (Terahertz) of a diseased leaf")
        axs.set_xlabel("Frequency (THz)")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc=(1.04, 0.5))
        plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "output_disease_first_layer.jpg"
        plot.savefig(path)
        return

    def plot_other_layer(self):
        plot, axs = plt.subplots()
        output = self.output()
        axs.plot(output[0], self.r_90, label="Reflectance other layer")
        axs.plot(output[0], 1 - self.t_90, label="Transmittance other layer")
        # print(self.t_90)
        # total = output[1] + output[2]
        # absorbed = 1 - total
        # axs.plot(output[0], absorbed, label="Absorption")

        axs.set_title(f"Optical spectrum (Terahertz) of a diseased leaf")
        axs.set_xlabel("Frequency (THz)")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc=(1.04, 0.5))
        plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "output_disease_other_layer.jpg"
        plot.savefig(path)
        return

    def output(self):
        """Returns frequency, reflectance, transmittance

        Returns:
            List[np.ndarray(float), np.ndarray(float), np.ndarray(float)]: [frequency, reflectance, transmittance]
        """
        return [self.frequency, self._reflectance(), self._transmittance()]

    def summary(self):
        """Plot and save a graph of reflectance and transmittance against frequency"""
        plot, axs = plt.subplots()
        output = self.output()
        axs.plot(output[0], output[1], label="Reflectance")
        axs.plot(output[0], output[2], label="Transmittance")

        total = output[1] + output[2]
        # absorbed = 1 - total
        # axs.plot(output[0], absorbed, label="Absorption")

        axs.set_title(f"Optical spectrum (Terahertz) of a diseased leaf")
        axs.set_xlabel("Frequency (THz)")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc=(1.04, 0.5))
        plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "output_disease_with_delta.jpg"
        plot.savefig(path)

    def test(self):
        # print(f"self.absorption_leaf: {self.absorption_leaf}")
        # print(f"self.absorption_d: {self.absorption_disease}")
        t01 = (
            2
            * self.refractive_index_disease
            / (self.refractive_index_disease + 1)
            * np.exp(
                -1j
                * self.absorption_disease
                * self.wavelength
                * 0.05
                / (2 * np.pi * self.refractive_index_disease)
            )
        )

        r01 = (self.refractive_index_disease - 1) / (
            self.refractive_index_disease + 1
        ) - 1j * self.absorption_disease * self.wavelength / (
            4 * np.pi * self.refractive_index_disease
        )

        tl2 = (
            2
            * self.refractive_index_leaf
            / (1 + self.refractive_index_leaf)
            * np.exp(
                -1j
                * self.absorption_leaf
                * self.wavelength
                * 1
                / (2 * np.pi * self.refractive_index_leaf)
            )
        )

        rl2 = (self.refractive_index_leaf - self.refractive_index_disease) / (
            self.refractive_index_disease + self.refractive_index_leaf
        ) - 1j * self.absorption_leaf * self.wavelength / (
            4 * np.pi * self.refractive_index_leaf
        )

        k1 = 2 * np.pi * self.refractive_index_disease / self.wavelength
        k2 = 2 * np.pi * self.refractive_index_leaf / self.wavelength

        R_total = r01 + (rl2 * np.exp(-2j * k2 * 1 - self.absorption_leaf * k2)) / (
            1 + r01 * rl2 * np.exp(-2j * k2 * 1 - self.absorption_leaf * k2)
        )

        T_total = (
            t01
            * tl2
            * np.exp(
                -1j * k2 * (1 + self.thickness_disease)
                - self.absorption_disease * k1
                - self.absorption_leaf * k2
            )
        ) / (1 + r01 * rl2 * np.exp(-2j * k2 * 1 - self.absorption_leaf * k2))

        return (abs(R_total)), (abs(T_total))

    def fresnel_first_plate(self, k, d1=0.05):
        # Calculate complex refractive index
        leaf_cri = self.get_complex_refractive_index()
        disease_cri = self.refractive_index_disease  # + 1j * self.absorption_disease


        # Calculate phase shifts at each interface
        disease_delta = 2 * np.pi * disease_cri * d1 / self.wavelength
        leaf_delta = 2 * np.pi * leaf_cri * 1 / self.wavelength

        # Calculate reflection coefficients at disease-leaf interface
        rdl = (disease_cri - leaf_cri) / (disease_cri + leaf_cri)

        # Calculate reflection coefficients at air-disease interface
        rad = (1 - disease_cri) / (1 + disease_cri)

        # Calculate transmission coefficients at disease-leaf interface
        tdl = 2 * disease_cri / (disease_cri + leaf_cri)

        # Calculate transmission coefficients at air-disease interface
        tad = 2 / (disease_cri + 1)

        # Calculate rdla, tdla

        rdla = rdl + (tdl**2 * rad * k**2) / (1 - rdl * rad * k**2)
        tdla = (tad * rad * k**2) / (1 - rad * rdl * k**2)

        # Calculate total reflection and transmission coefficients

        # r = rad + (tad**2 * rdla * np.exp(-2j * disease_delta)) / (
        #     1 + rad * rdla * np.exp(-2j * disease_delta)
        # )

        # t = (tad * tdla * np.exp(-1j * disease_delta)) / (
        #     1 + tad * tdla * np.exp(-2j * disease_delta)
        # )
        r = rad + (tad**2 * rdla) / (1 - rad * rdla)

        t = (tad * tdla) / (1 - tad * tdla)
        return abs(r), abs(t)

    def comparison(self):
        plot, axs = plt.subplots()
        _, R, T = self.calc_leaf_r_and_t(self.N, self.Cw, self.Cm)
        R_dis, T_dis = disease_model.fresnel()
        axs.plot(self.frequency, R, label="R_original")
        axs.plot(self.frequency, 1 - T, label="T_original")
        axs.plot(self.frequency, R_dis, label="R_disease")
        axs.plot(self.frequency, 1 - T_dis, label="T_disease")
        # total = output[1] + output[2]
        # absorbed = 1 - total
        # axs.plot(output[0], absorbed, label="Absorption")

        axs.set_title(f"Optical spectrum (Terahertz) of a diseased leaf delta")
        axs.set_xlabel("Frequency (THz)")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc="upper left")
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "test_disease_with_delta.jpg"
        plot.savefig(path)


if __name__ == "__main__":
    disease_model = MultiPlateModel_THZ_disease(2.107, 0.000244, 0.00225)
    disease_model.plot_first_layer()
    disease_model.plot_other_layer()
    disease_model.summary()
    # R_dis, T_dis = disease_model.fresnel()
    # print(f"R_Disease = {R_dis}")
    # print(f"T_Disease = {T_dis}")
    # print(f"r_alpha = {disease_model.get_r_alpha()}")
    # print(f"t_alpha - {disease_model.get_t_alpha()}")

    # disease_model.combined_output()
    # disease_model.calc_leaf_r_and_t( 2.107, 0.000244, 0.00225)
