from data_another_style import data
import numpy as np
import matplotlib.pyplot as plt
from multiplatemodel_THZ import MultiPlateModel_THZ

database = np.array(data)


class PlotData:
    def __init__(self):

        self.frequency = np.linspace(0.2, 1.8, 100)
        self.wavelength = 3e8 / (self.frequency * 10**12) * 1_000_000
        self.refractive_index_array = np.linspace(2.1, 1.7, 100)
        self.dry_matter_absorption_coeff = np.linspace(5, 140, 100)
        self.water_absorption_coeff = np.linspace(100, 300, 100)
        self.albino_absorption_coeff = np.linspace(50, 200, 100)

    def plot_water_comparison(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.set_figheight(8)
        fig.set_figwidth(16)
        plt.subplot(1, 2, 1)
        ax1.plot(database[:, 0] / 1000, database[:, 4], color="k")
        ax1.grid()
        ax1.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax1.set_ylabel(r"Specific absorption coefficient $(cm^{-1})$", fontsize=20)
        ax1.set_ylim(-20, 320)
        ax1.tick_params(labelsize=16)
        ax1.set_title(
            "Specific absorption coefficient \n of water (PROSPECT-5)", fontsize=20
        )

        plt.subplot(1, 2, 2)
        ax2.plot(self.wavelength, self.water_absorption_coeff, color="k")
        ax2.grid()
        ax2.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_title("Specific absorption coefficient \n of water (THz)", fontsize=20)
        plt.tight_layout()
        plt.savefig("comparison_water.jpg")
        return

    def plot_dry_matter_comparison(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.set_figheight(8)
        fig.set_figwidth(16)
        plt.subplot(1, 2, 1)
        ax1.plot(database[:, 0] / 1000, database[:, 5], color="k")
        ax1.grid()
        ax1.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax1.set_ylabel(r"Specific absorption coefficient $(cm^{-1})$", fontsize=20)
        ax1.set_ylim(-10, 150)
        ax1.tick_params(labelsize=16)
        ax1.set_title(
            "Specific absorption coefficient \n of dry matter (PROSPECT-5)", fontsize=20
        )

        plt.subplot(1, 2, 2)
        ax2.plot(self.wavelength, self.dry_matter_absorption_coeff, color="k")
        ax2.grid()
        ax2.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_title(
            "Specific absorption coefficient \n of dry matter (THz)", fontsize=20
        )
        plt.tight_layout()
        plt.savefig("comparison_dry_matter.jpg")
        return

    def plot_varying_cw(self):
        cw_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(12, 5), sharey=True, sharex=True
        )

        fig.suptitle("Varying Cw")
        ax1.set_title("Reflectance")
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid()
        ax1.set_ylabel("Coefficient")

        ax2.set_title("Transmittance")
        ax2.set_ylim(-0.2, 1.2)
        ax2.grid()
        ax2.set_xlabel("Frequency (THz)")

        ax3.set_title("Absorbance")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(1.2, 30, i, 0.01)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
        fig.savefig("vary_cw.jpg")

    def plot_varying_cm(self):
        cw_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(12, 5), sharey=True, sharex=True
        )

        fig.suptitle("Varying Cm")
        ax1.set_title("Reflectance")
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid()
        ax1.set_ylabel("Coefficient")

        ax2.set_title("Transmittance")
        ax2.set_ylim(-0.2, 1.2)
        ax2.grid()
        ax2.set_xlabel("Frequency (THz)")

        ax3.set_title("Absorbance")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(1.2, 30, 0.001, i)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
        fig.savefig("vary_cm.jpg")

    def plot_varying_N(self):
        N_values = [1, 1.5, 2, 2.5, 3]
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(12, 5), sharey=True, sharex=True
        )

        fig.suptitle("Varying N")
        ax1.set_title("Reflectance")
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid()
        ax1.set_ylabel("Coefficient")

        ax2.set_title("Transmittance")
        ax2.set_ylim(-0.2, 1.2)
        ax2.grid()
        ax2.set_xlabel("Frequency (THz)")

        ax3.set_title("Absorbance")
        ax3.set_ylim(-0.2, 1.2)
        # ax3.set_xlim(0.2, 0.4)
        ax3.grid()
        for i in N_values:
            model = MultiPlateModel_THZ(i, 30, 0.001, 0.01)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
        fig.savefig("vary_N.jpg")


if __name__ == "__main__":
    plotter = PlotData()
    # plotter.plot_water_comparison()
    # plotter.plot_dry_matter_comparison()
    plotter.plot_varying_N()

    pass
