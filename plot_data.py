from data_another_style import data
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    plotter = PlotData()
    plotter.plot_water_comparison()
    plotter.plot_dry_matter_comparison()

    pass
