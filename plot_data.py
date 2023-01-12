from data_another_style import data
import numpy as np
import matplotlib.pyplot as plt
from multiplatemodel_THZ import MultiPlateModel_THZ
from multiplatemodel_THZ_inverse import MultiPlateModel_THZ_inv
import random
import scipy
import json

database = np.array(data)


class PlotData:
    def __init__(self):

        self.frequency = np.linspace(0.2, 1.8, 100)
        self.wavelength = 3e8 / (self.frequency * 10**12) * 1_000_000
        self.refractive_index_array = np.linspace(2.1, 1.7, 100)
        self.dry_matter_absorption_coeff = np.linspace(5, 140, 100)
        self.water_absorption_coeff = np.linspace(100, 300, 100)
        self.albino_absorption_coeff = np.linspace(50, 200, 100)

    def plot_chlorophyll(self):
        fig, axs = plt.subplots()
        no = 100
        axs.plot(database[:no, 0], database[:no, 3])
        axs.grid()
        fig.savefig("data_chlorophyll.jpg")

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
        cw_values = [0.000100, 0.000500, 0.001000, 0.005000, 0.010000, 0.020000]
        cw_str = [str(x) for x in cw_values]
        cw_str = [x.ljust(8, "0") for x in cw_str]
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

        ax3.set_title("Absorption")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(2.0, 30, i, 0.01)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(labels=cw_str, loc="right", bbox_to_anchor=(1.451, 0.5))
        fig.savefig("vary_cw.jpg")

    def plot_varying_cw_freshleaves(self):
        cw_values = [0.00400, 0.040000]
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

        ax3.set_title("Absorption")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(2, 30, i, 0.0092)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
        fig.savefig("vary_cw_fresh_test.jpg")

    def plot_varying_cm_fresh(self):
        cw_values = [0.001000, 0.005000, 0.010000, 0.020000]
        cw_str = [str(x) for x in cw_values]
        cw_str = [x.ljust(8, "0") for x in cw_str]
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(12, 5), sharey=True, sharex=True
        )

        fig.suptitle(r"Varying Cm (N = 1.2, Cw = 0.020000 $cm^{-1}$)")
        ax1.set_title("Reflectance")
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid()
        ax1.set_ylabel("Coefficient")

        ax2.set_title("Transmittance")
        ax2.set_ylim(-0.2, 1.2)
        ax2.grid()
        ax2.set_xlabel("Frequency (THz)")

        ax3.set_title("Absorption")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(1.2, 30, 0.02, i)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)

        ax3.legend(labels=cw_str, loc="right", bbox_to_anchor=(1.451, 0.5))
        fig.savefig("vary_cm_fresh.jpg")

    def plot_varying_cm_dry(self):
        cw_values = [0.000100, 0.000500, 0.001000, 0.005000, 0.010000, 0.020000]
        cw_str = [str(x) for x in cw_values]
        cw_str = [x.ljust(8, "0") for x in cw_str]
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

        ax3.set_title("Absorption")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid()
        for i in cw_values:
            model = MultiPlateModel_THZ(1.2, 30, 0.001, i)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            ax1.plot(output[0], output[1], label=i)
            ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(labels=cw_str, loc="right", bbox_to_anchor=(1.451, 0.5))
        fig.savefig("vary_cm_dry.jpg")

    def plot_varying_N(self):
        N_values = [1.0, 1.5, 2.0, 2.5, 3.0]
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

        ax3.set_title("Absorption")
        ax3.set_ylim(-0.2, 1.2)
        # ax3.set_ylim(0.5, 0.7)
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
        fig.savefig("vary_N_zoom_in.jpg")
        # fig.savefig("vary_N.jpg")

    def plot_varying_N_zoomed(self):
        N_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        fig, ax3 = plt.subplots(1, 1, figsize=(8, 5), sharey=True, sharex=True)

        # fig.suptitle("Varying N (Absorption)")
        ax3.set_ylabel("Coefficient")
        ax3.set_xlabel("Frequency (THz)")
        ax3.set_title("Varying N (Absorption)")
        ax3.set_ylim(0.5, 0.7)
        ax3.set_xlim(0.2, 0.4)
        ax3.grid()
        for i in N_values:
            model = MultiPlateModel_THZ(i, 30, 0.001, 0.01)
            output = model.output()
            absorbance = 1 - output[1] - output[2]
            # ax1.plot(output[0], output[1], label=i)
            # ax2.plot(output[0], output[2], label=i)
            ax3.plot(output[0], absorbance, label=i)
        ax3.legend(loc="right", bbox_to_anchor=(1.13, 0.5))
        fig.savefig("vary_N_zoom_in.jpg")
        # fig.savefig("vary_N.jpg")

    def test_inverse_no_noise(self, iter: int = 50):
        gen_cw = []
        gen_cm = []
        gen_N = []
        inv_cw = []
        inv_cm = []
        inv_N = []
        data = dict()

        for i in range(iter):
            print(f"{i}")
            random_N = random.uniform(1.0, 3.5)
            random_cw = random.uniform(0.00006, 0.010)
            random_cm = random.uniform(0.00190, 0.0165)
            title = f"regress_{i}"
            model = MultiPlateModel_THZ(random_N, 0, random_cw, random_cm, title)
            simulated_output = model.output()[1:]
            try:
                inv_model = MultiPlateModel_THZ_inv(
                    simulated_output[0], simulated_output[1]
                )

                inv_output = inv_model.get_values()
            except RuntimeWarning:
                i -= 1
                print("Runtime Error")
                continue
            gen_cw.append(random_cw)
            gen_cm.append(random_cm)
            gen_N.append(random_N)
            inv_cw.append(inv_output.x[0])
            inv_cm.append(inv_output.x[1])
            inv_N.append(inv_output.x[2])
            data[i] = (
                random_cw,
                random_cm,
                random_N,
                inv_output.x[0],
                inv_output.x[1],
                inv_output.x[2],
            )
        with open("inv_data_bounded_100.json", "w") as outfile:
            json.dump(data, outfile)

        # Linear regression
        cw_regress = scipy.stats.linregress(gen_cw, inv_cw)
        cm_regress = scipy.stats.linregress(gen_cm, inv_cm)
        N_regress = scipy.stats.linregress(gen_N, inv_N)
        x = np.linspace(0, 0.010, 100)
        y = np.linspace(0, 0.0175, 100)
        z = np.linspace(1, 3.5, 100)
        # Plot graph for Cw
        plot, axs = plt.subplots()
        axs.scatter(gen_cw, inv_cw)
        axs.plot(x, x, "r")
        axs.set_title(f"Cw, $r^{2}$ = {cw_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_cw.jpg"
        plot.savefig(path)

        # Plot graph for Cm
        plot, axs = plt.subplots()
        axs.scatter(gen_cm, inv_cm)
        axs.plot(y, y, "r")
        axs.set_title(f"Cm, $r^{2}$ = {cm_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_cm.jpg"
        plot.savefig(path)

        # Plot graph for Cm
        plot, axs = plt.subplots()
        axs.scatter(gen_N, inv_N)
        axs.plot(z, z, "r")
        axs.set_title(f"N, $r^{2}$ = {N_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_N.jpg"
        plot.savefig(path)

        return

    def inverse_noise(self, iter: int = 50):
        snr = 1 / 30
        gen_cw = []
        gen_cm = []
        gen_N = []
        inv_cw = []
        inv_cm = []
        inv_N = []
        data = dict()

        for i in range(iter):
            print(f"{i}")
            random_N = random.uniform(1.0, 3.5)
            random_cw = random.uniform(0.00006, 0.010)
            random_cm = random.uniform(0.00190, 0.0165)
            title = f"regress_{i}"
            model = MultiPlateModel_THZ(random_N, 0, random_cw, random_cm, title)
            simulated_output = model.output()[1:]
            try:

                noisy_r = (
                    simulated_output[0] * np.random.normal(0, snr, 100)
                    + simulated_output[0]
                )
                noisy_t = (
                    simulated_output[1] * np.random.normal(0, snr, 100)
                    + simulated_output[1]
                )
                inv_model = MultiPlateModel_THZ_inv(noisy_r, noisy_t)

                inv_output = inv_model.get_values()
            except RuntimeWarning:
                i -= 1
                print("Runtime Error")
                continue
            gen_cw.append(random_cw)
            gen_cm.append(random_cm)
            gen_N.append(random_N)
            inv_cw.append(inv_output.x[0])
            inv_cm.append(inv_output.x[1])
            inv_N.append(inv_output.x[2])
            data[i] = (
                random_cw,
                random_cm,
                random_N,
                inv_output.x[0],
                inv_output.x[1],
                inv_output.x[2],
            )
        with open("inv_data_bounded_noisy_100.json", "w") as outfile:
            json.dump(data, outfile)

        # Linear regression
        cw_regress = scipy.stats.linregress(gen_cw, inv_cw)
        cm_regress = scipy.stats.linregress(gen_cm, inv_cm)
        N_regress = scipy.stats.linregress(gen_N, inv_N)
        x = np.linspace(0, 0.010, 100)
        y = np.linspace(0, 0.0175, 100)
        z = np.linspace(1, 3.5, 100)
        # Plot graph for Cw
        plot, axs = plt.subplots()
        axs.scatter(gen_cw, inv_cw)
        axs.plot(x, x, "r")
        axs.set_title(f"Cw, $r^{2}$ = {cw_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_cw_noisy.jpg"
        plot.savefig(path)

        # Plot graph for Cm
        plot, axs = plt.subplots()
        axs.scatter(gen_cm, inv_cm)
        axs.plot(y, y, "r")
        axs.set_title(f"Cm, $r^{2}$ = {cm_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_cm_noisy.jpg"
        plot.savefig(path)

        # Plot graph for Cm
        plot, axs = plt.subplots()
        axs.scatter(gen_N, inv_N)
        axs.plot(z, z, "r")
        axs.set_title(f"N, $r^{2}$ = {N_regress.rvalue**2:.3f}")
        axs.set_xlabel("Simulated")
        axs.set_ylabel("Inversed")
        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        # plt.legend(loc=(1.04, 0.5))
        # plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "bounded_inv_N_noisy.jpg"
        plot.savefig(path)

        return

    def open_up_json(self):
        with open("inv_data_unbounded_100.json") as json_file:
            data = json.load(json_file)
            # data[i] = (random_cw,random_cm,random_N,inv_cw,inv_cm,inv_N)
        cw_check = []
        N_check = []
        final_check = []
        for value in data.values():
            if value[4] < 0.0020:
                cw_check.append(value[0])
            if value[5] <= 1.1:
                N_check.append(value[0])
            if value[4] < 0.0020 and value[5] <= 1.1:
                final_check.append(value[0])

        fig, ax = plt.subplots()
        ax.hist(cw_check, edgecolor="black")
        ax.set_title(
            r"Simulated Cw values where inversed Cm values < 0.002000 g $cm^{-1}$"
        )
        ax.set_xlabel("Simulated Cw values")
        ax.set_ylabel("Number of datapoints")
        # ax.grid()
        fig.savefig("histogram cw values from cm.jpg")

        fig, ax = plt.subplots()
        ax.hist(N_check, edgecolor="black")
        ax.set_title(r"Simulated Cw values where inversed N values ~ 1.0")
        ax.set_xlabel("Simulated Cw values")
        ax.set_ylabel("Number of datapoints")
        # ax.grid()
        fig.savefig("histogram cw values from N.jpg")

        fig, ax = plt.subplots()
        ax.hist(final_check, edgecolor="black")
        ax.set_title(r"Simulated Cw values from both variables")
        ax.set_xlabel("Simulated Cw values")
        ax.set_ylabel("Number of datapoints")
        # ax.grid()
        fig.savefig("histogram cw values from both cw and N.jpg")
        return

    def plot_noisy_spectrum(self):
        model = MultiPlateModel_THZ(1.8, 30, 0.0008, 0.0062)

        plot, axs = plt.subplots()
        output = model.output()
        axs.plot(output[0], output[1], "k", label="Reflectance-clean")
        axs.plot(output[0], output[2], "k", label="Transmittance-clean")

        total = output[1] + output[2]
        absorbed = 1 - total
        # axs.plot(output[0], absorbed, label="Absorption")

        snr = 1 / 5
        noisy_r = output[1] * np.random.normal(0, snr, 100) + output[1]
        noisy_t = output[2] * np.random.normal(0, snr, 100) + output[2]

        axs.plot(output[0], noisy_r, label="Reflectance-noisy")
        axs.plot(output[0], noisy_t, label="Transmittance-noisy")

        axs.set_title(f"Noise vs no noise")
        axs.set_xlabel("Frequency (THz)")
        axs.set_ylabel("")
        axs.set_ylim(-0.2, 1.2)

        axs.grid(which="both", color=(0.8, 0.8, 0.8))
        plt.legend(loc=(1.04, 0.5))
        plt.legend(bbox_to_anchor=(1, 0.4), loc="center right")
        path = "noise_test.jpg"
        plot.savefig(path)
        return


if __name__ == "__main__":
    plotter = PlotData()
    # plotter.inverse_noise(100)
    # plotter.open_up_json()
    plotter.plot_varying_cw()
    # plotter.plot_varying_cm_fresh()
    plotter.plot_varying_cm_dry()
    # plotter.plot_chlorophyll()

    pass
