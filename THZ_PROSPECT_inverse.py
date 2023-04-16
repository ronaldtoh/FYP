import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from data_another_style import data
from THZ_PROSPECT import THZ_PROSPECT
import json
import scipy.optimize


class MultiPlateModel_THZ_inv:
    def __init__(
        self,
        reflectance_array,
        transmittance_array,
        initial_N=2.0,
        initial_cw=0.001,
        initial_cm=0.01,
    ):
        self.measured_reflectance_array = reflectance_array
        self.measured_transmittance_array = transmittance_array
        self.initial = np.array([initial_N, initial_cw, initial_cm])
        pass

    def generate_lookup_table(self):
        cm_values = np.linspace(0.001900, 0.016500, 100)
        cw_values = np.linspace(0.000100, 0.040000, 100)

        storeroom = {}
        for cm in cm_values:
            for cw in cw_values:
                output = THZ_PROSPECT(self.N, 0, cw, cm).output()[1:]
                storeroom[(cw, cm)] = output

        with open("LUT.json", "w") as outfile:
            json.dump(storeroom, outfile)

        return

    def get_values(self):
        def cost_function(x):
            cw, cm, N_iter = x
            output = THZ_PROSPECT(N_iter, 0, cw, cm).output()[1:]

            refl_diff = output[0] - self.measured_reflectance_array
            trans_diff = output[1] - self.measured_transmittance_array

            refl_cost = sum(refl_diff**2 / len(refl_diff))
            trans_cost = sum(trans_diff**2 / len(refl_diff))
            RMSE_cost = refl_cost + trans_cost

            return RMSE_cost

        result = scipy.optimize.minimize(
            cost_function,
            self.initial,
            bounds=((0.00006, 0.010), (0.00190, 0.0165), (1, 3.5)),
            method="L-BFGS-B",
        )
        return result


if __name__ == "__main__":
    model = THZ_PROSPECT(2.107, 35.2, 0.000244, 0.00225, "dry lettuce")  # Dry Lettuce)
    output_arr = model.output()
    inv = MultiPlateModel_THZ_inv(output_arr[1], output_arr[2])
    res = inv.get_values()
    print(res.x)
    pass
