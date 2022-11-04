from data_another_style import data
import numpy as np
import matplotlib.pyplot as plt

database = np.array(data)
plt.figure()
plt.plot(database[:, 0], database[:, 3], label="Chlorophyll a+b")
plt.plot(database[:, 0], database[:, 4], label="Water")
plt.plot(database[:, 0], database[:, 5], label="Dry matter")
plt.grid()
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Specific Absorption Coefficient")

plt.savefig("dataplot.jpg")
