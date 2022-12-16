import numpy as np

k = np.array([[3, 4, 5], [7, 8, 7]])
b = np.array([[5, 5, 2], [5, 5, 2]])
a = np.array([[1, 2, 2], [3, 4, 9]])
ts = (np.divide(k**2, (6 * b**3)) + np.divide(k, b) - b / 2) - (
    (np.divide(k**2, 6 * a**3)) + np.divide(k, a) - a / 2
)
print(ts)

ts2 = ((k**2) / (6 * b**3) + k / b - b / 2) - (
    ((k**2) / (6 * a**3)) + k / a - a / 2
)
print(ts2)

test_arr = np.array([1, 2, 3, 4, 5])

print(k / b)


# tp1 = (-2 * r2 * (b - a)) / (rp**2)
# tp2 = (-2 * r2 * rp) * np.log(b / a) / (rm**2)
# tp3 = r2 * ((b**-1) - (a**-1)) / 2
# tp4 = (
#     16
#     * r2**2
#     * (r2**2 + 1)
#     * np.log((2 * rp * b - rm**2) / (2 * rp * a - rm**2))
#     / (rp**3 * rm**2)
# )
# tp5 = (
#     16
#     * r2**3
#     * ((2 * rp * b - rm**2) ** -1 - (2 * rp * a - rm**2) ** -1)
#     / (rp**3 * rm**2)
# )
# tp = tp1 + tp2 + tp3 + tp4 + tp5

print(np.cos(np.radians(40)))
