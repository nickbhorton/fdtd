import numpy as np
import matplotlib.pyplot as plt

from analysis import bistatic_echo_width


def load_bistatic_echo_width(data_path):
    compare_data = np.load(data_path)
    phi_compare = compare_data["phi"]
    result_compare = compare_data["result"]
    return phi_compare, result_compare


def compute_rmse(data_path1, data_path2, t_sig):
    phi, bist_s = bistatic_echo_width(data_path1, t_sig)
    _, bist_a = load_bistatic_echo_width(data_path2)
    dphi = phi[1] - phi[0]
    rmse = np.sqrt(np.sum((bist_s - bist_a) ** 2) * dphi) / np.sqrt(
        np.sum(bist_a**2) * dphi
    )
    return rmse


target_frequency = 10e9
t_sig = 3.0 / target_frequency

thickness = [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]

data_paths = [
    "data2/pml_1.npz",
    "data2/pml_2.npz",
    "data2/pml_3.npz",
    "data2/pml_4.npz",
    "data2/pml_5.npz",
]
rmse = [
    compute_rmse(data_paths[i], "tmz_pec_compare.npz", t_sig)
    for i in range(len(thickness))
]
fig, ax = plt.subplots()
ax.plot(thickness, rmse, color="black")
ax.scatter(thickness, rmse, color="black")
ax.set_xlabel(r"PML Thickness in wavelengths")
ax.set_ylabel("RMSE")
fig.savefig("compare_pml_thickness.png", dpi=200)
plt.show()
