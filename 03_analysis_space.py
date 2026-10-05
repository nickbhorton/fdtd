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

space = [10, 15, 20, 25, 30]

data_paths = [
    "data2/space_10.npz",
    "data2/space_15.npz",
    "data2/space_20.npz",
    "data2/space_25.npz",
    "data2/space_30.npz",
]
rmse = [
    compute_rmse(data_paths[i], "tmz_pec_compare.npz", t_sig) for i in range(len(space))
]
fig, ax = plt.subplots()
ax.plot(space, rmse, color="black")
ax.scatter(space, rmse, color="black")
ax.set_xlabel(r"Spacial Discritization (samples per wavelength)")
ax.set_ylabel("RMSE")
fig.savefig("compare_space.png", dpi=200)
plt.show()
