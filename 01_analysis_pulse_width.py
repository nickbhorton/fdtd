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
t_sig = [
    0.1 / target_frequency,
    1.0 / target_frequency,
    2.0 / target_frequency,
    3.0 / target_frequency,
    4.0 / target_frequency,
    5.0 / target_frequency,
]
data_paths = [
    "data2/t_sig_0_4k.npz",
    "data2/t_sig_1_2k.npz",
    "data2/t_sig_2_4k.npz",
    "data2/t_sig_3_4k.npz",
    "data2/t_sig_4_4k.npz",
    "data2/t_sig_5_5k.npz",
]
rmse = [
    compute_rmse(data_paths[i], "tmz_pec_compare.npz", t_sig[i])
    for i in range(len(t_sig))
]
fig, ax = plt.subplots(layout="constrained")
ax.plot(t_sig, rmse, color="black")
ax.scatter(t_sig, rmse, color="black")
ax.set_xlabel(r"$t_\sigma$")
ax.set_ylabel("RMSE")
fig.savefig("compare_pulse_width.png", dpi=200)
plt.show()
