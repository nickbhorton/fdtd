import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2, jn

plt.rcParams.update(
    {
        "axes.grid": False,
        "axes.edgecolor": "black",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 12,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
    }
)

center_frequency = 10e9
target_frequency = np.linspace(9e9,11e9,100)
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / center_frequency
a = wavelength / 2

sig_mono = np.zeros_like(target_frequency)

for i, f in enumerate(target_frequency):
    k = 2.0 * np.pi * f / phase_velocity
    phi_lin = np.pi
    E0 = 1.0
    N = 50
    sig_n = np.zeros((2 * N + 1,), dtype=np.complex128)
    for n in range(-N, N + 1):
        j = n + N
        sig_n[j] = (
            ((-1.0j) ** n * jn(n, k * a) / hankel2(n, k * a))
            * np.exp(1.0j * (2 * n + 1) * np.pi / 4)
            * np.exp(1.0j * n * phi_lin)
        )
    sig_mono[i] = ((4.0 * a) / (k * a)) * np.abs(np.sum(sig_n, axis=0)) ** 2

fig, ax = plt.subplots()
ax.plot(target_frequency * 1e-9, sig_mono, color="black", linewidth=2)

np.savez(
    "tmz_pec_monostatic.npz",
    **{"frequency": target_frequency, "sigma_monostatic": sig_mono},
)

plt.show()
