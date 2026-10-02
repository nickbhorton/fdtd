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

f = 10e9
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
a = 4.0 * wavelength / 2
k = 2.0 * np.pi * f / phase_velocity

grid_points = 200
wavelength_grid = 5.0
x = np.linspace(
    -wavelength * wavelength_grid, wavelength * wavelength_grid, grid_points
)
y = np.linspace(
    -wavelength * wavelength_grid, wavelength * wavelength_grid, grid_points
)
x, y = np.meshgrid(x, y)
rho = np.sqrt(x**2 + y**2)
phi = np.atan2(y, x)
phi_lin = np.linspace(0, 2.0 * np.pi, 1000)

E0 = 1.0

Ez_i = E0 * np.exp(-1.0j * k * x)

N = 50
Ez_s_n = np.zeros((2 * N + 1,) + x.shape, dtype=np.complex128)
sig_n = np.zeros((2 * N + 1, len(phi_lin)), dtype=np.complex128)
for n in range(-N, N + 1):
    i = n + N
    Ez_s_n[i][rho > a] = (
        ((-1.0j) ** n * jn(n, k * a) / hankel2(n, k * a))
        * hankel2(n, k * rho)
        * np.exp(1.0j * n * phi)
    )[rho > a]
    sig_n[i] = (
        ((-1.0j) ** n * jn(n, k * a) / hankel2(n, k * a))
        * np.exp(1.0j * (2 * n + 1) * np.pi / 4)
        * np.exp(1.0j * n * phi_lin)
    )

Ez_s = -E0 * np.sum(Ez_s_n, axis=0)
Ez_s[rho <= a] = -Ez_i[rho <= a]

Ez_t = Ez_i + Ez_s

sig = ((4.0 * a) / (k * a)) * np.abs(np.sum(sig_n, axis=0)) ** 2

fig = plt.figure(figsize=(8, 6), layout="constrained")
ax = [
    [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)],
    [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4, projection="polar")],
]
mesh_00 = ax[0][0].pcolormesh(x, y, np.real(Ez_i), cmap="bwr")
mesh_01 = ax[0][1].pcolormesh(x, y, np.real(Ez_s), cmap="bwr")
mesh_10 = ax[1][0].pcolormesh(x, y, np.real(Ez_t), cmap="bwr")
ax[1][1].plot(phi_lin, np.log10((sig / (4.0 * wavelength / 2))))
fig.colorbar(mesh_00)
fig.colorbar(mesh_01)
fig.colorbar(mesh_10)

np.savez(
    "compare.npz",
    **{"phi": phi_lin, "result": np.log10((sig / (4.0 * wavelength / 2)))},
)

plt.show()
