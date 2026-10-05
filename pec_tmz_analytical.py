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
a = wavelength / 2
k = 2.0 * np.pi * f / phase_velocity

grid_points = 500
wavelength_grid = 2.0
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

fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
mesh_0 = ax[0].pcolormesh(
    x / wavelength, y / wavelength, np.real(Ez_s), cmap="bwr", vmin=-1.0, vmax=1.0
)
mesh_1 = ax[1].pcolormesh(
    x / wavelength, y / wavelength, np.real(Ez_t), cmap="bwr", vmin=-1.0, vmax=1.0
)

ax[0].set_xticks([-2, 0, 2])
ax[1].set_yticks([-2, 0, 2])

for a in ax.flat:
    a.set_xlabel(r"$x/\lambda$")
    a.set_ylabel(r"$y/\lambda$")

ax[0].set_title("Scattered")
ax[1].set_title("Total")

cbar1 = fig.colorbar(mesh_0)
cbar2 = fig.colorbar(mesh_1)
cbars = [cbar1, cbar2]
for cbar in cbars:
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticks([-1, 0, 1])

fig.savefig("plots/analytical_pec_tmz_Ez_fields.png", dpi=200)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(phi_lin, 20.0 * np.log10(sig), color="black", linewidth=2)

fig.savefig("plots/analytical_pec_tmz_bistatic_echo_width.png", dpi=200)

np.savez(
    "tmz_pec_compare.npz",
    **{"phi": phi_lin, "result": sig},
)

plt.show()
