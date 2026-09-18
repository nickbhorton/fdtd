import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches
from scipy.constants import epsilon_0, mu_0

gold_conductivity = 4.1e7


def compute_wavelength(frequency, phase_velocity):
    return phase_velocity / frequency


def materials_to_phase_velocity(epsilon, mu):
    return 1 / np.sqrt(epsilon * mu)


def stability_condition_2d(phase_velocity, delta_x, delta_y):
    return 1 / (phase_velocity * np.sqrt(1 / delta_x**2 + 1 / delta_y**2))


frequency = 10e9

epsilon_background = 1.0 * epsilon_0
mu_background = 1.0 * mu_0
sigma_background = 0.0

phase_velocity = materials_to_phase_velocity(epsilon_background, mu_background)
wavelength = compute_wavelength(frequency, phase_velocity)

delta_x = wavelength / 30
delta_y = wavelength / 30

print("Numerical dispersion", np.pi**2 / 8 * (delta_x / wavelength) ** 2)

x_min = 0
x_max = 10 * wavelength
y_min = 0
y_max = 10 * wavelength

x_Ez = np.arange(x_min, x_max, delta_x)
y_Ez = np.arange(y_min, y_max, delta_y)

x_Hx = (x_Ez + delta_x / 2)[:-1]
y_Hx = y_Ez

x_Hy = x_Ez
y_Hy = (y_Ez + delta_y / 2)[:-1]

x_Ez, y_Ez = np.meshgrid(x_Ez, y_Ez)
x_Hx, y_Hx = np.meshgrid(x_Hx, y_Hx)
x_Hy, y_Hy = np.meshgrid(x_Hy, y_Hy)

xidx_Jz = int(x_Ez.shape[0] / 10)
yidx_Jz = int(x_Ez.shape[0] / 2)

# setup materials
sigma = np.zeros_like(x_Ez) * sigma_background
epsilon = np.ones_like(x_Ez) * epsilon_background
mu = np.ones_like(x_Ez) * mu_background

# ((x, y) (w, h))
conductor_rects = [((3, 7.0), (1, 3)), ((3, 0), (1, 3)), ((3, 3.5), (1, 3))]
for (x, y), (w, h) in conductor_rects:
    sigma[
        (x_Ez > x * wavelength)
        & (x_Ez < (x + w) * wavelength)
        & (y_Ez > y * wavelength)
        & (y_Ez < (y + h) * wavelength)
    ] = gold_conductivity

# permittivity_rects = [((6.5, 3), (1, 4))]
permittivity_rects = []
for (x, y), (w, h) in permittivity_rects:
    epsilon[
        (x_Ez > x * wavelength)
        & (x_Ez < (x + w) * wavelength)
        & (y_Ez > y * wavelength)
        & (y_Ez < (y + h) * wavelength)
    ] = 5.0 * epsilon_0

delta_t = stability_condition_2d(phase_velocity, delta_x, delta_y)

alpha = np.ones_like(x_Ez) * (epsilon / delta_t) - sigma / 2
beta = np.ones_like(x_Ez) * (epsilon / delta_t) + sigma / 2

t_period = np.arange(0.0, 1 / frequency, delta_t)
t = np.arange(0.0, 40 / frequency, delta_t)
Jz_xidx_yidx = np.zeros_like(t)
Jz_xidx_yidx[0 : len(t_period)] = np.sin(t_period * 2 * np.pi * frequency)

# plot pulse

fig, ax = plt.subplots()
ax.plot(t, Jz_xidx_yidx)
plt.show()

# plot grid

# fig, ax = plt.subplots()
# ax.scatter(x_Ez.flatten(), y_Ez.flatten(), color="black", label="Ez grid")
# ax.scatter(x_Hx.flatten(), y_Hx.flatten(), color="red", label="Hx grid")
# ax.scatter(x_Hy.flatten(), y_Hy.flatten(), color="blue", label="Hy grid")
# ax.legend()
# plt.show()


class ArrayAnimation:
    def __init__(self):
        self.Jz = np.zeros_like(x_Ez)
        self.Ez = np.zeros_like(x_Ez)
        self.Hx = np.zeros_like(x_Hx)
        self.Hy = np.zeros_like(x_Hy)

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_box_aspect((y_max - y_min) / (x_max - x_min))

        self.mesh = self.ax.pcolormesh(x_Ez, y_Ez, self.Ez, cmap="bwr")
        self.cbar = self.fig.colorbar(self.mesh)

        for i in range(len(conductor_rects)):
            self.ax.add_patch(
                patches.Rectangle(
                    (
                        conductor_rects[i][0][0] * wavelength,
                        conductor_rects[i][0][1] * wavelength,
                    ),
                    conductor_rects[i][1][0] * wavelength,
                    conductor_rects[i][1][1] * wavelength,
                    linewidth=2,
                    edgecolor="none",
                    facecolor="grey",
                    alpha=0.3,
                )
            )
        for i in range(len(permittivity_rects)):
            self.ax.add_patch(
                patches.Rectangle(
                    (
                        permittivity_rects[i][0][0] * wavelength,
                        permittivity_rects[i][0][1] * wavelength,
                    ),
                    permittivity_rects[i][1][0] * wavelength,
                    permittivity_rects[i][1][1] * wavelength,
                    linewidth=2,
                    edgecolor="none",
                    facecolor="green",
                    alpha=0.3,
                )
            )
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def update(self, frame):
        self.Jz[yidx_Jz, xidx_Jz] = Jz_xidx_yidx[frame]
        self.Ez[1:-1, 1:-1] = (
            1.0
            / beta[1:-1, 1:-1]
            * (
                alpha[1:-1, 1:-1] * self.Ez[1:-1, 1:-1]
                + 1 / delta_x * (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1])
                - 1 / delta_y * (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])
                - self.Jz[1:-1, 1:-1]
            )
        )
        self.Hx = self.Hx - delta_t / (mu[:, 1:] * delta_y) * (
            self.Ez[:, 1:] - self.Ez[:, :-1]
        )
        self.Hy = self.Hy + delta_t / (mu[1:, :] * delta_x) * (
            self.Ez[1:, :] - self.Ez[:-1, :]
        )
        self.mesh.set_clim(
            vmin=-self.Ez[x_Ez > 4.0 * wavelength].max(),
            vmax=self.Ez[x_Ez > 4.0 * wavelength].max(),
        )
        self.cbar.update_normal(self.mesh)
        self.mesh.set_array(self.Ez)
        return (self.mesh,)

    def animate(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(t), interval=1, blit=False
        )
        plt.show()


a = ArrayAnimation()
a.animate()
