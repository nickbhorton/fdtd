import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.constants import epsilon_0, mu_0


def compute_wavelength(frequency, phase_velocity):
    return phase_velocity / frequency


def materials_to_phase_velocity(epsilon_r, mu_r):
    return 1 / np.sqrt(epsilon_r * epsilon_0 * mu_r * mu_0)


def stability_condition_2d(phase_velocity, delta_x, delta_y):
    return 1 / (phase_velocity * np.sqrt(1 / delta_x**2 + 1 / delta_y**2))


# fig, ax = plt.subplots()
# ax.scatter(x_Ez.flatten(), y_Ez.flatten(), color="black", label="Ez grid")
# ax.scatter(x_Hx.flatten(), y_Hx.flatten(), color="red", label="Hx grid")
# ax.scatter(x_Hy.flatten(), y_Hy.flatten(), color="blue", label="Hy grid")
# ax.legend()
# plt.show()


frequency = 10e9

epsilon_r = 1.0
mu_r = 1.0
epsilon = epsilon_r * epsilon_0
mu = mu_r * mu_0

phase_velocity = materials_to_phase_velocity(epsilon_r, mu_r)
wavelength = compute_wavelength(frequency, phase_velocity)

delta_x = wavelength / 50
delta_y = wavelength / 50
delta_t = stability_condition_2d(phase_velocity, delta_x, delta_y)

print("Numerical dispersion", np.pi**2 / 8 * (delta_x / wavelength) ** 2)

x_Ez = np.arange(-1 * wavelength, 1 * wavelength, delta_x)
y_Ez = np.arange(-1 * wavelength, 1 * wavelength, delta_y)

x_Hx = (x_Ez + delta_x / 2)[:-1]
y_Hx = y_Ez

x_Hy = x_Ez
y_Hy = (y_Ez + delta_y / 2)[:-1]

x_Ez, y_Ez = np.meshgrid(x_Ez, y_Ez)
x_Hx, y_Hx = np.meshgrid(x_Hx, y_Hx)
x_Hy, y_Hy = np.meshgrid(x_Hy, y_Hy)

xidx_Jz = int(x_Ez.shape[0] / 4)
yidx_Jz = int(x_Ez.shape[1] / 4)

alpha = np.ones_like(x_Ez) * (epsilon / delta_t)
beta = np.ones_like(x_Ez) * (epsilon / delta_t)

t_period = np.arange(0.0, 1 / frequency, delta_t)
t = np.arange(0.0, 40 / frequency, delta_t)
Jz_xidx_yidx = np.zeros_like(t)
Jz_xidx_yidx[0 : len(t_period)] = np.sin(t_period * 2 * np.pi * frequency)

# fig, ax = plt.subplots()
# ax.plot(t, Jz_xidx_yidx)
# plt.show()


class ArrayAnimation:
    def __init__(self):
        self.Jz = np.zeros_like(x_Ez)
        self.Ez = np.zeros_like(x_Ez)
        self.Hx = np.zeros_like(x_Hx)
        self.Hy = np.zeros_like(x_Hy)

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.mesh = self.ax.pcolormesh(x_Ez, y_Ez, self.Ez, cmap="bwr")
        self.cbar = self.fig.colorbar(self.mesh)

    def update(self, frame):
        self.Jz[xidx_Jz, yidx_Jz] = Jz_xidx_yidx[frame]
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
        self.Hx = self.Hx - delta_t / (mu * delta_y) * (
            self.Ez[:, 1:] - self.Ez[:, :-1]
        )
        self.Hy = self.Hy + delta_t / (mu * delta_x) * (
            self.Ez[1:, :] - self.Ez[:-1, :]
        )
        self.mesh.set_clim(vmin=-self.Ez.max(), vmax=self.Ez.max())
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
