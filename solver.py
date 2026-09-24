import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0

from lib import (
    compute_wavelength,
    materials_to_phase_velocity,
    stability_condition_2d,
)
from timeseries import TimeSeries


def mgpulse(t, t_sig, frequency):
    # result = np.exp(-0.5 * (t / t_sig) ** 2) * np.sin(2.0 * np.pi * frequency * t)
    result = np.sin(2.0 * np.pi * frequency * t)
    return result


def Ez_pw(frequency, x, t, eps_r, mu_r=1.0, E0=1.0):
    v_phase = materials_to_phase_velocity(eps_r * epsilon_0, mu_0 * mu_r)
    coord = t - x / v_phase
    return E0 * mgpulse(coord, 1 / frequency, frequency)


def Hy_pw(frequency, x, t, eps_r, mu_r=1.0, E0=1.0):
    v_phase = materials_to_phase_velocity(eps_r * epsilon_0, mu_0 * mu_r)
    coord = t - x / v_phase
    eta = np.sqrt(mu_r * mu_0 / eps_r / epsilon_0)
    return -E0 / eta * mgpulse(coord, 1 / frequency, frequency)


class Solver:
    def __init__(self, defaults_path: Path):
        self.defaults = json.loads(defaults_path.read_text())

        self.frequency = self.defaults["frequency"]

        self.epsilon_background = self.defaults["background_epsilon_r"] * epsilon_0
        self.mu_background = self.defaults["background_mu_r"] * mu_0
        self.sigma_background = self.defaults["background_sigma"]

        self.phase_velocity = materials_to_phase_velocity(
            self.epsilon_background, self.mu_background
        )
        self.wavelength = compute_wavelength(self.frequency, self.phase_velocity)

        # gui option initialization
        self.picoseconds = self.defaults["picoseconds"]
        self.delta_x = self.wavelength / self.defaults["wavelength_discretization_x"]
        self.delta_y = self.wavelength / self.defaults["wavelength_discretization_y"]
        self.wavelengths_x = self.defaults["wavelengths_in_x"]
        self.wavelengths_y = self.defaults["wavelengths_in_y"]

        self.setup_solver()
        self.convert_to_save_data_to_time_series()

    def setup_solver(self):
        # first delete any TimeSeries that exist
        self.time_series_array = []

        self.x_min = 0
        self.x_max = self.wavelengths_x * self.wavelength
        self.y_min = 0
        self.y_max = self.wavelengths_y * self.wavelength

        self.x_Ez = np.arange(self.x_min, self.x_max, self.delta_x)
        self.y_Ez = np.arange(self.y_min, self.y_max, self.delta_y)
        self.x_Hx = self.x_Ez
        self.y_Hx = (self.y_Ez + self.delta_y / 2)[:-1]
        self.x_Hy = (self.x_Ez + self.delta_x / 2)[:-1]
        self.y_Hy = self.y_Ez

        self.x_Ez, self.y_Ez = np.meshgrid(self.x_Ez, self.y_Ez)
        self.x_Hx, self.y_Hx = np.meshgrid(self.x_Hx, self.y_Hx)
        self.x_Hy, self.y_Hy = np.meshgrid(self.x_Hy, self.y_Hy)

        self.delta_t = stability_condition_2d(
            self.phase_velocity, self.delta_x, self.delta_y
        )

        # setup materials
        self.epsilon_Ez = np.ones_like(self.x_Ez) * self.epsilon_background
        self.epsilon_Hx = np.ones_like(self.x_Hx) * self.epsilon_background
        self.epsilon_Hy = np.ones_like(self.x_Hy) * self.epsilon_background

        for x0, y0, radius, eps_r in self.defaults["permittivity_circles"]:
            xc = self.x_min + (self.x_max - self.x_min) * x0
            yc = self.y_min + (self.y_max - self.y_min) * y0
            rr = (self.x_max - self.x_min) * radius
            self.epsilon_Ez[(self.x_Ez - xc) ** 2 + (self.y_Ez - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )
            self.epsilon_Hx[(self.x_Hx - xc) ** 2 + (self.y_Hx - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )
            self.epsilon_Hy[(self.x_Hy - xc) ** 2 + (self.y_Hy - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )

        # plot epsilon
        # fig, ax = plt.subplots()
        # mesh = ax.pcolormesh(self.x_Ez, self.y_Ez, self.epsilon)
        # fig.colorbar(mesh)
        # plt.show()
        # exit(0)

        self.mu_Ez = np.ones_like(self.x_Ez) * self.mu_background
        self.mu_Hx = np.ones_like(self.x_Hx) * self.mu_background
        self.mu_Hy = np.ones_like(self.x_Hy) * self.mu_background

        self.sigma_x_Ez = np.zeros_like(self.x_Ez)
        self.sigma_y_Ez = np.zeros_like(self.x_Ez)
        self.sigma_x_Hx = np.zeros_like(self.x_Hx)
        self.sigma_y_Hx = np.zeros_like(self.x_Hx)
        self.sigma_x_Hy = np.zeros_like(self.x_Hy)
        self.sigma_y_Hy = np.zeros_like(self.x_Hy)

        x_line_Ez = self.x_Ez[0, :]
        y_line_Ez = self.y_Ez[:, 0]
        x_line_Hx = self.x_Hx[0, :]
        y_line_Hx = self.y_Hx[:, 0]
        x_line_Hy = self.x_Hy[0, :]
        y_line_Hy = self.y_Hy[:, 0]

        percent_inset = self.defaults["PML_inset_as_uniform"]
        x_left = (self.x_max - self.x_min) * percent_inset
        x_right = (self.x_max - self.x_min) * (1 - percent_inset)
        y_bot = (self.y_max - self.y_min) * percent_inset
        y_top = (self.y_max - self.y_min) * (1 - percent_inset)

        # calculate sigma curves for each side
        sigma_max = self.defaults["sigma_max"]
        falloff_exponent = self.defaults["sigma_falloff_exponent"]
        # x left
        if self.defaults["left_boundary"] == "PML":
            x_line_left_Ez = x_line_Ez[x_line_Ez < x_left]
            x_line_left_Ez = (x_line_left_Ez - np.min(x_line_left_Ez)) / (
                np.max(x_line_left_Ez) - np.min(x_line_left_Ez)
            )
            x_line_left_Ez = x_line_left_Ez[::-1]
            x_line_left_Ez = x_line_left_Ez**falloff_exponent
            x_line_left_Ez *= sigma_max
            x_line_left_Hx = x_line_Hx[x_line_Hx < x_left]
            x_line_left_Hx = (x_line_left_Hx - np.min(x_line_left_Hx)) / (
                np.max(x_line_left_Hx) - np.min(x_line_left_Hx)
            )
            x_line_left_Hx = x_line_left_Hx[::-1]
            x_line_left_Hx = x_line_left_Hx**falloff_exponent
            x_line_left_Hx *= sigma_max
            x_line_left_Hy = x_line_Hy[x_line_Hy < x_left]
            x_line_left_Hy = (x_line_left_Hy - np.min(x_line_left_Hy)) / (
                np.max(x_line_left_Hy) - np.min(x_line_left_Hy)
            )
            x_line_left_Hy = x_line_left_Hy[::-1]
            x_line_left_Hy = x_line_left_Hy**falloff_exponent
            x_line_left_Hy *= sigma_max
            for i in range(self.sigma_x_Ez.shape[0]):
                self.sigma_x_Ez[i, :][x_line_Ez < x_left] = x_line_left_Ez
            for i in range(self.sigma_x_Hx.shape[0]):
                self.sigma_x_Hx[i, :][x_line_Hx < x_left] = x_line_left_Hx
            for i in range(self.sigma_x_Hy.shape[0]):
                self.sigma_x_Hy[i, :][x_line_Hy < x_left] = x_line_left_Hy
        elif self.defaults["left_boundary"] == "PEC":
            pass
        else:
            print("defaults.left_boundary can be either PML or PEC. Assuming PEC")

        # x right
        if self.defaults["right_boundary"] == "PML":
            x_line_right_Ez = x_line_Ez[x_line_Ez > x_right]
            x_line_right_Ez = (x_line_right_Ez - np.min(x_line_right_Ez)) / (
                np.max(x_line_right_Ez) - np.min(x_line_right_Ez)
            )
            x_line_right_Ez = x_line_right_Ez**falloff_exponent
            x_line_right_Ez *= sigma_max
            x_line_right_Hx = x_line_Hx[x_line_Hx > x_right]
            x_line_right_Hx = (x_line_right_Hx - np.min(x_line_right_Hx)) / (
                np.max(x_line_right_Hx) - np.min(x_line_right_Hx)
            )
            x_line_right_Hx = x_line_right_Hx**falloff_exponent
            x_line_right_Hx *= sigma_max
            x_line_right_Hy = x_line_Hy[x_line_Hy > x_right]
            x_line_right_Hy = (x_line_right_Hy - np.min(x_line_right_Hy)) / (
                np.max(x_line_right_Hy) - np.min(x_line_right_Hy)
            )
            x_line_right_Hy = x_line_right_Hy**falloff_exponent
            x_line_right_Hy *= sigma_max
            for i in range(self.sigma_x_Ez.shape[0]):
                self.sigma_x_Ez[i, :][x_line_Ez > x_right] = x_line_right_Ez
            for i in range(self.sigma_x_Hx.shape[0]):
                self.sigma_x_Hx[i, :][x_line_Hx > x_right] = x_line_right_Hx
            for i in range(self.sigma_x_Hy.shape[0]):
                self.sigma_x_Hy[i, :][x_line_Hy > x_right] = x_line_right_Hy
        elif self.defaults["right_boundary"] == "PEC":
            pass
        else:
            print("defaults.right_boundary can be either PML or PEC. Assuming PEC")

        # y top
        if self.defaults["top_boundary"] == "PML":
            y_line_top_Ez = y_line_Ez[y_line_Ez > y_top]
            y_line_top_Ez = (y_line_top_Ez - np.min(y_line_top_Ez)) / (
                np.max(y_line_top_Ez) - np.min(y_line_top_Ez)
            )
            y_line_top_Ez = y_line_top_Ez**falloff_exponent
            y_line_top_Ez *= sigma_max
            y_line_top_Hx = y_line_Hx[y_line_Hx > y_top]
            y_line_top_Hx = (y_line_top_Hx - np.min(y_line_top_Hx)) / (
                np.max(y_line_top_Hx) - np.min(y_line_top_Hx)
            )
            y_line_top_Hx = y_line_top_Hx**falloff_exponent
            y_line_top_Hx *= sigma_max
            y_line_top_Hy = y_line_Hy[y_line_Hy > y_top]
            y_line_top_Hy = (y_line_top_Hy - np.min(y_line_top_Hy)) / (
                np.max(y_line_top_Hy) - np.min(y_line_top_Hy)
            )
            y_line_top_Hy = y_line_top_Hy**falloff_exponent
            y_line_top_Hy *= sigma_max
            for i in range(self.sigma_x_Ez.shape[1]):
                self.sigma_y_Ez[:, i][y_line_Ez > y_top] = y_line_top_Ez
            for i in range(self.sigma_x_Hx.shape[1]):
                self.sigma_y_Hx[:, i][y_line_Hx > y_top] = y_line_top_Hx
            for i in range(self.sigma_x_Hy.shape[1]):
                self.sigma_y_Hy[:, i][y_line_Hy > y_top] = y_line_top_Hy
        elif self.defaults["top_boundary"] == "PEC":
            pass
        else:
            print("defaults.top_boundary can be either PML or PEC. Assuming PEC")

        # y bot
        if self.defaults["bot_boundary"] == "PML":
            y_line_bot_Ez = y_line_Ez[y_line_Ez < y_bot]
            y_line_bot_Ez = (y_line_bot_Ez - np.min(y_line_bot_Ez)) / (
                np.max(y_line_bot_Ez) - np.min(y_line_bot_Ez)
            )
            y_line_bot_Ez = y_line_bot_Ez[::-1]
            y_line_bot_Ez = y_line_bot_Ez**falloff_exponent
            y_line_bot_Ez *= sigma_max
            y_line_bot_Hx = y_line_Hx[y_line_Hx < y_bot]
            y_line_bot_Hx = (y_line_bot_Hx - np.min(y_line_bot_Hx)) / (
                np.max(y_line_bot_Hx) - np.min(y_line_bot_Hx)
            )
            y_line_bot_Hx = y_line_bot_Hx[::-1]
            y_line_bot_Hx = y_line_bot_Hx**falloff_exponent
            y_line_bot_Hx *= sigma_max
            y_line_bot_Hy = y_line_Hy[y_line_Hy < y_bot]
            y_line_bot_Hy = (y_line_bot_Hy - np.min(y_line_bot_Hy)) / (
                np.max(y_line_bot_Hy) - np.min(y_line_bot_Hy)
            )
            y_line_bot_Hy = y_line_bot_Hy[::-1]
            y_line_bot_Hy = y_line_bot_Hy**falloff_exponent
            y_line_bot_Hy *= sigma_max
            for i in range(self.sigma_x_Ez.shape[1]):
                self.sigma_y_Ez[:, i][y_line_Ez < y_bot] = y_line_bot_Ez
            for i in range(self.sigma_x_Hx.shape[1]):
                self.sigma_y_Hx[:, i][y_line_Hx < y_bot] = y_line_bot_Hx
            for i in range(self.sigma_x_Hy.shape[1]):
                self.sigma_y_Hy[:, i][y_line_Hy < y_bot] = y_line_bot_Hy
        elif self.defaults["bot_boundary"] == "PEC":
            pass
        else:
            print("defaults.bot_boundary can be either PML or PEC. Assuming PEC")

        # derived materials
        self.alpha_x_Ez = self.epsilon_Ez / self.delta_t - self.sigma_x_Ez / 2
        self.alpha_y_Ez = self.epsilon_Ez / self.delta_t - self.sigma_y_Ez / 2
        self.beta_x_Ez = self.epsilon_Ez / self.delta_t + self.sigma_x_Ez / 2
        self.beta_y_Ez = self.epsilon_Ez / self.delta_t + self.sigma_y_Ez / 2
        self.alpha_x_Hx = self.epsilon_Hx / self.delta_t - self.sigma_x_Hx / 2
        self.alpha_y_Hx = self.epsilon_Hx / self.delta_t - self.sigma_y_Hx / 2
        self.beta_x_Hx = self.epsilon_Hx / self.delta_t + self.sigma_x_Hx / 2
        self.beta_y_Hx = self.epsilon_Hx / self.delta_t + self.sigma_y_Hx / 2
        self.alpha_x_Hy = self.epsilon_Hy / self.delta_t - self.sigma_x_Hy / 2
        self.alpha_y_Hy = self.epsilon_Hy / self.delta_t - self.sigma_y_Hy / 2
        self.beta_x_Hy = self.epsilon_Hy / self.delta_t + self.sigma_x_Hy / 2
        self.beta_y_Hy = self.epsilon_Hy / self.delta_t + self.sigma_y_Hy / 2

        # setup current stuff
        self.xidx_Jz = int(self.x_Ez.shape[0] / 2)
        self.yidx_Jz = int(self.x_Ez.shape[0] / 2)
        self.t = np.arange(0.0, self.picoseconds * 1e-12, self.delta_t)
        self.period = 1 / self.frequency
        self.Jz_xidx_yidx = mgpulse(self.t, self.period, self.frequency)

        # relocate temp fields
        self.Jz = np.zeros_like(self.x_Ez)
        self.Ez = np.zeros_like(self.x_Ez)
        self.Ez_sx = np.zeros_like(self.x_Ez)
        self.Ez_sy = np.zeros_like(self.x_Ez)
        self.Hx = np.zeros_like(self.x_Hx)
        self.Hy = np.zeros_like(self.x_Hy)

        # reallocated to_save
        self.Ez_to_save = np.zeros((len(self.t),) + self.Ez_sx.shape, dtype=np.float32)
        self.Hx_to_save = np.zeros((len(self.t),) + self.Hx.shape, dtype=np.float32)
        self.Hy_to_save = np.zeros((len(self.t),) + self.Hy.shape, dtype=np.float32)

    def get_solution_memory_size(self):
        floats_2d_slice = len(np.arange(self.x_min, self.x_max, self.delta_x)) * len(
            np.arange(self.y_min, self.y_max, self.delta_y)
        )
        # 3 for Ez Hy Hx and 4 bytes per float32
        return floats_2d_slice * len(self.t) * 3 * 4

    def update_t(self):
        self.delta_t = stability_condition_2d(
            self.phase_velocity, self.delta_x, self.delta_y
        )
        self.t = np.arange(0.0, self.picoseconds * 1e-12, self.delta_t)

    def set_delta_x(self, new_delta_x):
        self.delta_x = new_delta_x
        self.update_t()

    def set_delta_y(self, new_delta_y):
        self.delta_y = new_delta_y
        self.update_t()

    def set_picoseconds(self, picoseconds):
        self.picoseconds = picoseconds
        self.update_t()

    def set_wavelengths_in_x(self, wavelengths_in_x):
        self.wavelengths_x = wavelengths_in_x
        self.update_xy_lim()

    def set_wavelengths_in_y(self, wavelengths_in_y):
        self.wavelengths_y = wavelengths_in_y
        self.update_xy_lim()

    def update_xy_lim(self):
        self.x_min = 0
        self.x_max = self.wavelengths_x * self.wavelength
        self.y_min = 0
        self.y_max = self.wavelengths_y * self.wavelength

    def field_time_step(self, time_index):
        # Insert current source
        # self.Jz[self.yidx_Jz, self.xidx_Jz] = self.Jz_xidx_yidx[time_index]

        box_size = self.defaults["PML_inset_as_uniform"] + 0.3
        x_left = self.x_min + (self.x_max - self.x_min) * box_size
        x_right = self.x_min + (self.x_max - self.x_min) * (1.0 - box_size)
        y_bot = self.y_min + (self.y_max - self.y_min) * box_size
        y_top = self.y_min + (self.y_max - self.y_min) * (1.0 - box_size)

        # mask creation
        x_idx_left = np.abs(self.x_Ez[0, :] - x_left).argmin()
        x_idx_right = np.abs(self.x_Ez[0, :] - x_right).argmin()
        y_idx_bot = np.abs(self.y_Ez[:, 0] - y_bot).argmin()
        y_idx_top = np.abs(self.y_Ez[:, 0] - y_top).argmin()
        # print(x_idx_left, x_idx_right, y_idx_bot, y_idx_top)

        # Hx update
        Hx_prev = self.Hx.copy()
        self.Hx[1:-1, :] = (1.0 / self.beta_y_Hx[1:-1, :]) * (
            self.alpha_y_Hx[1:-1, :] * self.Hx[1:-1, :]
            - (self.epsilon_Hx[1:-1, :] / (self.mu_Hx[1:-1, :] * self.delta_y))
            * (self.Ez[2:-1, :] - self.Ez[1:-2, :])
        )
        # surface Hx update bot
        self.Hx[y_idx_bot - 1, x_idx_left : x_idx_right + 1] = (
            1.0 / self.beta_y_Hx[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
        ) * (
            self.alpha_y_Hx[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
            * Hx_prev[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
            - (
                self.epsilon_Hx[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
                / (
                    self.mu_Hx[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
                    * self.delta_y
                )
            )
            * (
                (
                    self.Ez[y_idx_bot, x_idx_left : x_idx_right + 1]
                    - Ez_pw(
                        self.frequency,
                        self.x_Ez[y_idx_bot, x_idx_left : x_idx_right + 1],
                        self.t[time_index] - self.delta_t / 2,
                        1.0,
                    )
                )
                - self.Ez[y_idx_bot - 1, x_idx_left : x_idx_right + 1]
            )
        )
        # surface Hx update top
        self.Hx[y_idx_top, x_idx_left : x_idx_right + 1] = (
            1.0 / self.beta_y_Hx[y_idx_top, x_idx_left : x_idx_right + 1]
        ) * (
            self.alpha_y_Hx[y_idx_top, x_idx_left : x_idx_right + 1]
            * Hx_prev[y_idx_top, x_idx_left : x_idx_right + 1]
            - (
                self.epsilon_Hx[y_idx_top, x_idx_left : x_idx_right + 1]
                / (self.mu_Hx[y_idx_top, x_idx_left : x_idx_right + 1] * self.delta_y)
            )
            * (
                self.Ez[y_idx_top + 1, x_idx_left : x_idx_right + 1]
                - (
                    self.Ez[y_idx_top, x_idx_left : x_idx_right + 1]
                    - Ez_pw(
                        self.frequency,
                        self.x_Ez[y_idx_top - 1, x_idx_left : x_idx_right + 1],
                        self.t[time_index] - self.delta_t / 2,
                        1.0,
                    )
                )
            )
        )

        # Hy update
        Hy_prev = self.Hy.copy()
        self.Hy[:, 1:-1] = (1.0 / self.beta_x_Hy[:, 1:-1]) * (
            self.alpha_x_Hy[:, 1:-1] * self.Hy[:, 1:-1]
            + (self.epsilon_Hy[:, 1:-1] / (self.mu_Hy[:, 1:-1] * self.delta_x))
            * (self.Ez[:, 2:-1] - self.Ez[:, 1:-2])
        )
        # surface Hy update left
        self.Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1] = (
            1.0 / self.beta_x_Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
        ) * (
            self.alpha_x_Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
            * Hy_prev[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
            + (
                self.epsilon_Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
                / (self.mu_Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1] * self.delta_x)
            )
            * (
                (
                    self.Ez[y_idx_bot : y_idx_top + 1, x_idx_left]
                    - Ez_pw(
                        self.frequency,
                        self.x_Ez[y_idx_bot : y_idx_top + 1, x_idx_left],
                        self.t[time_index] - self.delta_t / 2,
                        1.0,
                    )
                )
                - self.Ez[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
            )
        )

        # Ez_sx update
        Ez_sx_prev = self.Ez_sx.copy()
        self.Ez_sx[1:-1, 1:-1] = (1.0 / self.beta_x_Ez[1:-1, 1:-1]) * (
            self.alpha_x_Ez[1:-1, 1:-1] * self.Ez_sx[1:-1, 1:-1]
            + (1.0 / self.delta_x) * (self.Hy[1:-1, 1:] - self.Hy[1:-1, :-1])
            - self.Jz[1:-1, 1:-1] / 2.0
        )

        # surface Ez_sz update left
        self.Ez_sx[y_idx_bot : y_idx_top + 1, x_idx_left] = (
            1.0 / self.beta_x_Ez[y_idx_bot : y_idx_top + 1, x_idx_left]
        ) * (
            self.alpha_x_Ez[y_idx_bot : y_idx_top + 1, x_idx_left]
            * Ez_sx_prev[y_idx_bot : y_idx_top + 1, x_idx_left]
            + (1.0 / self.delta_x)
            * (
                self.Hy[y_idx_bot : y_idx_top + 1, x_idx_left]
                - (
                    self.Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1]
                    + Hy_pw(
                        self.frequency,
                        self.x_Hy[y_idx_bot : y_idx_top + 1, x_idx_left - 1],
                        self.t[time_index],
                        1.0,
                    )
                )
            )
            - self.Jz[y_idx_bot : y_idx_top + 1, x_idx_left] / 2.0
        )

        # Ez_sy update
        Ez_sy_prev = self.Ez_sy.copy()
        self.Ez_sy[1:-1, 1:-1] = (1.0 / self.beta_y_Ez[1:-1, 1:-1]) * (
            self.alpha_y_Ez[1:-1, 1:-1] * self.Ez_sy[1:-1, 1:-1]
            - (1.0 / self.delta_y) * (self.Hx[1:, 1:-1] - self.Hx[0:-1, 1:-1])
            - self.Jz[1:-1, 1:-1] / 2.0
        )

        self.Ez = self.Ez_sx + self.Ez_sy

        # self.Ez[np.abs(self.Ez) > 100.0] = 0.0
        # self.Hx[
        #     np.abs(self.Hx)
        #     > 100.0 / np.sqrt(self.mu_background / self.epsilon_background)
        # ] = 100.0
        # self.Hy[
        #     np.abs(self.Hy)
        #     > 100.0 / np.sqrt(self.mu_background / self.epsilon_background)
        # ] = 100.0

        # save new fields to time index
        self.Ez_to_save[time_index] = self.Ez
        self.Hx_to_save[time_index] = self.Hx
        self.Hy_to_save[time_index] = self.Hy

    # THIS WILL DELETE self.[Ez,Hx,Hy]_to_save
    def convert_to_save_data_to_time_series(self):
        self.time_series_array = []
        self.time_series_array.append(
            TimeSeries(
                self.Ez_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        del self.Ez_to_save
        self.time_series_array.append(
            TimeSeries(
                self.Hx_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        del self.Hx_to_save
        self.time_series_array.append(
            TimeSeries(
                self.Hy_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        del self.Hy_to_save
