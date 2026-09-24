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
    result = np.exp(-0.5 * (t / t_sig) ** 2) * np.sin(2.0 * np.pi * frequency * t)
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

        self.x_Hx = (self.x_Ez + self.delta_x / 2)[:-1]
        self.y_Hx = self.y_Ez

        self.x_Hy = self.x_Ez
        self.y_Hy = (self.y_Ez + self.delta_y / 2)[:-1]

        self.x_Ez, self.y_Ez = np.meshgrid(self.x_Ez, self.y_Ez)
        self.x_Hx, self.y_Hx = np.meshgrid(self.x_Hx, self.y_Hx)
        self.x_Hy, self.y_Hy = np.meshgrid(self.x_Hy, self.y_Hy)

        self.delta_t = stability_condition_2d(
            self.phase_velocity, self.delta_x, self.delta_y
        )

        # setup materials
        self.epsilon = np.ones_like(self.x_Ez) * self.epsilon_background

        for x0, y0, radius, eps_r in self.defaults["permittivity_circles"]:
            xc = self.x_min + (self.x_max - self.x_min) * x0
            yc = self.y_min + (self.y_max - self.y_min) * y0
            rr = (self.x_max - self.x_min) * radius
            self.epsilon[(self.x_Ez - xc) ** 2 + (self.y_Ez - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )

        # plot epsilon
        # fig, ax = plt.subplots()
        # mesh = ax.pcolormesh(self.x_Ez, self.y_Ez, self.epsilon)
        # fig.colorbar(mesh)
        # plt.show()
        # exit(0)

        self.mu = np.ones_like(self.x_Ez) * self.mu_background
        self.sigma_x = np.zeros_like(self.x_Ez)
        self.sigma_y = np.zeros_like(self.x_Ez)

        x_line = self.x_Ez[0, :]
        y_line = self.y_Ez[:, 0]

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
            x_line_left = x_line[x_line < x_left]
            x_line_left = (x_line_left - np.min(x_line_left)) / (
                np.max(x_line_left) - np.min(x_line_left)
            )
            x_line_left = x_line_left[::-1]
            x_line_left = x_line_left**falloff_exponent
            x_line_left *= sigma_max
            for i in range(len(x_line)):
                self.sigma_x[i, :][x_line < x_left] = x_line_left
        elif self.defaults["left_boundary"] == "PEC":
            pass
        else:
            print("defaults.left_boundary can be either PML or PEC. Assuming PEC")

        # x right
        if self.defaults["right_boundary"] == "PML":
            x_line_right = x_line[x_line > x_right]
            x_line_right = (x_line_right - np.min(x_line_right)) / (
                np.max(x_line_right) - np.min(x_line_right)
            )
            x_line_right = x_line_right**falloff_exponent
            x_line_right *= sigma_max
            for i in range(len(x_line)):
                self.sigma_x[i, :][x_line > x_right] = x_line_right
        elif self.defaults["right_boundary"] == "PEC":
            pass
        else:
            print("defaults.right_boundary can be either PML or PEC. Assuming PEC")

        # y top
        if self.defaults["top_boundary"] == "PML":
            y_line_top = y_line[y_line > y_top]
            y_line_top = (y_line_top - np.min(y_line_top)) / (
                np.max(y_line_top) - np.min(y_line_top)
            )
            y_line_top = y_line_top**falloff_exponent
            y_line_top *= sigma_max
            for i in range(len(y_line)):
                self.sigma_y[:, i][y_line > y_top] = y_line_top
        elif self.defaults["top_boundary"] == "PEC":
            pass
        else:
            print("defaults.top_boundary can be either PML or PEC. Assuming PEC")

        # y bot
        if self.defaults["bot_boundary"] == "PML":
            y_line_bot = y_line[y_line < y_bot]
            y_line_bot = (y_line_bot - np.min(y_line_bot)) / (
                np.max(y_line_bot) - np.min(y_line_bot)
            )
            y_line_bot = y_line_bot[::-1]
            y_line_bot = y_line_bot**falloff_exponent
            y_line_bot *= sigma_max
            for i in range(len(y_line)):
                self.sigma_y[:, i][y_line < y_bot] = y_line_bot
        elif self.defaults["bot_boundary"] == "PEC":
            pass
        else:
            print("defaults.bot_boundary can be either PML or PEC. Assuming PEC")

        # plot sigmas
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        # mesh1 = ax[0].pcolormesh(self.x_Ez, self.y_Ez, self.sigma_x, cmap="jet")
        # fig.colorbar(mesh1)
        # mesh2 = ax[1].pcolormesh(self.x_Ez, self.y_Ez, self.sigma_y, cmap="jet")
        # fig.colorbar(mesh2)
        # plt.show()
        # exit(0)

        # derived materials
        self.alpha_x = self.epsilon / self.delta_t - self.sigma_x / 2
        self.alpha_y = self.epsilon / self.delta_t - self.sigma_y / 2
        self.beta_x = self.epsilon / self.delta_t + self.sigma_x / 2
        self.beta_y = self.epsilon / self.delta_t + self.sigma_y / 2

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
        self.Jz[self.yidx_Jz, self.xidx_Jz] = self.Jz_xidx_yidx[time_index]

        x1 = 1
        x2 = self.Ez.shape[1] - 1
        y1 = 1
        y2 = self.Ez.shape[0] - 1

        self.Jz[self.yidx_Jz, self.xidx_Jz] = self.Jz_xidx_yidx[time_index]

        # H update
        self.Hx[y1 : y2 - 1, x1:x2] = (1.0 / self.beta_y[y1 : y2 - 1, x1:x2]) * (
            self.alpha_y[y1 : y2 - 1, x1:x2] * self.Hx[y1 : y2 - 1, x1:x2]
            - (
                self.epsilon[y1 : y2 - 1, x1:x2]
                / (self.mu[y1 : y2 - 1, x1:x2] * self.delta_y)
            )
            * (self.Ez[y1 + 1 : y2, x1:x2] - self.Ez[y1 : y2 - 1, x1:x2])
        )
        self.Hy[y1:y2, x1 : x2 - 1] = (1.0 / self.beta_x[y1:y2, x1 : x2 - 1]) * (
            self.alpha_x[y1:y2, x1 : x2 - 1] * self.Hy[y1:y2, x1 : x2 - 1]
            + (
                self.epsilon[y1:y2, x1 : x2 - 1]
                / (self.mu[y1:y2, x1 : x2 - 1] * self.delta_x)
            )
            * (self.Ez[y1:y2, x1 + 1 : x2] - self.Ez[y1:y2, x1 : x2 - 1])
        )

        # E update
        self.Ez_sx[y1:y2, x1:x2] = (1.0 / self.beta_x[y1:y2, x1:x2]) * (
            self.alpha_x[y1:y2, x1:x2] * self.Ez_sx[y1:y2, x1:x2]
            + (1.0 / self.delta_x)
            * (self.Hy[y1:y2, x1:x2] - self.Hy[y1:y2, x1 - 1 : x2 - 1])
            - self.Jz[y1:y2, x1:x2] / 2.0
        )
        self.Ez_sy[y1:y2, x1:x2] = (1.0 / self.beta_y[y1:y2, x1:x2]) * (
            self.alpha_y[y1:y2, x1:x2] * self.Ez_sy[y1:y2, x1:x2]
            - (1.0 / self.delta_y)
            * (self.Hx[y1:y2, x1:x2] - self.Hx[y1 - 1 : y2 - 1, x1:x2])
            - self.Jz[y1:y2, x1:x2] / 2.0
        )
        self.Ez = self.Ez_sx + self.Ez_sy

        # self.Ez = Ez_pw(self.frequency, self.x_Ez, self.t[time_index], 1.0)
        # self.Hy = Hy_pw(self.frequency, self.x_Hy, self.t[time_index], 1.0)

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
