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

pulse = False
pulse_t_sig = 3.0


def mgpulse(t, t_sig, frequency, t0=0.0):
    result = np.exp(-0.5 * ((t - t0) / t_sig) ** 2) * np.sin(
        2.0 * np.pi * frequency * (t - t0)
    )
    return result


def sinusoid(frequency, t, x, eps_r=1.0, mu_r=1.0):
    w = 2.0 * np.pi * frequency
    k = w * np.sqrt(epsilon_0 * eps_r * mu_0 * mu_r)
    t_ramp = 3.0 / frequency
    envelope = 1.0 if t >= t_ramp else 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))
    return envelope * np.sin(w * t - k * x)


def Ey_pw(frequency, x, t, eps_r, mu_r=1.0, E0=1.0):
    v_phase = materials_to_phase_velocity(eps_r * epsilon_0, mu_0 * mu_r)
    coord = t - x / v_phase
    t_sig = pulse_t_sig / frequency
    t0 = 4.0 * t_sig
    if pulse:
        return E0 * mgpulse(coord, t_sig, frequency, t0)
    return E0 * sinusoid(frequency, t, x)


def Hz_pw(frequency, x, t, eps_r, mu_r=1.0, E0=1.0):
    v_phase = materials_to_phase_velocity(eps_r * epsilon_0, mu_0 * mu_r)
    coord = t - x / v_phase
    t_sig = pulse_t_sig / frequency
    t0 = 4.0 * t_sig
    eta = np.sqrt(mu_r * mu_0 / eps_r / epsilon_0)
    if pulse:
        return E0 / eta * mgpulse(coord, t_sig, frequency, t0)
    return E0 / eta * sinusoid(frequency, t, x)


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

        self.x_Hz = np.arange(self.x_min, self.x_max, self.delta_x)
        self.y_Hz = np.arange(self.y_min, self.y_max, self.delta_y)
        self.x_Ex = self.x_Hz
        self.y_Ex = (self.y_Hz + self.delta_y / 2)[:-1]
        self.x_Ey = (self.x_Hz + self.delta_x / 2)[:-1]
        self.y_Ey = self.y_Hz

        self.x_Hz, self.y_Hz = np.meshgrid(self.x_Hz, self.y_Hz)
        self.x_Ex, self.y_Ex = np.meshgrid(self.x_Ex, self.y_Ex)
        self.x_Ey, self.y_Ey = np.meshgrid(self.x_Ey, self.y_Ey)

        self.delta_t = stability_condition_2d(
            self.phase_velocity, self.delta_x, self.delta_y
        )

        # setup materials
        self.epsilon_Hz = np.ones_like(self.x_Hz) * self.epsilon_background
        self.epsilon_Ex = np.ones_like(self.x_Ex) * self.epsilon_background
        self.epsilon_Ey = np.ones_like(self.x_Ey) * self.epsilon_background

        for x0, y0, radius, eps_r in self.defaults["permittivity_circles"]:
            xc = self.x_min + (self.x_max - self.x_min) * x0
            yc = self.y_min + (self.y_max - self.y_min) * y0
            rr = (self.x_max - self.x_min) * radius
            self.epsilon_Hz[(self.x_Hz - xc) ** 2 + (self.y_Hz - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )
            self.epsilon_Ex[(self.x_Ex - xc) ** 2 + (self.y_Ex - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )
            self.epsilon_Ey[(self.x_Ey - xc) ** 2 + (self.y_Ey - yc) ** 2 < rr**2] = (
                eps_r * epsilon_0
            )

        # plot epsilon
        # fig, ax = plt.subplots()
        # mesh = ax.pcolormesh(self.x_Hz, self.y_Hz, self.epsilon)
        # fig.colorbar(mesh)
        # plt.show()
        # exit(0)

        self.mu_Hz = np.ones_like(self.x_Hz) * self.mu_background
        self.mu_Ex = np.ones_like(self.x_Ex) * self.mu_background
        self.mu_Ey = np.ones_like(self.x_Ey) * self.mu_background

        self.sigma_x_Hz = np.zeros_like(self.x_Hz)
        self.sigma_y_Hz = np.zeros_like(self.x_Hz)
        self.sigma_x_Ex = np.zeros_like(self.x_Ex)
        self.sigma_y_Ex = np.zeros_like(self.x_Ex)
        self.sigma_x_Ey = np.zeros_like(self.x_Ey)
        self.sigma_y_Ey = np.zeros_like(self.x_Ey)

        x_line_Hz = self.x_Hz[0, :]
        y_line_Hz = self.y_Hz[:, 0]
        x_line_Ex = self.x_Ex[0, :]
        y_line_Ex = self.y_Ex[:, 0]
        x_line_Ey = self.x_Ey[0, :]
        y_line_Ey = self.y_Ey[:, 0]

        self.percent_inset_PML = self.defaults["PML_inset_as_uniform"]
        self.x_left = (self.x_max - self.x_min) * self.percent_inset_PML
        self.x_right = (self.x_max - self.x_min) * (1 - self.percent_inset_PML)
        self.y_bot = (self.y_max - self.y_min) * self.percent_inset_PML
        self.y_top = (self.y_max - self.y_min) * (1 - self.percent_inset_PML)

        # calculate sigma curves for each side
        sigma_max = self.defaults["sigma_max"]
        falloff_exponent = self.defaults["sigma_falloff_exponent"]
        # x left
        if self.defaults["left_boundary"] == "PML":
            x_line_left_Hz = x_line_Hz[x_line_Hz < self.x_left]
            x_line_left_Hz = (x_line_left_Hz - np.min(x_line_left_Hz)) / (
                np.max(x_line_left_Hz) - np.min(x_line_left_Hz)
            )
            x_line_left_Hz = x_line_left_Hz[::-1]
            x_line_left_Hz = x_line_left_Hz**falloff_exponent
            x_line_left_Hz *= sigma_max
            x_line_left_Ex = x_line_Ex[x_line_Ex < self.x_left]
            x_line_left_Ex = (x_line_left_Ex - np.min(x_line_left_Ex)) / (
                np.max(x_line_left_Ex) - np.min(x_line_left_Ex)
            )
            x_line_left_Ex = x_line_left_Ex[::-1]
            x_line_left_Ex = x_line_left_Ex**falloff_exponent
            x_line_left_Ex *= sigma_max
            x_line_left_Ey = x_line_Ey[x_line_Ey < self.x_left]
            x_line_left_Ey = (x_line_left_Ey - np.min(x_line_left_Ey)) / (
                np.max(x_line_left_Ey) - np.min(x_line_left_Ey)
            )
            x_line_left_Ey = x_line_left_Ey[::-1]
            x_line_left_Ey = x_line_left_Ey**falloff_exponent
            x_line_left_Ey *= sigma_max
            for i in range(self.sigma_x_Hz.shape[0]):
                self.sigma_x_Hz[i, :][x_line_Hz < self.x_left] = x_line_left_Hz
            for i in range(self.sigma_x_Ex.shape[0]):
                self.sigma_x_Ex[i, :][x_line_Ex < self.x_left] = x_line_left_Ex
            for i in range(self.sigma_x_Ey.shape[0]):
                self.sigma_x_Ey[i, :][x_line_Ey < self.x_left] = x_line_left_Ey
        elif self.defaults["left_boundary"] == "PEC":
            pass
        else:
            print("defaults.left_boundary can be either PML or PEC. Assuming PEC")

        # x right
        if self.defaults["right_boundary"] == "PML":
            x_line_right_Hz = x_line_Hz[x_line_Hz > self.x_right]
            x_line_right_Hz = (x_line_right_Hz - np.min(x_line_right_Hz)) / (
                np.max(x_line_right_Hz) - np.min(x_line_right_Hz)
            )
            x_line_right_Hz = x_line_right_Hz**falloff_exponent
            x_line_right_Hz *= sigma_max
            x_line_right_Ex = x_line_Ex[x_line_Ex > self.x_right]
            x_line_right_Ex = (x_line_right_Ex - np.min(x_line_right_Ex)) / (
                np.max(x_line_right_Ex) - np.min(x_line_right_Ex)
            )
            x_line_right_Ex = x_line_right_Ex**falloff_exponent
            x_line_right_Ex *= sigma_max
            x_line_right_Ey = x_line_Ey[x_line_Ey > self.x_right]
            x_line_right_Ey = (x_line_right_Ey - np.min(x_line_right_Ey)) / (
                np.max(x_line_right_Ey) - np.min(x_line_right_Ey)
            )
            x_line_right_Ey = x_line_right_Ey**falloff_exponent
            x_line_right_Ey *= sigma_max
            for i in range(self.sigma_x_Hz.shape[0]):
                self.sigma_x_Hz[i, :][x_line_Hz > self.x_right] = x_line_right_Hz
            for i in range(self.sigma_x_Ex.shape[0]):
                self.sigma_x_Ex[i, :][x_line_Ex > self.x_right] = x_line_right_Ex
            for i in range(self.sigma_x_Ey.shape[0]):
                self.sigma_x_Ey[i, :][x_line_Ey > self.x_right] = x_line_right_Ey
        elif self.defaults["right_boundary"] == "PEC":
            pass
        else:
            print("defaults.right_boundary can be either PML or PEC. Assuming PEC")

        # y top
        if self.defaults["top_boundary"] == "PML":
            y_line_top_Hz = y_line_Hz[y_line_Hz > self.y_top]
            y_line_top_Hz = (y_line_top_Hz - np.min(y_line_top_Hz)) / (
                np.max(y_line_top_Hz) - np.min(y_line_top_Hz)
            )
            y_line_top_Hz = y_line_top_Hz**falloff_exponent
            y_line_top_Hz *= sigma_max
            y_line_top_Ex = y_line_Ex[y_line_Ex > self.y_top]
            y_line_top_Ex = (y_line_top_Ex - np.min(y_line_top_Ex)) / (
                np.max(y_line_top_Ex) - np.min(y_line_top_Ex)
            )
            y_line_top_Ex = y_line_top_Ex**falloff_exponent
            y_line_top_Ex *= sigma_max
            y_line_top_Ey = y_line_Ey[y_line_Ey > self.y_top]
            y_line_top_Ey = (y_line_top_Ey - np.min(y_line_top_Ey)) / (
                np.max(y_line_top_Ey) - np.min(y_line_top_Ey)
            )
            y_line_top_Ey = y_line_top_Ey**falloff_exponent
            y_line_top_Ey *= sigma_max
            for i in range(self.sigma_x_Hz.shape[1]):
                self.sigma_y_Hz[:, i][y_line_Hz > self.y_top] = y_line_top_Hz
            for i in range(self.sigma_x_Ex.shape[1]):
                self.sigma_y_Ex[:, i][y_line_Ex > self.y_top] = y_line_top_Ex
            for i in range(self.sigma_x_Ey.shape[1]):
                self.sigma_y_Ey[:, i][y_line_Ey > self.y_top] = y_line_top_Ey
        elif self.defaults["top_boundary"] == "PEC":
            pass
        else:
            print("defaults.top_boundary can be either PML or PEC. Assuming PEC")

        # y bot
        if self.defaults["bot_boundary"] == "PML":
            y_line_bot_Hz = y_line_Hz[y_line_Hz < self.y_bot]
            y_line_bot_Hz = (y_line_bot_Hz - np.min(y_line_bot_Hz)) / (
                np.max(y_line_bot_Hz) - np.min(y_line_bot_Hz)
            )
            y_line_bot_Hz = y_line_bot_Hz[::-1]
            y_line_bot_Hz = y_line_bot_Hz**falloff_exponent
            y_line_bot_Hz *= sigma_max
            y_line_bot_Ex = y_line_Ex[y_line_Ex < self.y_bot]
            y_line_bot_Ex = (y_line_bot_Ex - np.min(y_line_bot_Ex)) / (
                np.max(y_line_bot_Ex) - np.min(y_line_bot_Ex)
            )
            y_line_bot_Ex = y_line_bot_Ex[::-1]
            y_line_bot_Ex = y_line_bot_Ex**falloff_exponent
            y_line_bot_Ex *= sigma_max
            y_line_bot_Ey = y_line_Ey[y_line_Ey < self.y_bot]
            y_line_bot_Ey = (y_line_bot_Ey - np.min(y_line_bot_Ey)) / (
                np.max(y_line_bot_Ey) - np.min(y_line_bot_Ey)
            )
            y_line_bot_Ey = y_line_bot_Ey[::-1]
            y_line_bot_Ey = y_line_bot_Ey**falloff_exponent
            y_line_bot_Ey *= sigma_max
            for i in range(self.sigma_x_Hz.shape[1]):
                self.sigma_y_Hz[:, i][y_line_Hz < self.y_bot] = y_line_bot_Hz
            for i in range(self.sigma_x_Ex.shape[1]):
                self.sigma_y_Ex[:, i][y_line_Ex < self.y_bot] = y_line_bot_Ex
            for i in range(self.sigma_x_Ey.shape[1]):
                self.sigma_y_Ey[:, i][y_line_Ey < self.y_bot] = y_line_bot_Ey
        elif self.defaults["bot_boundary"] == "PEC":
            pass
        else:
            print("defaults.bot_boundary can be either PML or PEC. Assuming PEC")

        # PEC: tangential E (Ex and Ey) is forced to zero inside the cylinder
        self.pec_mask_Ex = np.zeros_like(self.x_Ex, dtype=bool)
        self.pec_mask_Ey = np.zeros_like(self.x_Ey, dtype=bool)
        for x0, y0, radius in self.defaults["conductivity_circles"]:
            xc = self.x_min + (self.x_max - self.x_min) * x0
            yc = self.y_min + (self.y_max - self.y_min) * y0
            rr = (self.x_max - self.x_min) * radius
            self.pec_mask_Ex |= (self.x_Ex - xc) ** 2 + (self.y_Ex - yc) ** 2 < rr**2
            self.pec_mask_Ey |= (self.x_Ey - xc) ** 2 + (self.y_Ey - yc) ** 2 < rr**2

        # derived materials
        self.alpha_x_Hz = self.epsilon_Hz / self.delta_t - self.sigma_x_Hz / 2
        self.alpha_y_Hz = self.epsilon_Hz / self.delta_t - self.sigma_y_Hz / 2
        self.beta_x_Hz = self.epsilon_Hz / self.delta_t + self.sigma_x_Hz / 2
        self.beta_y_Hz = self.epsilon_Hz / self.delta_t + self.sigma_y_Hz / 2
        self.alpha_x_Ex = self.epsilon_Ex / self.delta_t - self.sigma_x_Ex / 2
        self.alpha_y_Ex = self.epsilon_Ex / self.delta_t - self.sigma_y_Ex / 2
        self.beta_x_Ex = self.epsilon_Ex / self.delta_t + self.sigma_x_Ex / 2
        self.beta_y_Ex = self.epsilon_Ex / self.delta_t + self.sigma_y_Ex / 2
        self.alpha_x_Ey = self.epsilon_Ey / self.delta_t - self.sigma_x_Ey / 2
        self.alpha_y_Ey = self.epsilon_Ey / self.delta_t - self.sigma_y_Ey / 2
        self.beta_x_Ey = self.epsilon_Ey / self.delta_t + self.sigma_x_Ey / 2
        self.beta_y_Ey = self.epsilon_Ey / self.delta_t + self.sigma_y_Ey / 2

        # setup current stuff
        self.xidx_Mz = int(self.x_Hz.shape[0] / 2)
        self.yidx_Mz = int(self.x_Hz.shape[0] / 2)
        self.t = np.arange(0.0, self.picoseconds * 1e-12, self.delta_t)
        self.period = 1 / self.frequency
        self.Mz_xidx_yidx = mgpulse(self.t, self.period, self.frequency)

        # relocate temp fields
        self.Mz = np.zeros_like(self.x_Hz)
        self.Hz = np.zeros_like(self.x_Hz)
        self.Hz_sx = np.zeros_like(self.x_Hz)
        self.Hz_sy = np.zeros_like(self.x_Hz)
        self.Ex = np.zeros_like(self.x_Ex)
        self.Ey = np.zeros_like(self.x_Ey)

        # reallocated to_save
        self.Hz_to_save = np.zeros((len(self.t),) + self.Hz_sx.shape, dtype=np.float32)
        self.Ex_to_save = np.zeros((len(self.t),) + self.Ex.shape, dtype=np.float32)
        self.Ey_to_save = np.zeros((len(self.t),) + self.Ey.shape, dtype=np.float32)

    def get_solution_memory_size(self):
        floats_2d_slice = len(np.arange(self.x_min, self.x_max, self.delta_x)) * len(
            np.arange(self.y_min, self.y_max, self.delta_y)
        )
        # 3 for Hz Ey Ex and 4 bytes per float32
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
        # Insert magnetic current source (TEz: Mz drives Hz)
        # self.Mz[self.yidx_Mz, self.xidx_Mz] = self.Mz_xidx_yidx[time_index]

        box_size = self.defaults["PML_inset_as_uniform"] + 0.2
        x_left = self.x_min + (self.x_max - self.x_min) * box_size
        x_right = self.x_min + (self.x_max - self.x_min) * (1.0 - box_size)
        y_bot = self.y_min + (self.y_max - self.y_min) * box_size
        y_top = self.y_min + (self.y_max - self.y_min) * (1.0 - box_size)

        # mask creation
        x_idx_left = np.abs(self.x_Hz[0, :] - x_left).argmin()
        x_idx_right = np.abs(self.x_Hz[0, :] - x_right).argmin()
        y_idx_bot = np.abs(self.y_Hz[:, 0] - y_bot).argmin()
        y_idx_top = np.abs(self.y_Hz[:, 0] - y_top).argmin()

        rows = slice(y_idx_bot, y_idx_top + 1)
        cols = slice(x_idx_left, x_idx_right + 1)

        # Incident field is Ey / Hz (propagating +x), so Ex_inc = 0 and the
        # top/bottom surface corrections only exist for Ex (needs Hz_inc) and
        # Hz_sx (left/right, needs Ey_inc).

        # Hz update (H-type, normalized by eps/mu like the old Hx/Hy updates)
        # Hz_sx update
        Hz_sx_prev = self.Hz_sx.copy()
        self.Hz_sx[1:-1, 1:-1] = (1.0 / self.beta_x_Hz[1:-1, 1:-1]) * (
            self.alpha_x_Hz[1:-1, 1:-1] * self.Hz_sx[1:-1, 1:-1]
            - (self.epsilon_Hz[1:-1, 1:-1] / (self.mu_Hz[1:-1, 1:-1] * self.delta_x))
            * (self.Ey[1:-1, 1:] - self.Ey[1:-1, :-1])
            - (self.epsilon_Hz[1:-1, 1:-1] / self.mu_Hz[1:-1, 1:-1])
            * self.Mz[1:-1, 1:-1]
            / 2.0
        )
        # surface Hz_sx update left (total-field node, scattered Ey at left - 1)
        self.Hz_sx[rows, x_idx_left] = (1.0 / self.beta_x_Hz[rows, x_idx_left]) * (
            self.alpha_x_Hz[rows, x_idx_left] * Hz_sx_prev[rows, x_idx_left]
            - (
                self.epsilon_Hz[rows, x_idx_left]
                / (self.mu_Hz[rows, x_idx_left] * self.delta_x)
            )
            * (
                self.Ey[rows, x_idx_left]
                - (
                    self.Ey[rows, x_idx_left - 1]
                    + Ey_pw(
                        self.frequency,
                        self.x_Ey[rows, x_idx_left - 1],
                        self.t[time_index] - self.delta_t / 2,
                        1.0,
                    )
                )
            )
            - (self.epsilon_Hz[rows, x_idx_left] / self.mu_Hz[rows, x_idx_left])
            * self.Mz[rows, x_idx_left]
            / 2.0
        )
        # surface Hz_sx update right (total-field node, scattered Ey at right)
        self.Hz_sx[rows, x_idx_right] = (1.0 / self.beta_x_Hz[rows, x_idx_right]) * (
            self.alpha_x_Hz[rows, x_idx_right] * Hz_sx_prev[rows, x_idx_right]
            - (
                self.epsilon_Hz[rows, x_idx_right]
                / (self.mu_Hz[rows, x_idx_right] * self.delta_x)
            )
            * (
                (
                    self.Ey[rows, x_idx_right]
                    + Ey_pw(
                        self.frequency,
                        self.x_Ey[rows, x_idx_right],
                        self.t[time_index] - self.delta_t / 2,
                        1.0,
                    )
                )
                - self.Ey[rows, x_idx_right - 1]
            )
            - (self.epsilon_Hz[rows, x_idx_right] / self.mu_Hz[rows, x_idx_right])
            * self.Mz[rows, x_idx_right]
            / 2.0
        )

        # Hz_sy update (no surface correction needed since Ex_inc = 0)
        self.Hz_sy[1:-1, 1:-1] = (1.0 / self.beta_y_Hz[1:-1, 1:-1]) * (
            self.alpha_y_Hz[1:-1, 1:-1] * self.Hz_sy[1:-1, 1:-1]
            + (self.epsilon_Hz[1:-1, 1:-1] / (self.mu_Hz[1:-1, 1:-1] * self.delta_y))
            * (self.Ex[1:, 1:-1] - self.Ex[0:-1, 1:-1])
            - (self.epsilon_Hz[1:-1, 1:-1] / self.mu_Hz[1:-1, 1:-1])
            * self.Mz[1:-1, 1:-1]
            / 2.0
        )
        self.Hz = self.Hz_sx + self.Hz_sy

        # Ex update
        Ex_prev = self.Ex.copy()
        self.Ex[1:-1, :] = (1.0 / self.beta_y_Ex[1:-1, :]) * (
            self.alpha_y_Ex[1:-1, :] * self.Ex[1:-1, :]
            + (1.0 / self.delta_y) * (self.Hz[2:-1, :] - self.Hz[1:-2, :])
        )
        # surface Ex update bot (scattered Ex, total Hz at y_idx_bot)
        self.Ex[y_idx_bot - 1, cols] = (1.0 / self.beta_y_Ex[y_idx_bot - 1, cols]) * (
            self.alpha_y_Ex[y_idx_bot - 1, cols] * Ex_prev[y_idx_bot - 1, cols]
            + (1.0 / self.delta_y)
            * (
                (
                    self.Hz[y_idx_bot, cols]
                    - Hz_pw(
                        self.frequency,
                        self.x_Hz[y_idx_bot, cols],
                        self.t[time_index],
                        1.0,
                    )
                )
                - self.Hz[y_idx_bot - 1, cols]
            )
        )
        # surface Ex update top (scattered Ex, total Hz at y_idx_top)
        self.Ex[y_idx_top, cols] = (1.0 / self.beta_y_Ex[y_idx_top, cols]) * (
            self.alpha_y_Ex[y_idx_top, cols] * Ex_prev[y_idx_top, cols]
            + (1.0 / self.delta_y)
            * (
                self.Hz[y_idx_top + 1, cols]
                - (
                    self.Hz[y_idx_top, cols]
                    - Hz_pw(
                        self.frequency,
                        self.x_Hz[y_idx_top, cols],
                        self.t[time_index],
                        1.0,
                    )
                )
            )
        )

        # Ey update
        Ey_prev = self.Ey.copy()
        self.Ey[:, 1:-1] = (1.0 / self.beta_x_Ey[:, 1:-1]) * (
            self.alpha_x_Ey[:, 1:-1] * self.Ey[:, 1:-1]
            - (1.0 / self.delta_x) * (self.Hz[:, 2:-1] - self.Hz[:, 1:-2])
        )
        # surface Ey update left (scattered Ey, total Hz at x_idx_left)
        self.Ey[rows, x_idx_left - 1] = (1.0 / self.beta_x_Ey[rows, x_idx_left - 1]) * (
            self.alpha_x_Ey[rows, x_idx_left - 1] * Ey_prev[rows, x_idx_left - 1]
            - (1.0 / self.delta_x)
            * (
                (
                    self.Hz[rows, x_idx_left]
                    - Hz_pw(
                        self.frequency,
                        self.x_Hz[rows, x_idx_left],
                        self.t[time_index],
                        1.0,
                    )
                )
                - self.Hz[rows, x_idx_left - 1]
            )
        )
        # surface Ey update right (scattered Ey, total Hz at x_idx_right)
        self.Ey[rows, x_idx_right] = (1.0 / self.beta_x_Ey[rows, x_idx_right]) * (
            self.alpha_x_Ey[rows, x_idx_right] * Ey_prev[rows, x_idx_right]
            - (1.0 / self.delta_x)
            * (
                self.Hz[rows, x_idx_right + 1]
                - (
                    self.Hz[rows, x_idx_right]
                    - Hz_pw(
                        self.frequency,
                        self.x_Hz[rows, x_idx_right],
                        self.t[time_index],
                        1.0,
                    )
                )
            )
        )

        # PEC cylinder: tangential E = 0
        self.Ex[self.pec_mask_Ex] = 0.0
        self.Ey[self.pec_mask_Ey] = 0.0

        # save new fields to time index
        self.Hz_to_save[time_index] = self.Hz
        self.Ex_to_save[time_index] = self.Ex
        self.Ey_to_save[time_index] = self.Ey

    def convert_to_save_data_to_time_series(self):
        self.time_series_array = []
        self.time_series_array.append(
            TimeSeries(
                self.Hz_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        # del self.Hz_to_save
        self.time_series_array.append(
            TimeSeries(
                self.Ex_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        # del self.Ex_to_save
        self.time_series_array.append(
            TimeSeries(
                self.Ey_to_save,
                int(
                    self.period
                    * self.defaults["colorbar_max_mean_in_periods"]
                    / self.delta_t
                ),
            )
        )
        # del self.Ey_to_save
