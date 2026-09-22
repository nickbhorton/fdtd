import threading

import dearpygui.dearpygui as dpg
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0, mu_0


def format_bytes(size: int, decimal: bool = False, precision: int = 2) -> str:
    """
    from gemini
    Convert a byte count into a human-readable string.

    :param size: Number of bytes.
    :param decimal: If True, uses decimal units (1 KB = 1000 B).
                    If False, uses binary units (1 KiB = 1024 B).
    :param precision: Number of decimal places to round to.
    :return: Formatted string representing human-readable size.
    """
    if size < 0:
        raise ValueError("Byte count cannot be negative.")

    base = 1000 if decimal else 1024
    units = (
        ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        if decimal
        else ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    )

    if size < base:
        return f"{size} B"

    unit_index = 0
    value = float(size)

    while value >= base and unit_index < len(units) - 1:
        value /= base
        unit_index += 1

    return f"{value:.{precision}f} {units[unit_index]}"


def compute_wavelength(frequency, phase_velocity):
    return phase_velocity / frequency


def materials_to_phase_velocity(epsilon, mu):
    return 1 / np.sqrt(epsilon * mu)


def stability_condition_2d(phase_velocity, delta_x, delta_y):
    return 1 / (phase_velocity * np.sqrt(1 / delta_x**2 + 1 / delta_y**2))


class TimeSeries:
    def __init__(self, data: np.ndarray):
        self.data = data.copy()

        # padding with white for Hx and Hy grids
        if self.data.shape[2] < self.data.shape[1]:
            self.data = np.pad(
                self.data,
                ((0, 0), (0, 0), (0, self.data.shape[1] - self.data.shape[2])),
                mode="constant",
                constant_values=0.0,
            )
        if self.data.shape[1] < self.data.shape[2]:
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.data.shape[2] - self.data.shape[1]), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        self.depth, self.width, self.height = self.data.shape
        self.max_field = np.max(
            np.max(
                (
                    np.abs(np.max(self.data, axis=(1, 2))),
                    np.abs(np.min(self.data, axis=(1, 2))),
                ),
                axis=0,
            )
        )

    def get_rgba_in_bwr(self, time_index, scale=1.0):
        if time_index == None:
            time_index = 0

        if self.max_field != 0.0:
            dat = self.data[time_index] + self.max_field * scale
            dat = dat / (2.0 * self.max_field * scale)
        else:
            dat = self.data[time_index]
        # matplotlib bwr logic assuming 0.0 < self.data[time_index] < 1.0
        rgba = np.empty((self.width, self.height, 4), dtype=np.float32)
        rgba[..., 0] = scale * np.piecewise(
            dat,
            [dat < 0.5, dat >= 0.5],
            [lambda val: 2.0 * val, 1.0],
        )
        rgba[..., 1] = scale * np.piecewise(
            dat,
            [dat < 0.5, dat >= 0.5],
            [lambda val: 2.0 * val, lambda val: 2.0 * (1.0 - val)],
        )
        rgba[..., 2] = scale * np.piecewise(
            dat,
            [dat < 0.5, dat >= 0.5],
            [1.0, lambda val: 2.0 * (1.0 - val)],
        )
        rgba[..., 3] = 1.0  # A
        return rgba.ravel()


class Solver:
    def __init__(self):
        self.frequency = 10e9

        self.epsilon_background = 1.0 * epsilon_0
        self.mu_background = 1.0 * mu_0
        self.sigma_background = 0.0

        self.phase_velocity = materials_to_phase_velocity(
            self.epsilon_background, self.mu_background
        )
        self.wavelength = compute_wavelength(self.frequency, self.phase_velocity)

        # gui option initialization
        self.picoseconds = 1000
        self.delta_x = self.wavelength / 50
        self.delta_y = self.wavelength / 50
        self.wavelengths_x = 4
        self.wavelengths_y = 4

        self.setup_solver()
        self.convert_to_save_data_to_time_series()

    def setup_solver(self):
        # print("Numerical dispersion", np.pi**2 / 8 * (delta_x / wavelength) ** 2)

        # first delete any TimeSeries that exist
        self.time_series_array = []

        self.x_min = 0
        self.x_max = self.wavelengths_x * self.wavelength
        self.y_min = 0
        self.y_max = self.wavelengths_y * self.wavelength

        x_Ez = np.arange(self.x_min, self.x_max, self.delta_x)
        y_Ez = np.arange(self.y_min, self.y_max, self.delta_y)

        x_Hx = (x_Ez + self.delta_x / 2)[:-1]
        y_Hx = y_Ez

        x_Hy = x_Ez
        y_Hy = (y_Ez + self.delta_y / 2)[:-1]

        x_Ez, y_Ez = np.meshgrid(x_Ez, y_Ez)
        x_Hx, y_Hx = np.meshgrid(x_Hx, y_Hx)
        x_Hy, y_Hy = np.meshgrid(x_Hy, y_Hy)

        self.xidx_Jz = int(x_Ez.shape[0] / 2)
        self.yidx_Jz = int(x_Ez.shape[0] / 2)

        self.delta_t = stability_condition_2d(
            self.phase_velocity, self.delta_x, self.delta_y
        )

        # setup materials
        self.epsilon = np.ones_like(x_Ez) * self.epsilon_background
        self.mu = np.ones_like(x_Ez) * self.mu_background
        self.sigma_x = np.zeros_like(self.epsilon)
        self.sigma_y = np.zeros_like(self.epsilon)

        # PML stuff
        x_left = (self.x_max - self.x_min) * 0.2
        x_right = (self.x_max - self.x_min) * 0.8
        y_top = (self.y_max - self.y_min) * 0.8
        y_bot = (self.y_max - self.y_min) * 0.2

        sigma_val = 1e3

        self.sigma_x[x_Ez < x_left] = sigma_val
        self.sigma_x[x_Ez > x_right] = sigma_val
        self.sigma_y[y_Ez < y_bot] = sigma_val
        self.sigma_y[y_Ez > y_top] = sigma_val

        # derived materials
        self.alpha_x = np.ones_like(x_Ez) * (
            (self.epsilon / self.delta_t) - self.sigma_x / 2
        )
        self.alpha_y = np.ones_like(x_Ez) * (
            (self.epsilon / self.delta_t) - self.sigma_y / 2
        )
        self.beta_x = np.ones_like(x_Ez) * (
            (self.epsilon / self.delta_t) + self.sigma_x / 2
        )
        self.beta_y = np.ones_like(x_Ez) * (
            (self.epsilon / self.delta_t) + self.sigma_y / 2
        )

        # fig, ax = plt.subplots(1, 2)
        # mesh1 = ax[0].pcolormesh(x_Ez, y_Ez, self.alpha_x)
        # mesh2 = ax[1].pcolormesh(x_Ez, y_Ez, self.alpha_y)
        # fig.colorbar(mesh1)
        # fig.colorbar(mesh2)
        # plt.show()

        self.t = np.arange(0.0, self.picoseconds * 1e-12, self.delta_t)
        t_sig = 1 / self.frequency
        self.Jz_xidx_yidx = np.exp(-0.5 * (self.t / t_sig) ** 2) * np.sin(
            2 * np.pi * self.frequency * self.t
        )

        # relocate temp fields
        self.Jz = np.zeros_like(x_Ez)
        self.Ez = np.zeros_like(x_Ez)
        self.Ez_x = np.zeros_like(x_Ez)
        self.Ez_y = np.zeros_like(x_Ez)
        self.Hx = np.zeros_like(x_Hx)
        self.Hy = np.zeros_like(x_Hy)

        # reallocated to_save
        self.Ez_to_save = np.zeros((len(self.t),) + self.Ez_x.shape, dtype=np.float32)
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

    # this is
    def field_time_step(self, time_index):
        # grab next current timestep
        self.Jz[self.yidx_Jz, self.xidx_Jz] = self.Jz_xidx_yidx[time_index]

        # update magnetic field
        self.Hx = (
            1.0
            / self.beta_y[:, 1:]
            * (
                self.alpha_y[:, 1:] * self.Hx
                - (self.epsilon[:, 1:] / (self.mu[:, 1:] * self.delta_y))
                * (self.Ez[:, 1:] - self.Ez[:, :-1])
            )
        )
        self.Hy = (
            1.0
            / self.beta_x[1:, :]
            * (
                self.alpha_x[1:, :] * self.Hy
                + (self.epsilon[1:, :] / (self.mu[1:, :] * self.delta_x))
                * (self.Ez[1:, :] - self.Ez[:-1, :])
            )
        )

        # update electric field
        self.Ez_x[1:-1, 1:-1] = (
            1.0
            / self.beta_x[1:-1, 1:-1]
            * (
                self.alpha_x[1:-1, 1:-1] * self.Ez_x[1:-1, 1:-1]
                + 1 / self.delta_x * (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1])
                - self.Jz[1:-1, 1:-1]
            )
        )
        self.Ez_y[1:-1, 1:-1] = (
            1.0
            / self.beta_y[1:-1, 1:-1]
            * (
                self.alpha_y[1:-1, 1:-1] * self.Ez_y[1:-1, 1:-1]
                - 1 / self.delta_y * (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])
                - self.Jz[1:-1, 1:-1]
            )
        )
        self.Ez = self.Ez_x + self.Ez_y

        # self.Hx = self.Hx - self.delta_t / (self.mu[:, 1:] * self.delta_y) * (
        #     self.Ez[:, 1:] - self.Ez[:, :-1]
        # )
        # self.Hy = self.Hy + self.delta_t / (self.mu[1:, :] * self.delta_x) * (
        #     self.Ez[1:, :] - self.Ez[:-1, :]
        # )

        # save new fields to time index
        self.Ez_to_save[time_index] = self.Ez
        self.Hx_to_save[time_index] = self.Hx
        self.Hy_to_save[time_index] = self.Hy

    # THIS WILL DELETE self.[Ez,Hx,Hy]_to_save
    def convert_to_save_data_to_time_series(self):
        self.time_series_array = []
        self.time_series_array.append(TimeSeries(self.Ez_to_save))
        del self.Ez_to_save
        self.time_series_array.append(TimeSeries(self.Hx_to_save))
        del self.Hx_to_save
        self.time_series_array.append(TimeSeries(self.Hy_to_save))
        del self.Hy_to_save


class App:
    def __init__(self):
        self.solver = Solver()
        self.time_series_index = 0

        dpg.create_context()

        with dpg.window(tag="primary_window"):  # type: ignore
            with dpg.window(
                tag="setup",
                no_move=True,
                no_resize=True,
                no_close=True,
                no_collapse=True,
                no_title_bar=True,
            ):  # type: ignore
                dpg.add_input_float(
                    label="Picoseconds",
                    default_value=self.solver.picoseconds,
                    callback=self.update_period_count,
                )
                dpg.add_input_int(
                    label="Wavelength Discretization in x",
                    tag="delta_x",
                    default_value=50,
                    callback=self.update_delta_x,
                )
                dpg.add_input_int(
                    label="Wavelength Discretization in y",
                    tag="delta_y",
                    default_value=50,
                    callback=self.update_delta_y,
                )
                dpg.add_input_int(
                    label="Wavelengths in x",
                    tag="lambda_x",
                    default_value=self.solver.wavelengths_x,
                    callback=self.update_lambda_x,
                )
                dpg.add_input_int(
                    label="Wavelengths in y",
                    tag="lambda_y",
                    default_value=self.solver.wavelengths_y,
                    callback=self.update_lambda_y,
                )
                dpg.add_text(
                    f"Memory Size: {self.solver.get_solution_memory_size()} bytes",
                    tag="memory_size",
                )
                dpg.add_button(
                    tag="solve_button", label="Solve", callback=self.run_solver
                )
                dpg.add_progress_bar(
                    tag="solve_progress_bar", default_value=0.0, overlay="0%", width=-1
                )
            with dpg.window(
                tag="show",
                no_move=True,
                no_resize=True,
                no_close=True,
                no_collapse=True,
                no_title_bar=True,
            ):  # type: ignore
                dpg.add_slider_int(
                    tag="time_series_slider",
                    min_value=0,
                    max_value=self.solver.time_series_array[
                        self.time_series_index
                    ].depth
                    - 1,
                    default_value=0,
                    callback=self.update_image,
                    width=930,
                )
                dpg.add_slider_float(
                    tag="colorbar_scale_slider",
                    min_value=0.0,
                    max_value=2.0,
                    default_value=1.0,
                    callback=self.update_image,
                    width=930,
                )
                dpg.add_radio_button(
                    ["Ez", "Hx", "Hy"], callback=self.update_field_type, horizontal=True
                )

        self.create_texture_first_time()
        self.update_memory_size(None, None)

        dpg.create_viewport(title="Dynamic Texture Update", width=1920, height=1080)
        dpg.set_primary_window("primary_window", True)
        dpg.setup_dearpygui()

        dpg.show_viewport()

        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()
        half_width = width // 2
        dpg.configure_item("setup", pos=[0, 0], width=half_width, height=height)
        dpg.configure_item("show", pos=[half_width, 0], width=half_width, height=height)

        dpg.start_dearpygui()

    def create_texture_first_time(self):
        with dpg.texture_registry(show=False, tag="my_texture_registry"):  # type: ignore
            dpg.add_dynamic_texture(
                width=self.solver.time_series_array[self.time_series_index].width,
                height=self.solver.time_series_array[self.time_series_index].height,
                default_value=self.solver.time_series_array[
                    self.time_series_index
                ].get_rgba_in_bwr(dpg.get_value("time_series_slider")),  # type: ignore
                tag="slice_texture",
            )
        dpg.add_image(
            "slice_texture", tag="image", width=930, height=930, parent="show"
        )

    def recreate_texture(self):
        # reset time_series_slider
        dpg.configure_item(
            "time_series_slider",
            max_value=len(self.solver.t) - 1,
        )
        dpg.set_value("time_series_slider", 0)

        # then delete stuff
        if dpg.does_item_exist("slice_texture"):
            dpg.delete_item("slice_texture")
        if dpg.does_alias_exist("slice_texture"):
            dpg.remove_alias("slice_texture")
        dpg.add_dynamic_texture(
            width=self.solver.time_series_array[self.time_series_index].width,
            height=self.solver.time_series_array[self.time_series_index].height,
            default_value=self.solver.time_series_array[
                self.time_series_index
            ].get_rgba_in_bwr(dpg.get_value("time_series_slider")),  # type: ignore
            tag="slice_texture",
            parent="my_texture_registry",
        )
        if dpg.does_item_exist("image"):
            dpg.delete_item("image")
        dpg.add_image(
            "slice_texture", tag="image", width=930, height=930, parent="show"
        )

    def update_image(self, sender, time_index):
        time_index = dpg.get_value("time_series_slider")
        scale = dpg.get_value("colorbar_scale_slider")
        dpg.set_value(
            "slice_texture",
            self.solver.time_series_array[self.time_series_index].get_rgba_in_bwr(
                time_index, scale
            ),
        )

    def update_field_type(self, sender, app_data):
        if app_data == "Ez":
            self.time_series_index = 0
        elif app_data == "Hy":
            self.time_series_index = 1
        else:
            self.time_series_index = 2

        self.update_image(None, dpg.get_value("time_series_slider"))

    def update_delta_x(self, sender, delta_x):
        self.solver.set_delta_x(self.solver.wavelength / delta_x)
        self.update_memory_size(None, self.solver.get_solution_memory_size())

    def update_delta_y(self, sender, delta_y):
        self.solver.set_delta_y(self.solver.wavelength / delta_y)
        self.update_memory_size(None, self.solver.get_solution_memory_size())

    def update_period_count(self, sender, period_count):
        self.solver.set_picoseconds(period_count)
        self.update_memory_size(None, self.solver.get_solution_memory_size())

    def update_lambda_x(self, sender, lambda_x):
        self.solver.set_wavelengths_in_x(lambda_x)
        self.update_memory_size(None, self.solver.get_solution_memory_size())

    def update_lambda_y(self, sender, lambda_y):
        self.solver.set_wavelengths_in_y(lambda_y)
        self.update_memory_size(None, self.solver.get_solution_memory_size())

    def update_memory_size(self, sender, memory_size):
        memory_bytes_int = self.solver.get_solution_memory_size()
        memory_bytes = format_bytes(memory_bytes_int)
        max_memory_bytes = format_bytes(memory_bytes_int + memory_bytes_int // 3)
        dpg.set_value(
            "memory_size",
            f"Memory Size: {memory_bytes} bytes (peak {max_memory_bytes})",
        )

    def run_solver(self):
        dpg.configure_item("solve_button", enabled=False)
        dpg.set_value("solve_progress_bar", 0.0)
        dpg.configure_item("solve_progress_bar", overlay="0%")

        # update solver based on GUI
        self.solver.set_delta_y(self.solver.wavelength / dpg.get_value("delta_y"))
        self.solver.setup_solver()

        threading.Thread(target=self._run_solver, daemon=True).start()

    def _run_solver(self):
        print("_run_solver")
        for i in range(len(self.solver.t)):
            self.solver.field_time_step(i)
            progress_value = (i + 1) / len(self.solver.t)
            dpg.set_value("solve_progress_bar", progress_value)
            dpg.configure_item(
                "solve_progress_bar", overlay=f"{int(progress_value * 100)}%"
            )
        self.solver.convert_to_save_data_to_time_series()

        # recreate texture at possibly new size
        self.recreate_texture()
        self.update_image(None, dpg.get_value("time_series_slider"))
        dpg.configure_item("solve_button", enabled=True)


a = App()
dpg.destroy_context()
