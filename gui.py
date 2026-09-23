import threading
from pathlib import Path

import dearpygui.dearpygui as dpg

from lib import format_bytes
from solver import Solver


class App:
    def __init__(self, solver_defaults_path: Path):
        self.solver = Solver(solver_defaults_path)
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
                    default_value=self.solver.defaults["wavelength_discretization_x"],
                    callback=self.update_delta_x,
                )
                dpg.add_input_int(
                    label="Wavelength Discretization in y",
                    tag="delta_y",
                    default_value=self.solver.defaults["wavelength_discretization_y"],
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
                    max_value=4.0,
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


a = App(Path("solver_default.json"))
dpg.destroy_context()
