import numpy as np


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
