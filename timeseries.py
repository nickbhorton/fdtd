import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


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
        self.max_field = np.average(
            np.max(
                (
                    np.abs(np.max(self.data, axis=(1, 2))),
                    np.abs(np.min(self.data, axis=(1, 2))),
                ),
                axis=0,
            )
        )

    def get_rgba_in_bwr(self, time_index, scale=1.0):
        # partially from Gemini
        if time_index is None:
            time_index = 0

        # Determine maximum field magnitude for scaling
        max_val = self.max_field * scale

        if max_val != 0.0:
            # Normalize range [-max_val, max_val] to [0.0, 1.0]
            # vmin and vmax ensure 0 maps exactly to 0.5 (white)
            norm = mcolors.Normalize(vmin=-max_val, vmax=max_val, clip=True)
            dat_norm = norm(self.data[time_index])
        else:
            dat_norm = np.full_like(self.data[time_index], 0.5)

        return plt.get_cmap("bwr")(dat_norm).astype(np.float32).ravel()
