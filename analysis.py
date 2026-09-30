from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np


def get_square(x, y, field):
    ri = 30
    ri2 = x.shape[0] - ri
    ys = [ri, ri2, ri2, ri, ri]
    xs = [ri, ri, ri2, ri2, ri]
    y_pairs = list(pairwise(ys))
    x_pairs = list(pairwise(xs))

    y1s = np.arange(y_pairs[0][0], y_pairs[0][1])
    x1s = np.ones_like(y1s) * x_pairs[0][0]

    x2s = np.arange(x_pairs[1][0], x_pairs[1][1])
    y2s = np.ones_like(x2s) * y_pairs[1][0]

    y3s = np.arange(y_pairs[2][0], y_pairs[2][1], -1)
    x3s = np.ones_like(y3s) * x_pairs[2][0]

    x4s = np.arange(x_pairs[3][0], x_pairs[3][1], -1)
    y4s = np.ones_like(x4s) * y_pairs[3][0]

    xs = np.concat([x1s, x2s, x3s, x4s])
    ys = np.concat([y1s, y2s, y3s, y4s])

    x_square_1d = x[ys, xs]
    y_square_1d = y[ys, xs]

    x_circle_norm = (x_square_1d - x_square_1d.min()) / (
        x_square_1d.max() - x_square_1d.min()
    )
    x_circle_norm -= 0.5
    y_circle_norm = (y_square_1d - y_square_1d.min()) / (
        y_square_1d.max() - y_square_1d.min()
    )
    y_circle_norm -= 0.5

    angle = np.atan2(y_circle_norm, x_circle_norm)
    # angle_deg = np.rad2deg(angle % (2.0 * np.pi))

    field_square = []
    for xi, yi in zip(xs, ys):
        field_square.append(field[:, yi, xi])
    field_square = np.array(field_square)  # angle, time
    return x_square_1d, y_square_1d, field_square, angle


data = np.load("data/pec_15_15_4k.npz")
x_Ez = data["x_Ez"]
y_Ez = data["y_Ez"]
t = data["t"]
Ez = data["Ez"]

x_square_1d, y_square_1d, Ez_square, angle = get_square(x_Ez, y_Ez, Ez)

# plots Ez at ti and then the square
ti = 400
fig, ax = plt.subplots(1, 2)
mesh = ax[0].pcolormesh(x_Ez, y_Ez, Ez[ti, :, :], cmap="bwr")
ax[0].scatter(x_square_1d, y_square_1d, 2, c=np.arange(len(x_square_1d)), cmap="jet")
ax[1].scatter(angle, Ez_square[:, ti], 2, c=np.arange(len(angle)), cmap="jet")
fig.colorbar(mesh)

# fft amplitudes at 10GHz
dt = t[1] - t[0]
fft_output = np.fft.fft(Ez_square, axis=1)
fs = np.fft.fftfreq(len(Ez_square[0, :]), d=dt)
amps = np.abs(fft_output)[:, : fft_output.shape[1] // 2]
amps[:, 1:] *= 2.0
fs = fs[: len(fs) // 2]

target_frequency = 10e9
idx = np.argmin(np.abs(fs - target_frequency))
amps_at_target_f = amps[:, idx]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(angle, amps_at_target_f**2, color="tab:blue", linewidth=2, label="Spiral")

plt.show()
