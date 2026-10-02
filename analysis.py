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

    #
    #  +--+
    # x|  |
    #  +--+
    #
    y1s = np.arange(y_pairs[0][0], y_pairs[0][1])
    x1s = np.ones_like(y1s) * x_pairs[0][0]
    nx1s = -np.ones_like(x1s)
    ny1s = np.zeros_like(y1s)

    #    x
    #  +--+
    #  |  |
    #  +--+
    #
    x2s = np.arange(x_pairs[1][0], x_pairs[1][1])
    y2s = np.ones_like(x2s) * y_pairs[1][0]
    nx2s = np.zeros_like(x2s)
    ny2s = np.ones_like(y2s)

    #
    #  +--+
    #  |  |x
    #  +--+
    #
    y3s = np.arange(y_pairs[2][0], y_pairs[2][1], -1)
    x3s = np.ones_like(y3s) * x_pairs[2][0]
    nx3s = np.ones_like(x3s)
    ny3s = np.zeros_like(y3s)

    #
    #  +--+
    #  |  |
    #  +--+
    #    x
    x4s = np.arange(x_pairs[3][0], x_pairs[3][1], -1)
    y4s = np.ones_like(x4s) * y_pairs[3][0]
    nx4s = np.zeros_like(x4s)
    ny4s = -np.ones_like(y4s)

    xs = np.concat([x1s, x2s, x3s, x4s])
    ys = np.concat([y1s, y2s, y3s, y4s])
    nxs = np.concat([nx1s, nx2s, nx3s, nx4s])
    nys = np.concat([ny1s, ny2s, ny3s, ny4s])
    ns = np.zeros((3, len(nxs)))
    ns[0] = nxs
    ns[1] = nys

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
    return x_square_1d, y_square_1d, field_square, angle, ns


data = np.load("data/pec_15_15_4k.npz")
t = data["t"]
x_Ez = data["x_Ez"]
y_Ez = data["y_Ez"]
# x_Hx = data["x_Hx"]
# y_Hx = data["y_Hx"]
# x_Hy = data["x_Hy"]
# y_Hy = data["y_Hy"]
Ez = data["Ez"]
Hx = data["Hx"]
Hy = data["Hy"]

x_square_1d_Ez, y_square_1d_Ez, Ez_square, angle_Ez, ns_Ez = get_square(x_Ez, y_Ez, Ez)
x_square_1d_Hx, y_square_1d_Hx, Hx_square, angle_Hx, ns_Hx = get_square(x_Ez, y_Ez, Hx)
x_square_1d_Hy, y_square_1d_Hy, Hy_square, angle_Hy, ns_Hy = get_square(x_Ez, y_Ez, Hy)

# fft amplitudes at 10GHz
dt = t[1] - t[0]
target_frequency = 10e9

Ez_fft_output = np.fft.fft(Ez_square, axis=1)
Ez_fs = np.fft.fftfreq(len(Ez_square[0, :]), d=dt)
Ez_amps = np.abs(Ez_fft_output)[:, : Ez_fft_output.shape[1] // 2]
Ez_amps[:, 1:] *= 2.0
Ez_fs = Ez_fs[: len(Ez_fs) // 2]
Ez_idx = np.argmin(np.abs(Ez_fs - target_frequency))
Ez_amps_at_target_f = Ez_amps[:, Ez_idx]

Hx_fft_output = np.fft.fft(Hx_square, axis=1)
Hx_fs = np.fft.fftfreq(len(Hx_square[0, :]), d=dt)
Hx_amps = np.abs(Hx_fft_output)[:, : Hx_fft_output.shape[1] // 2]
Hx_amps[:, 1:] *= 2.0
Hx_fs = Hx_fs[: len(Hx_fs) // 2]
Hx_idx = np.argmin(np.abs(Hx_fs - target_frequency))
Hx_amps_at_target_f = Hx_amps[:, Hx_idx]

Hy_fft_output = np.fft.fft(Hy_square, axis=1)
Hy_fs = np.fft.fftfreq(len(Hy_square[0, :]), d=dt)
Hy_amps = np.abs(Hy_fft_output)[:, : Hy_fft_output.shape[1] // 2]
Hy_amps[:, 1:] *= 2.0
Hy_fs = Hy_fs[: len(Hy_fs) // 2]
Hy_idx = np.argmin(np.abs(Hy_fs - target_frequency))
Hy_amps_at_target_f = Hy_amps[:, Hy_idx]

E_square = np.zeros((3, len(Ez_square)))
H_square = np.zeros((3, len(Hx_square)))

E_square[2] = Ez_amps_at_target_f
H_square[0] = Hx_amps_at_target_f
H_square[1] = Hy_amps_at_target_f

M_eff = -np.cross(ns_Ez, E_square, axisa=0, axisb=0)
J_eff = np.cross(ns_Hx, H_square, axisa=0, axisb=0)

fig, ax = plt.subplots(2, 3)
ax[0][0].scatter(angle_Ez, M_eff[:, 0], 2, color="black")
ax[0][1].scatter(angle_Ez, M_eff[:, 1], 2, color="black")
ax[0][2].scatter(angle_Ez, M_eff[:, 2], 2, color="black")
ax[1][0].scatter(angle_Ez, J_eff[:, 0], 2, color="black")
ax[1][1].scatter(angle_Ez, J_eff[:, 1], 2, color="black")
ax[1][2].scatter(angle_Ez, J_eff[:, 2], 2, color="black")
