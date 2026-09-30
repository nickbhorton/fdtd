from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np

data = np.load("data/pec_40_40_4k.npz")
x = data["x_Ez"]
y = data["y_Ez"]
t = data["t"]
Ez = data["Ez"]

ri = 100
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

x_circle1 = x[ys, xs]
y_circle1 = y[ys, xs]

x_circle = (x_circle1 - x_circle1.min()) / (x_circle1.max() - x_circle1.min())
x_circle -= 0.5
y_circle = (y_circle1 - y_circle1.min()) / (y_circle1.max() - y_circle1.min())
y_circle -= 0.5

angle = np.atan2(y_circle, x_circle)
angle = np.rad2deg(angle % (2.0 * np.pi))

Ez_circle = []
for xi, yi in zip(xs, ys):
    Ez_circle.append(Ez[:, yi, xi])
Ez_circle = np.array(Ez_circle)  # angle, time

ti = 700
fig, ax = plt.subplots(1, 2)
mesh = ax[0].pcolormesh(x, y, Ez[ti, :, :], cmap="bwr")
ax[0].scatter(x_circle1, y_circle1, 2, c=np.arange(len(x_circle1)), cmap="jet")
fig.colorbar(mesh)
ax[1].scatter(angle, Ez_circle[:, ti], 2, c=np.arange(len(angle)), cmap="jet")

dt = t[1] - t[0]
fft_output = np.fft.fft(Ez_circle, axis=1)
fs = np.fft.fftfreq(len(Ez_circle[0, :]), d=dt)
amps = np.abs(fft_output)[:, : fft_output.shape[1] // 2]
amps[:, 1:] *= 2.0
fs = fs[: len(fs) // 2]

idx = np.argmin(np.abs(fs - 10.0e9))
amps_10GHz = amps[:, idx]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(np.deg2rad(angle), amps_10GHz, color="tab:blue", linewidth=2, label="Spiral")

fig, ax = plt.subplots()
mesh = ax.scatter(x_circle1, y_circle1, c=amps_10GHz, cmap="jet")
plt.show()
