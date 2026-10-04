import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0


plt.rcParams.update(
    {
        "axes.grid": False,
        "axes.edgecolor": "black",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 12,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
    }
)


def cartesian_vector_to_spherical(vector_cartesian, theta, phi):
    # From Gemini because I am lazy and didn't want to get this wrong
    vector_cartesian = np.asarray(vector_cartesian)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    cartesian_to_spherical_matrix = np.array(
        [
            [sin_t * cos_p, sin_t * sin_p, cos_t],
            [cos_t * cos_p, cos_t * sin_p, -sin_t],
            [-sin_p, cos_p, np.zeros_like(phi)],
        ]
    )
    return np.einsum(
        "ij...,j...->i...", cartesian_to_spherical_matrix, vector_cartesian
    )


def get_square(x, y, field, yi1, yi2, xi1, xi2):
    ys = [yi1, yi2, yi2, yi1, yi1]
    xs = [xi1, xi1, xi2, xi2, xi1]
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

    field_square = []
    for xi, yi in zip(xs, ys):
        field_square.append(field[:, yi, xi])
    field_square = np.array(field_square)  # angle, time
    return x_square_1d, y_square_1d, field_square, angle, ns


def fourier_transform_amplitude(field, target_frequency, t):
    return (field * np.exp(-2.0j * np.pi * target_frequency * t)[np.newaxis, :]).sum(
        axis=1
    ) * (t[1] - t[0])


data = np.load("data/bist_tmz_pec_fields_40_40_2p6k.npz")
t_E = data["t"]
dt = t_E[1] - t_E[0]
t_H = data["t"] - dt / 2
x_Ez = data["x_Ez"]
y_Ez = data["y_Ez"]
x_Hx = data["x_Hx"]
y_Hx = data["y_Hx"]
x_Hy = data["x_Hy"]
y_Hy = data["y_Hy"]
Ez = data["Ez"]
Hx = data["Hx"]
Hy = data["Hy"]


defaults_path = Path("solver_default.json")
defaults = json.loads(defaults_path.read_text())
# This is OK because my grid is always exactly square
x_line_Ez = x_Ez[0, :]
integration_surface_uniform = defaults["PML_inset_as_uniform"] + 0.05
x_left = (x_Ez.max() - x_Ez.min()) * integration_surface_uniform
x_right = (x_Ez.max() - x_Ez.min()) * (1 - integration_surface_uniform)
x_left_i = np.abs(x_line_Ez - x_left).argmin()
x_right_i = np.abs(x_line_Ez - x_right).argmin()

x_square_1d, y_square_1d, Ez_square, angle_Ez, ns_Ez = get_square(
    x_Ez, y_Ez, Ez, x_left_i, x_right_i, x_left_i, x_right_i
)

(
    x_square_1d_Hx_above,
    y_square_1d_Hx_above,
    Hx_square_above,
    angle_Hx_above,
    ns_Hx_above,
) = get_square(x_Hx, y_Hx, Hx, x_left_i, x_right_i, x_left_i, x_right_i)
(
    x_square_1d_Hx_below,
    y_square_1d_Hx_below,
    Hx_square_below,
    angle_Hx_below,
    ns_Hx_below,
) = get_square(x_Hx, y_Hx, Hx, x_left_i - 1, x_right_i - 1, x_left_i, x_right_i)

x_square_1d_Hx = (x_square_1d_Hx_above + x_square_1d_Hx_below) / 2
y_square_1d_Hx = (y_square_1d_Hx_above + y_square_1d_Hx_below) / 2
Hx_square = (Hx_square_above + Hx_square_below) / 2
angle_Hx = (angle_Hx_above + angle_Hx_below) / 2
ns_Hx = (ns_Hx_above + ns_Hx_below) / 2

(
    x_square_1d_Hy_above,
    y_square_1d_Hy_above,
    Hy_square_above,
    angle_Hy_above,
    ns_Hy_above,
) = get_square(x_Hy, y_Hy, Hy, x_left_i, x_right_i, x_left_i, x_right_i)
(
    x_square_1d_Hy_below,
    y_square_1d_Hy_below,
    Hy_square_below,
    angle_Hy_below,
    ns_Hy_below,
) = get_square(x_Hy, y_Hy, Hy, x_left_i, x_right_i, x_left_i - 1, x_right_i - 1)

x_square_1d_Hy = (x_square_1d_Hy_above + x_square_1d_Hy_below) / 2
y_square_1d_Hy = (y_square_1d_Hy_above + y_square_1d_Hy_below) / 2
Hy_square = (Hy_square_above + Hy_square_below) / 2
angle_Hy = (angle_Hy_above + angle_Hy_below) / 2
ns_Hy = (ns_Hy_above + ns_Hy_below) / 2

# fft amplitudes at target_frequency
dx = x_Ez[0, 1] - x_Ez[0, 0]
dy = y_Ez[1, 0] - y_Ez[0, 0]
target_frequency = 10e9

phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / target_frequency

Ez_at_f = fourier_transform_amplitude(Ez_square, target_frequency, t_E)
Hx_at_f = fourier_transform_amplitude(Hx_square, target_frequency, t_H)
Hy_at_f = fourier_transform_amplitude(Hy_square, target_frequency, t_H)

# fig, ax = plt.subplots()
# ax.scatter(x_square_1d_Ez, y_square_1d_Ez)

E_to_cross = np.zeros((3, len(Ez_square)), dtype=np.complex128)
H_to_cross = np.zeros((3, len(Hx_square)), dtype=np.complex128)

E_to_cross[2] = Ez_at_f
H_to_cross[0] = Hx_at_f
H_to_cross[1] = Hy_at_f

M_eff = -np.cross(ns_Ez, E_to_cross, axisa=0, axisb=0)
J_eff = np.cross(ns_Hx, H_to_cross, axisa=0, axisb=0)

# fig, ax = plt.subplots(1, 3)
# marker_size = 50
# ax[0].scatter(
#     x_square_1d_Ez, y_square_1d_Ez, marker_size, c=np.abs(M_Ez_eff[:, 0]), cmap="jet"
# )
# ax[1].scatter(
#     x_square_1d_Ez, y_square_1d_Ez, marker_size, c=np.abs(M_Ez_eff[:, 1]), cmap="jet"
# )
# ax[2].scatter(
#     x_square_1d_Ez, y_square_1d_Ez, marker_size, c=np.abs(M_Ez_eff[:, 2]), cmap="jet"
# )
# ax[0].scatter(
#     x_square_1d_Hx,
#     y_square_1d_Hx,
#     marker_size,
#     c=np.abs(J_Hx_eff[:, 0]),
#     cmap="bwr",
#     marker="v",
# )
# ax[1].scatter(
#     x_square_1d_Hx,
#     y_square_1d_Hx,
#     marker_size,
#     c=np.abs(J_Hx_eff[:, 1]),
#     cmap="bwr",
#     marker="v",
# )
# ax[2].scatter(
#     x_square_1d_Hx,
#     y_square_1d_Hx,
#     marker_size,
#     c=np.abs(J_Hx_eff[:, 2]),
#     cmap="bwr",
#     marker="v",
# )
# ax[0].scatter(
#     x_square_1d_Hy,
#     y_square_1d_Hy,
#     marker_size,
#     c=np.abs(J_Hy_eff[:, 0]),
#     cmap="viridis",
#     marker="X",
# )
# ax[1].scatter(
#     x_square_1d_Hy,
#     y_square_1d_Hy,
#     marker_size,
#     c=np.abs(J_Hy_eff[:, 1]),
#     cmap="viridis",
#     marker="X",
# )
# ax[2].scatter(
#     x_square_1d_Hy,
#     y_square_1d_Hy,
#     marker_size,
#     c=np.abs(J_Hy_eff[:, 2]),
#     cmap="viridis",
#     marker="X",
# )


phi = np.linspace(0, 2.0 * np.pi, 1000)
theta = np.pi / 2

Nz = np.zeros_like(phi, dtype=np.complex128)
Lx = np.zeros_like(phi, dtype=np.complex128)
Ly = np.zeros_like(phi, dtype=np.complex128)

k0 = 2.0 * np.pi * target_frequency * np.sqrt(epsilon_0 * mu_0)
Z0 = np.sqrt(mu_0 / epsilon_0)
for i, theta_i in enumerate(phi):
    Nz[i] = np.sum(
        dx
        * J_eff[:, 2]
        * np.exp(
            1j * k0 * (np.cos(theta_i) * x_square_1d + np.sin(theta_i) * y_square_1d)
        )
    )
    Lx[i] = np.sum(
        dx
        * M_eff[:, 0]
        * np.exp(
            1j * k0 * (np.cos(theta_i) * x_square_1d + np.sin(theta_i) * y_square_1d)
        )
    )
    Ly[i] = np.sum(
        dx
        * M_eff[:, 1]
        * np.exp(
            1j * k0 * (np.cos(theta_i) * x_square_1d + np.sin(theta_i) * y_square_1d)
        )
    )

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(phi, np.real(Nz))
# ax[0].plot(phi, np.imag(Nz))
# ax[1].plot(phi, np.real(Lx))
# ax[1].plot(phi, np.imag(Lx))
# ax[2].plot(phi, np.real(Ly))
# ax[2].plot(phi, np.imag(Ly))

N = np.zeros((3, len(Nz)), dtype=Nz.dtype)
N[2, :] = Nz

L = np.zeros((3, len(Lx)), dtype=Lx.dtype)
L[0, :] = Lx
L[1, :] = Ly

N_spherical = cartesian_vector_to_spherical(N, theta * np.ones_like(phi), phi)
L_spherical = cartesian_vector_to_spherical(L, theta * np.ones_like(phi), phi)

# fig, ax = plt.subplots(2, 3)
# ax[0][0].plot(phi, np.real(N_spherical[0, :]))
# ax[0][0].plot(phi, np.imag(N_spherical[0, :]))
# ax[0][1].plot(phi, np.real(N_spherical[1, :]))
# ax[0][1].plot(phi, np.imag(N_spherical[1, :]))
# ax[0][2].plot(phi, np.real(N_spherical[2, :]))
# ax[0][2].plot(phi, np.imag(N_spherical[2, :]))

# ax[1][0].plot(phi, np.real(L_spherical[0, :]))
# ax[1][0].plot(phi, np.imag(L_spherical[0, :]))
# ax[1][1].plot(phi, np.real(L_spherical[1, :]))
# ax[1][1].plot(phi, np.imag(L_spherical[1, :]))
# ax[1][2].plot(phi, np.real(L_spherical[2, :]))
# ax[1][2].plot(phi, np.imag(L_spherical[2, :]))

r = 1000.0
E_theta = (
    -1j
    * k0
    * np.exp(-1j * k0 * r)
    / (4.0 * np.pi * r)
    * (L_spherical[2, :] + Z0 * N_spherical[1, :])
)
E_phi = (
    1j
    * k0
    * np.exp(-1j * k0 * r)
    / (4.0 * np.pi * r)
    * (L_spherical[1, :] + Z0 * N_spherical[2, :])
)


def mgpulse_ft(f, t_sig, frequency, t0=0.0):
    G = lambda ff: t_sig * np.sqrt(2 * np.pi) * np.exp(-2.0 * (np.pi * t_sig * ff) ** 2)
    return np.exp(-2j * np.pi * f * t0) * (G(f - frequency) - G(f + frequency)) / 2j


# fig, ax = plt.subplots(1, 2)
# ax[0].plot(phi, np.real(E_theta), linewidth=2)
# ax[0].plot(phi, np.imag(E_theta), linewidth=2)
# ax[1].plot(phi, np.real(E_phi), linewidth=2)
# ax[1].plot(phi, np.imag(E_phi), linewidth=2)


t_sig = 3.0 / target_frequency
t0 = 4.0 * t_sig
E_inc = mgpulse_ft(
    target_frequency, t_sig, target_frequency, t0
)  # f_src = the pulse's center frequency
sigma_2d = (
    (k0 / 4) * np.abs(L_spherical[2] + Z0 * N_spherical[1]) ** 2 / np.abs(E_inc) ** 2
)

compare_data = np.load("tmz_pec_compare.npz")
phi_compare = compare_data["phi"]
result_compare = compare_data["result"]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, layout="constrained", figsize=(8,8))
ax.set_rticks([-25, -15])
ax.set_rlabel_position(90)
ax.set_xlabel(r"$\phi$")
ax.plot(
    phi_compare,
    20.0 * np.log10(result_compare),
    linewidth=2,
    color="black",
    alpha=0.5,
    label=r"$dB(\sigma_{\text{2D}}^{\text{\Sigma}})$",
)
ax.plot(
    phi,
    20.0 * np.log10(sigma_2d),
    linewidth=2,
    color="red",
    alpha=0.5,
    label=r"$dB(\sigma_{\text{2D}}^{FDTD})$",
)
ax.legend()
fig.savefig("plots/bist_pec_tmz.png", dpi=200)


plt.show()
