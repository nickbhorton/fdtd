from math import floor

import matplotlib.pyplot as plt
import numpy as np

from solver import Ez_pw, Hy_pw

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


def field_plot(data, index, field_fun, field_str, surf_offset, suptitle, savefig_title):

    t = data["t"]
    x = data[f"x_{field_str}"]
    y = data[f"y_{field_str}"]
    field = data[f"{field_str}"]

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     np.arange(len(t)), [Ez_pw(10e9, x_Ez[0, 0], ti, 1.0) for ti in t]
    # )

    percent_inset_PML = 0.1666666666666666
    x_left_PML = (x.max() - x.min()) * percent_inset_PML
    x_right_PML = (x.max() - x.min()) * (1 - percent_inset_PML)
    y_bot_PML = (y.max() - y.min()) * percent_inset_PML
    y_top_PML = (y.max() - y.min()) * (1 - percent_inset_PML)

    percent_inset_surf = percent_inset_PML + surf_offset
    x_left_surf = (x.max() - x.min()) * percent_inset_surf
    x_right_surf = (x.max() - x.min()) * (1 - (percent_inset_surf - 0.001))
    y_bot_surf = (y.max() - y.min()) * percent_inset_surf
    y_top_surf = (y.max() - y.min()) * (1 - percent_inset_surf)

    ix_PML = np.where((x[0, :] > x_left_PML) & (x[0, :] < x_right_PML))[0]
    iy_PML = np.where((y[:, 0] > y_bot_PML) & (y[:, 0] < y_top_PML))[0]
    ix_surf = np.where((x[0, :] > x_left_surf) & (x[0, :] < x_right_surf))[0]
    iy_surf = np.where((y[:, 0] > y_bot_surf) & (y[:, 0] < y_top_surf))[0]

    wavelength = 299792458.0 / 10e9

    fig1, ax1 = plt.subplots(2, 2, layout="constrained", figsize=(8, 8))

    x_Ez_i = x[np.ix_(iy_PML, ix_PML)]
    y_Ez_i = y[np.ix_(iy_PML, ix_PML)]

    Ez_i_scattered = field[index].copy()
    Ez_i_scattered[np.ix_(iy_surf, ix_surf)] -= field_fun(
        10e9, x[np.ix_(iy_surf, ix_surf)], t[index], 1.0
    )
    Ez_i_scattered = Ez_i_scattered[np.ix_(iy_PML, ix_PML)]

    Ez_i_total = Ez_i_scattered + field_fun(10e9, x_Ez_i, t[index], 1.0)
    Ez_i_inc = field_fun(10e9, x_Ez_i, t[index], 1.0)

    data_list = [Ez_i_inc, Ez_i_scattered, Ez_i_total, field[index]]
    vmin = [d.min() for d in data_list]
    vmax = [d.max() for d in data_list]
    vv = [max(np.abs(vm), np.abs(vx)) for vm, vx in zip(vmin, vmax)]
    mesh1_11 = ax1[0][0].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_inc,
        cmap="bwr",
        vmin=-vv[0],
        vmax=vv[0],
    )
    mesh1_01 = ax1[0][1].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_scattered,
        cmap="bwr",
        vmin=-vv[1],
        vmax=vv[1],
    )
    mesh1_10 = ax1[1][0].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_total,
        cmap="bwr",
        vmin=-vv[2],
        vmax=vv[2],
    )
    mesh1_00 = ax1[1][1].pcolormesh(
        x / wavelength,
        y / wavelength,
        field[index],
        cmap="bwr",
        vmin=-vv[3],
        vmax=vv[3],
    )
    ax1[0][0].set_xticks([1, 5 / 2, 5], labels=["-2", "0", "2"])
    ax1[0][1].set_xticks([1, 5 / 2, 5], labels=["-2", "0", "2"])
    ax1[1][0].set_xticks([1, 5 / 2, 5], labels=["-2", "0", "2"])
    ax1[1][1].set_xticks([0, 6 / 2, 6], labels=["-3", "0", "3"])

    for ax in ax1.flat:
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")

    ax1[0][0].set_title("Incident")
    ax1[0][1].set_title("Scattered")
    ax1[1][0].set_title("Total")
    ax1[1][1].set_title("Simulation")

    cbar1 = fig1.colorbar(mesh1_00)
    cbar2 = fig1.colorbar(mesh1_01)
    cbar3 = fig1.colorbar(mesh1_10)
    cbar4 = fig1.colorbar(mesh1_11)
    cbars = [cbar1, cbar2, cbar3, cbar4]
    for cbar in cbars:
        cbar.ax.tick_params(labelsize=7)
    fig1.suptitle(suptitle)
    fig1.savefig(savefig_title, dpi=200)


data = [np.load("data/plots_tmz_pec_fields_40_40_1k.npz")]

index = [410, 410, 410]
funs = [Ez_pw, lambda x1, x2, x3, x4: np.zeros_like(x2), Hy_pw]
field_str = ["Ez", "Hx", "Hy"]
suptitles = [r"$E_z^{\text{TM}_z}$", r"$H_x^{\text{TM}_z}$", r"$H_y^{\text{TM}_z}$"]
savefigs = [
    "plots/sim_pec_tmz_Ez_fields.png",
    "plots/sim_pec_tmz_Hx_fields.png",
    "plots/sim_pec_tmz_Hy_fields.png",
]

for i in range(len(index)):
    field_plot(
        data[floor(i / 3)],
        index[i],
        funs[i],
        field_str[i],
        0.2,
        suptitles[i],
        savefigs[i],
    )
# plt.show()
