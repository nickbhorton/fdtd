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

data_pec = np.load("data/pec_40_40_1_5k_sin.npz")
t_pec = data_pec["t"]
x_Ez_pec = data_pec["x_Ez"]
y_Ez_pec = data_pec["y_Ez"]
Ez_pec = data_pec["Ez"]
Hx_pec = data_pec["Hx"]
Hy_pec = data_pec["Hy"]

data_die = np.load("data/die_40_40_4k_sin.npz")
t_die = data_die["t"]
x_Ez_die = data_die["x_Ez"]
y_Ez_die = data_die["y_Ez"]
Ez_die = data_die["Ez"]
Hx_die = data_die["Hx"]
Hy_die = data_die["Hy"]


def field_plot(
    t,
    x_Ez,
    y_Ez,
    index,
    field,
    field_fun,
    surf_offset,
    surf_offset_right=0.0,
    suptitle="suptitle",
):

    percent_inset_PML = 0.1
    x_left_PML = (x_Ez.max() - x_Ez.min()) * percent_inset_PML
    x_right_PML = (x_Ez.max() - x_Ez.min()) * (1 - percent_inset_PML)
    y_bot_PML = (y_Ez.max() - y_Ez.min()) * percent_inset_PML
    y_top_PML = (y_Ez.max() - y_Ez.min()) * (1 - percent_inset_PML)

    percent_inset_surf = percent_inset_PML + surf_offset
    x_left_surf = (x_Ez.max() - x_Ez.min()) * percent_inset_surf
    x_right_surf = (x_Ez.max() - x_Ez.min()) * (
        1 - (percent_inset_surf + surf_offset_right)
    )
    y_bot_surf = (y_Ez.max() - y_Ez.min()) * percent_inset_surf
    y_top_surf = (y_Ez.max() - y_Ez.min()) * (1 - percent_inset_surf)

    ix_PML = np.where((x_Ez[0, :] > x_left_PML) & (x_Ez[0, :] < x_right_PML))[0]
    iy_PML = np.where((y_Ez[:, 0] > y_bot_PML) & (y_Ez[:, 0] < y_top_PML))[0]
    ix_surf = np.where((x_Ez[0, :] > x_left_surf) & (x_Ez[0, :] < x_right_surf))[0]
    iy_surf = np.where((y_Ez[:, 0] > y_bot_surf) & (y_Ez[:, 0] < y_top_surf))[0]

    wavelength = 299792458.0 / 10e9

    fig1, ax1 = plt.subplots(2, 2, layout="constrained", figsize=(7, 7))

    x_Ez_i = x_Ez[np.ix_(iy_PML, ix_PML)]
    y_Ez_i = y_Ez[np.ix_(iy_PML, ix_PML)]

    Ez_i_scattered = field[index].copy()
    Ez_i_scattered[np.ix_(iy_surf, ix_surf)] -= field_fun(
        10e9, x_Ez[np.ix_(iy_surf, ix_surf)], t[index], 1.0
    )
    Ez_i_scattered = Ez_i_scattered[np.ix_(iy_PML, ix_PML)]

    Ez_i_total = Ez_i_scattered + field_fun(10e9, x_Ez_i, t[index], 1.0)
    Ez_i_inc = field_fun(10e9, x_Ez_i, t[index], 1.0)

    data_list = [Ez_i_inc, Ez_i_scattered, Ez_i_total, field[index]]
    vmin = min(d.min() for d in data_list)
    vmax = max(d.max() for d in data_list)
    mesh1_11 = ax1[0][0].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_inc,
        cmap="bwr",
    )
    mesh1_01 = ax1[0][1].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_scattered,
        cmap="bwr",
    )
    mesh1_10 = ax1[1][0].pcolormesh(
        x_Ez_i / wavelength,
        y_Ez_i / wavelength,
        Ez_i_total,
        cmap="bwr",
    )
    mesh1_00 = ax1[1][1].pcolormesh(
        x_Ez / wavelength,
        y_Ez / wavelength,
        field[index],
        cmap="bwr",
    )
    ax1[0][0].set_xticks([1.25, 11.25 / 2, 11.25], labels=["-5", "0", "5"])
    ax1[0][1].set_xticks([1.25, 11.25 / 2, 11.25], labels=["-5", "0", "5"])
    ax1[1][0].set_xticks([1.25, 11.25 / 2, 11.25], labels=["-5", "0", "5"])
    ax1[1][1].set_xticks([0, 12.5 / 2, 12.5], labels=["-6", "0", "6"])

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

    return fig1


index_pec = 792  # theta \approx 0
index_pec = 707  # theta \approx 180
index_die = 2234
fig, ax = plt.subplots()
ax.scatter(
    np.arange(len(t_die)), [Ez_pw(10e9, x_Ez_die[0, 0], ti, 1.0) for ti in t_die]
)
print(Ez_die.shape)

fig_TMz_Ez_pec = field_plot(
    t_pec,
    x_Ez_pec,
    y_Ez_pec,
    index_pec,
    Ez_pec,
    Ez_pw,
    0.199,
    0.0,
    r"$E_z^{\text{TM}_z}$",
)
fig_TMz_Hy_pec = field_plot(
    t_pec,
    x_Ez_pec,
    y_Ez_pec,
    index_pec,
    Hy_pec,
    Hy_pw,
    0.199,
    0.001,
    r"$H_y^{\text{TM}_z}$",
)
fig_TMz_Hx_pec = field_plot(
    t_pec,
    x_Ez_pec,
    y_Ez_pec,
    index_pec,
    Hx_pec,
    lambda x1, x2, x3, x4: np.zeros_like(x2),
    0.199,
    0.0,
    r"$H_x^{\text{TM}_z}$",
)
fig_TMz_Ez_die = field_plot(
    t_die,
    x_Ez_die,
    y_Ez_die,
    index_die,
    Ez_die,
    Ez_pw,
    0.199,
    0.0,
    r"$E_z^{\text{TM}_z}$",
)
fig_TMz_Hy_die = field_plot(
    t_die,
    x_Ez_die,
    y_Ez_die,
    index_die,
    Hy_die,
    Hy_pw,
    0.199,
    0.001,
    r"$H_y^{\text{TM}_z}$",
)
fig_TMz_Hx_die = field_plot(
    t_die,
    x_Ez_die,
    y_Ez_die,
    index_die,
    Hx_die,
    lambda x1, x2, x3, x4: np.zeros_like(x2),
    0.199,
    0.0,
    r"$H_x^{\text{TM}_z}$",
)
figs = [
    fig_TMz_Ez_pec,
    fig_TMz_Ez_die,
    fig_TMz_Hx_pec,
    fig_TMz_Hx_die,
    fig_TMz_Hy_pec,
    fig_TMz_Hy_die,
]
for i, fig in enumerate(figs):
    fig.savefig(f"TM_f{i}.png", dpi=200)
plt.show()
