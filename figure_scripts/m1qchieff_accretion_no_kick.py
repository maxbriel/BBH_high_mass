import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib.colors import LogNorm
from matplotlib.ticker import (AutoMinorLocator, LogLocator, MultipleLocator,
                               NullFormatter)
from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from posydon.popsyn.synthetic_population import Rates
from scipy.interpolate import RegularGridInterpolator

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

MASS_CUTOFF = 39.76734837  # Mass cutoff for the analysis
# based on BGP bin

# Load BGP data from Anarya (send on slack)
data_file = "../data/hm_dists_gwtc5_40.h5"

with h5py.File(data_file, "r") as hf:
    matrix1 = hf["2D"]["p_m1q"][()]
    mbins = hf["2D"]["mass1"][()]
    qbins = hf["2D"]["mass_ratio"][()]

    matrix_m1chi = hf["2D"]["p_m1chi"][()]
    mbins = hf["2D"]["mass1"][()]
    chibins = hf["2D"]["chi_eff"][()]

matrix1 = np.array(matrix1) * (1 + 0.2) ** 2.7
matrix_m1chi = np.array(matrix_m1chi) * (1 + 0.2) ** 2.7

# Define colormaps
cm = Colormap("tol:YlOrBr")
cm_grays = Colormap("colorbrewer:Greys")

# Define the data directory and folder types
data_dir = "../data/main_figure/"
folder_types = ["Eddington-limited", "GRMHD", "conservative"]
SFH_type = "IllustrisTNG"

# define bins for histograms/contours
mass_bins = np.linspace(MASS_CUTOFF, 200, 31)
q_bins = np.linspace(0, 1, 31)
chi_bins = np.linspace(-1, 1, 51)

fig, axes = plt.subplots(2, 3, figsize=(3.38 * 2, 2.535 * 1.7))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(
        data_dir,
        folder_type + ".h5",
    )
    # co_contact_file = os.path.join(folder_path, 'CO_contact.h5')
    data = Rates(co_contact_file, "BBH", SFH_type)
    print("Rates data loaded.")
    max_mass = np.maximum(
        data.population["S1_mass"].values, data.population["S2_mass"].values
    )
    mask = max_mass >= MASS_CUTOFF

    z_max = 0.25
    z_min = 0.15
    z_event_mask = (data.z_events >= z_min) & (data.z_events <= z_max)
    volume = get_shell_comoving_volume(z_min, z_max)

    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    M1_mass = np.max(filtered_population[["S1_mass", "S2_mass"]], axis=1).to_numpy()
    mass_ratio = filtered_population["mass_ratio"].to_numpy()
    chi_eff = filtered_population["chi_eff"].to_numpy()

    # Top row: primary mass vs mass ratio
    H0, xedges0, yedges0 = np.histogram2d(
        M1_mass,
        mass_ratio,
        bins=(mass_bins, q_bins),
        weights=np.nansum(weights, axis=1) / volume,
    )

    H0 /= np.diff(mass_bins)[:, None] * np.diff(q_bins)
    H0_sorted = np.sort(H0.flatten())[::-1]
    H0_cumsum = H0_sorted.cumsum()
    H0_total = H0_cumsum[-1]
    # Find contour levels for 1, 2, 3 sigma (containing 68.3%, 95.4%, 99.7% of the total)
    level0_1 = H0_sorted[np.searchsorted(H0_cumsum, 0.683 * H0_total)]
    level0_2 = H0_sorted[np.searchsorted(H0_cumsum, 0.954 * H0_total)]
    level0_3 = H0_sorted[np.searchsorted(H0_cumsum, 0.997 * H0_total)]
    levels0 = [level0_3, level0_2, level0_1]

    x_centers = 0.5 * (xedges0[:-1] + xedges0[1:])
    y_centers = 0.5 * (yedges0[:-1] + yedges0[1:])
    H0_padded = np.pad(H0, pad_width=1, mode="constant", constant_values=0)
    # Extend centers to match the padded array
    dx = x_centers[1] - x_centers[0]
    dy = y_centers[1] - y_centers[0]
    x_centers_padded = np.concatenate(
        [[x_centers[0] - dx], x_centers, [x_centers[-1] + dx]]
    )
    y_centers_padded = np.concatenate(
        [[y_centers[0] - dy], y_centers, [y_centers[-1] + dy]]
    )

    # Use bin edges directly with the histogram data
    # mbins and qbins are already the edges from histogram2d
    axes[0, i].contour(
        x_centers_padded,
        y_centers_padded,
        H0_padded.T,
        levels=levels0,
        colors=cm([0.2, 0.6, 1.0]),
        linewidths=[2.0, 2.0, 2.0],
    )

    axes[0, i].pcolor(
        mbins,
        qbins,
        matrix1.T,
        norm=LogNorm(vmin=matrix1[matrix1 != 0].min(), vmax=matrix1.max()),
        cmap=cm_grays.to_mpl(),
    )

    # Top bottom row: primary mass vs chi_eff
    H1, xedges1, yedges1 = np.histogram2d(
        M1_mass,
        chi_eff,
        bins=(mass_bins, chi_bins),
        weights=np.nansum(weights, axis=1) / volume,
    )
    H1 /= np.diff(mass_bins)[:, None] * np.diff(chi_bins)
    H1_sorted = np.sort(H1.flatten())[::-1]
    H1_cumsum = H1_sorted.cumsum()
    H1_total = H1_cumsum[-1]
    # Find contour levels for 1, 2, 3 sigma (containing 68.3%, 95.4%, 99.7% of the total)
    level1_1 = H1_sorted[np.searchsorted(H1_cumsum, 0.683 * H1_total)]
    level1_2 = H1_sorted[np.searchsorted(H1_cumsum, 0.954 * H1_total)]
    level1_3 = H1_sorted[np.searchsorted(H1_cumsum, 0.997 * H1_total)]
    levels1 = [level1_3, level1_2, level1_1]

    x_centers = 0.5 * (xedges1[:-1] + xedges1[1:])
    y_centers = 0.5 * (yedges1[:-1] + yedges1[1:])
    H1_padded = np.pad(H1, pad_width=1, mode="constant", constant_values=0)
    # Extend centers to match the padded array
    dx = x_centers[1] - x_centers[0]
    dy = y_centers[1] - y_centers[0]
    x_centers_padded = np.concatenate(
        [[x_centers[0] - dx], x_centers, [x_centers[-1] + dx]]
    )
    y_centers_padded = np.concatenate(
        [[y_centers[0] - dy], y_centers, [y_centers[-1] + dy]]
    )

    axes[1, i].contour(
        x_centers_padded,
        y_centers_padded,
        H1_padded.T,
        levels=levels1,
        colors=cm([0.2, 0.6, 1.0]),
        linewidths=[2.0, 2.0, 2.0],
    )

    axes[1, i].pcolor(
        mbins,
        chibins,
        matrix_m1chi.T,
        norm=LogNorm(
            vmin=matrix_m1chi[matrix_m1chi != 0].min(),
            vmax=matrix_m1chi[matrix_m1chi != 0].max(),
        ),
        cmap=cm_grays.to_mpl(),
    )

# axes and stuff
axes[0, 0].set_title("Eddington-limited")
axes[0, 1].set_title("GRRMHD")
axes[0, 2].set_title("Conservative")

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.set_xlim(MASS_CUTOFF, 200)
    ax.xaxis.set_ticks([40, 50, 60, 80, 100, 150, 200])
    ax.set_xticklabels([])

# Top row: x-axis at top
for ax in axes[0, :]:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(r"$\mathrm{M}_{1}$")
    ax.xaxis.set_ticks([40, 50, 60, 80, 100, 150, 200])
    ax.set_xticklabels(["", 50, 60, 80, 100, 150, 200])
    ax.set_ylim(0.1, 1)


# Bottom row: x-axis at bottom
for ax in axes[1, :]:
    ax.set_xlabel(r"$\mathrm{M}_{1}$")
    ax.xaxis.set_ticks([40, 50, 60, 80, 100, 150, 200])
    ax.set_xticklabels(["", 50, 60, 80, 100, 150, 200])
    ax.set_ylim(-1, 1)


axes[0, 0].set_ylabel(r"$q = \mathrm{M}_2/\mathrm{M}_1$")
axes[1, 0].set_ylabel(r"$\chi_\mathrm{eff}$")

for ax in axes[:, 1]:
    ax.set_yticklabels([])

for ax in axes[:, 2]:
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

output_dir = "../figures"
plt.savefig(
    f"{output_dir}/png/m1qchieff_accretion_no_kick.png", bbox_inches="tight", dpi=300
)
plt.savefig(f"{output_dir}/pdf/m1qchieff_accretion_no_kick.pdf", bbox_inches="tight")
