import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmap import Colormap
from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from posydon.popsyn.synthetic_population import Rates
from scipy.stats import gaussian_kde

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

MASS_CUTOFF = 39.76734837  # Mass cutoff for the analysis
Z_MIN = 0.15
Z_MAX = 0.25
Z_EVAL = 0.2

# load BGP data from Anarya (send on slack)
#data_file = "../data/hm_dists_gwtc5_40.h5"
data_file = "../data/hm_dists_m1rates_gwtc5_39.h5"


with h5py.File(data_file, "r") as hf:
    mass_ratio_bins_model = hf["1D"]["mass_ratio"][:]
    pdf_mass_ratio = hf["1D"]["p_mass_ratio"][:]


Rp_mass_ratio = (
    np.array(pdf_mass_ratio)
    / np.trapz(np.array(pdf_mass_ratio), mass_ratio_bins_model, axis=1)[:, None]
)
Rpm_5 = np.percentile(Rp_mass_ratio, q=5, axis=0)
Rpm_95 = np.percentile(Rp_mass_ratio, q=95, axis=0)
R_pm_med = np.percentile(Rp_mass_ratio, q=50, axis=0)


# define colourmaps
cm = Colormap("tol:vibrant")
colours = cm([0.1, 0.5, 0.8])

# define bins
mass_ratio_bins = np.linspace(0, 1.1, 31)

# setup figure
fig, axes = plt.subplots(1, 3, figsize=(3.38 * 2, 2.535 * 0.7))

# plot BGP distribution
for ax in axes:
    ax.fill_between(
        mass_ratio_bins_model, Rpm_5, Rpm_95, alpha=0.3, color="gray", label="LVK $\\texttt{GWTC-5.0}$"
    )


# Define the data directory and folder types
data_dir = "../data/main_figure/"
folder_types = ["Eddington-limited", "GRMHD", "conservative"]
SFH_type = "IllustrisTNG"

ax = axes[0]


def get_histogram_data(co_contact_file):

    data = Rates(co_contact_file, "BBH", SFH_type)

    max_mass = np.maximum(
        data.population["S1_mass"].values, data.population["S2_mass"].values
    )
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)

    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    mass_ratio = filtered_population["mass_ratio"].to_numpy()

    # get histogram
    h, _ = np.histogram(
        mass_ratio,
        bins=mass_ratio_bins,
        weights=np.nansum(weights, axis=1) / volume,
    )

    return h / np.diff(mass_ratio_bins)


# no kick population
for i, folder_type in enumerate(folder_types):
    co_contact_file = os.path.join(
        data_dir,
        folder_type + ".h5",
    )
    h = get_histogram_data(co_contact_file)
    # set label
    label = folder_type
    if folder_type == "conservative":
        label = "Conservative"
    elif folder_type == "GRMHD":
        label = "GRRMHD"

    ax.step(
        mass_ratio_bins[:-1],
        h / np.sum(h * np.diff(mass_ratio_bins)),
        lw=2,
        label=label,
        color=colours[i],
        where="post",
    )


# low kicked populations
co_contact_file = "../data/" + "kicks_Eddington-limited/low_kick.h5"
h = get_histogram_data(co_contact_file)
axes[1].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="Eddington-limited",
    color=colours[0],
    where="post",
)

co_contact_file = "../data/" + "kick_figure_GRMHD/low_kick.h5"
h = get_histogram_data(co_contact_file)
axes[1].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="GRMHD",
    color=colours[1],
    where="post",
)

co_contact_file = "../data/" + "kicks_conservative/low_kick.h5"
h = get_histogram_data(co_contact_file)
axes[1].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="Conservative",
    color=colours[2],
    where="post",
)
axes[1].set_title("Low kick")

# normal kick population

co_contact_file = "../data/" + "kicks_Eddington-limited/normal_kick.h5"
h = get_histogram_data(co_contact_file)
axes[2].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="Eddington-limited",
    color=colours[0],
    where="post",
)

co_contact_file = "../data/" + "kick_figure_GRMHD/normal_kick.h5"
h = get_histogram_data(co_contact_file)
axes[2].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="GRMHD",
    color=colours[1],
    where="post",
)

co_contact_file = "../data/" + "kicks_conservative/normal_kick.h5"
h = get_histogram_data(co_contact_file)
axes[2].step(
    mass_ratio_bins[:-1],
    h / np.sum(h * np.diff(mass_ratio_bins)),
    lw=2,
    label="Conservative",
    color=colours[2],
    where="post",
)

axes[2].set_title("Normal kick")

ax.set_title("No kick")

for ax in axes:
    ax.set_yscale("log")
    ax.set_xlim(0.0001, 0.999)
    ax.set_ylim(5e-2, 10)
    ax.set_xlabel(r"$q = \mathrm{M}_2/\mathrm{M}_1$")

axes[1].set_yticks([])
axes[2].set_yticks([])

axes[0].set_ylabel("PDF")


axes[0].legend(bbox_to_anchor=(0.1, -0.25), loc="upper left", ncol=4)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

output_dir = "../figures"
plt.savefig(
    f"{output_dir}/png/intrinsic_mass_ratio_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"{output_dir}/pdf/intrinsic_mass_ratio_distribution.pdf", bbox_inches="tight"
)
