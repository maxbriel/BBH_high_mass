import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
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

# Define the data directory and folder types
data_dir = "../data/main_figure/"
folder_types = ["Eddington-limited", "GRMHD", "conservative"]
SFH_type = "IllustrisTNG"

title_mapping = {
    "no_kick": "",
    "low_kick": "-Low",
    "normal_kick": "-Normal",
}

# setup colourmaps
cm = Colormap("tol:vibrant")
colours = cm([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])

# define bins for histograms/contours
chi1_bins = np.linspace(0, 1.1, 36)

def get_histogram_data(co_contact_file):
    
    data = Rates(co_contact_file, "BBH", SFH_type)
    print(co_contact_file)
    max_mass = np.maximum(
        data.population["S1_mass"].values, data.population["S2_mass"].values
    )
    mask = max_mass >=  MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chi1 = np.where(
        filtered_population["S1_mass"] >= filtered_population["S2_mass"],
        filtered_population["S1_spin"].to_numpy(),
        filtered_population["S2_spin"].to_numpy(),
    )

    # get histogram
    h, _ = np.histogram(
        chi1, bins=chi1_bins, weights=np.nansum(weights, axis=1) / volume, density=True
    )
    print("Total rate density:", np.nansum(weights) / volume, "Gpc^-3 yr^-1")
    return h


fig, ax = plt.subplots(1, 1, figsize=(3.38, 2.535))

# Eddington-limited
co_contact_file = os.path.join(
    data_dir,
    "Eddington-limited.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(
    chi1_bins[:-1], h, lw=2, where="post", label="Eddington-limited", color=colours[-1]
)

data_dir = "../data/kick_figure_GRMHD/"
# GRMHD no kick
co_contact_file = os.path.join(
    data_dir,
    "no_kick.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(chi1_bins[:-1], h, lw=2, where="post", label="GRMHD", color=colours[0])

co_contact_file = os.path.join(
    data_dir,
    "low_kick.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(
    chi1_bins[:-1], h, lw=2, where="post", label="GRMHD-Low kick", color=colours[1]
)

co_contact_file = os.path.join(
    data_dir,
    "normal_kick.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(
    chi1_bins[:-1], h, lw=2, where="post", label="GRMHD-Normal kick", color=colours[2]
)


# Conservative no kick

data_dir = "../data/main_figure/"
co_contact_file = os.path.join(
    data_dir,
    "conservative.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(chi1_bins[:-1], h, lw=2, where="post", label="Conservative", color=colours[3])


data_dir = "../data/kicks_conservative/"
co_contact_file = os.path.join(
    data_dir,
    "low_kick.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(
    chi1_bins[:-1], h, lw=2, where="post", label="Cons.-Low kick", color=colours[4]
)

co_contact_file = os.path.join(
    data_dir,
    "normal_kick.h5",
)
h = get_histogram_data(co_contact_file)
plt.step(
    chi1_bins[:-1], h, lw=2, where="post", label="Cons.-Normal kick", color=colours[5]
)


# data_dir = '../data/kick_figure_GRMHD/'
# folder_types = ['no_kick', 'low_kick', 'normal_kick']
# linestyles = ['solid', 'dashed', 'dotted']

# SFH_type = 'IllustrisTNG'
# for i, folder_type in enumerate(folder_types):
#     print(f"Processing folder: {folder_type}")
#     co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
#     #co_contact_file = os.path.join(folder_path, 'CO_contact.h5')
#     h = get_histogram_data(co_contact_file)

#     if folder_type in tuple(title_mapping.keys()):
#         label = title_mapping[folder_type]
#     else:
#         label = folder_type

#     plt.step(chi1_bins[:-1],
#              h,
#              lw=2,
#              where='post',
#              label="GRMHD"+label,
#              color=colours[i+1])


# data_dir = '../data/kicks_conservative/'
# folder_types = [ 'low_kick', 'normal_kick']
# SFH_type = 'IllustrisTNG'
# for i, folder_type in enumerate(folder_types):
#     print(f"Processing folder: {folder_type}")
#     co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
#     #co_contact_file = os.path.join(folder_path, 'CO_contact.h5')
#     h = get_histogram_data(co_contact_file)


#     if folder_type in tuple(title_mapping.keys()):
#         label = title_mapping[folder_type]
#     else:
#         label = folder_type

#     plt.step(chi1_bins[:-1],
#              h,
#              lw=2,
#              where='post',
#              label='Conservative'+label,
#              color=colours[i+4])


# plt.ylim(0, 10)
plt.xlim(0, 1.0)
plt.ylim(0)
# plt.ylim(1e-1, 300)
# plt.ylim(1e-1, 400)
# plt.yscale('log')
plt.xlabel(r"$\chi_{\mathrm{1}}$")
plt.ylabel("PDF")
plt.legend(ncol=2, bbox_to_anchor=(0.45, -0.15), loc="upper center")

output_dir = "../figures"
plt.savefig(f"{output_dir}/png/intrinsic_chi_1.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/pdf/intrinsic_chi_1.pdf", bbox_inches="tight")
