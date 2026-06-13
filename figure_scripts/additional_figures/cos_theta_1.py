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


# Define the data directory and folder types
data_dir = "/home/users/b/briel/scratch/high_mass_physics/data/main_figure/"
folder_types = ["Eddington-limited", "GRMHD", "conservative"]
SFH_type = "IllustrisTNG"

title_mapping = {
    "no_kick": "",
    "low_kick": "-Low",
    "normal_kick": "-Normal",
}

cm = Colormap("tol:vibrant")

colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

cos_theta_bins = np.linspace(-1, 1, 51)

for i, folder_type in enumerate(folder_types[:-2]):
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
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chirp_mass = filtered_population["chirp_mass"].to_numpy()
    mass_ratio = filtered_population["mass_ratio"].to_numpy()
    S1_tilt = np.where(
        filtered_population["S1_mass"] >= filtered_population["S2_mass"],
        filtered_population["S1_spin_orbit_tilt"].to_numpy(),
        filtered_population["S2_spin_orbit_tilt"].to_numpy(),
    )
    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    h, _ = np.histogram(
        np.cos(S1_tilt),
        bins=cos_theta_bins,
        weights=np.nansum(weights, axis=1) / volume,
        density=True,
    )

    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.step(cos_theta_bins[:-1], h, lw=1, label="Eddington", color=colours[i])


data_dir = "/home/users/b/briel/scratch/high_mass_physics/data/figure_2/"
folder_types = ["no_kick", "low_kick", "normal_kick"]
linestyles = ["solid", "dashed", "dotted"]

SFH_type = "IllustrisTNG"
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
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chirp_mass = filtered_population["chirp_mass"].to_numpy()
    mass_ratio = filtered_population["mass_ratio"].to_numpy()
    S1_tilt = np.where(
        filtered_population["S1_mass"] >= filtered_population["S2_mass"],
        filtered_population["S1_spin_orbit_tilt"].to_numpy(),
        filtered_population["S2_spin_orbit_tilt"].to_numpy(),
    )
    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    h, _ = np.histogram(
        np.cos(S1_tilt),
        bins=cos_theta_bins,
        weights=np.nansum(weights, axis=1) / volume,
        density=True,
    )

    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.step(
        cos_theta_bins[:-1],
        h,
        lw=1,
        label="Cons." + label,
        color=colours[5],
        ls=linestyles[i],
    )

data_dir = "/home/users/b/briel/scratch/high_mass_physics/data/kick_figure_GRMHD/"
folder_types = ["no_kick", "low_kick", "normal_kick"]
SFH_type = "IllustrisTNG"
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
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chirp_mass = filtered_population["chirp_mass"].to_numpy()
    mass_ratio = filtered_population["mass_ratio"].to_numpy()
    S1_tilt = np.where(
        filtered_population["S1_mass"] >= filtered_population["S2_mass"],
        filtered_population["S1_spin_orbit_tilt"].to_numpy(),
        filtered_population["S2_spin_orbit_tilt"].to_numpy(),
    )
    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    h, _ = np.histogram(
        np.cos(S1_tilt),
        bins=cos_theta_bins,
        weights=np.nansum(weights, axis=1) / volume,
        density=True,
    )

    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.step(
        cos_theta_bins[:-1],
        h,
        lw=1,
        label="GRMHD" + label,
        color=colours[3],
        ls=linestyles[i],
    )


plt.ylim(0, 10)
plt.xlim(-1.0, 1.0)
plt.xlabel(r"$\cos(\theta_1)$")
plt.ylabel("PDF")
plt.legend(ncol=2)

output_dir = "/home/users/b/briel/scratch/high_mass_physics/figures"
plt.savefig(f"{output_dir}/intrinsic_cos_theta_1.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/intrinsic_cos_theta_1.pdf", bbox_inches="tight")
