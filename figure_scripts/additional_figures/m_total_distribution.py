import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from posydon.popsyn.synthetic_population import Rates
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

# load spline
spline_file = "/home/users/b/briel/scratch/high_mass_physics/data/spline_data/ppd_pdfs_mean_m1qzchieff_mmin40_collector_only_10000w_10000s_rng129_zspline.h5"

file = h5py.File(spline_file, "r")

mass_pdf = file["mass_pdf"][:].T
mass_grid_pdf = file["mgrid"][:]
q_bins_pdf = file["qgrid"][:]

# Transform 2D PDF P(m1, q) to 1D histogram of M_total
# Simple approach: bin each (m1, q) point by its M_total = m1(1 + q)
total_mass_grid = np.linspace(40, 300, 500)
mass_pdf_total = np.zeros_like(total_mass_grid)

# Accumulate PDF values into M_total bins
for i in range(len(mass_grid_pdf)):
    for j in range(len(q_bins_pdf)):
        m1 = mass_grid_pdf[i]
        q = q_bins_pdf[j]
        M_total = m1 * (1 + q)

        # Find the bin for this M_total
        idx = np.searchsorted(total_mass_grid, M_total)
        if 0 < idx < len(total_mass_grid):
            # Add the PDF value (not weighted by grid spacing)
            mass_pdf_total[idx] += mass_pdf[i, j]

# Normalize to make it a proper PDF
dM_total = np.diff(total_mass_grid)
dM_total = np.append(dM_total, dM_total[-1])
norm = np.sum(mass_pdf_total * dM_total)
if norm > 0:
    mass_pdf_total /= norm

# Create interpolator object
pdf_interpolator = interp1d(
    total_mass_grid, mass_pdf_total, kind="cubic", bounds_error=False, fill_value=0.0
)

##mean_pdf = np.mean(mass_pdf_total, axis=0)
# std_pdf = np.std(mass_pdf_total, axis=0)
# Use 95% confidence interval (2.5th to 97.5th percentile)
# percentile_2_5 = np.percentile(mass_pdf_total, 2.5, axis=0)
# percentile_97_5 = np.percentile(mass_pdf_total, 97.5, axis=0)

# Plot mean with filled uncertainty region
fig, ax = plt.subplots(1, 1, figsize=(3.38, 2.535))

plt.plot(total_mass_grid, mass_pdf_total, color="black", linewidth=2, label="Spline")
# plt.fill_between(total_mass_grid, percentile_2_5, percentile_97_5,
#                 color='black',
#                 alpha=0.2,
#                 edgecolor='none')

# Define the data directory and folder types
data_dir = "/home/users/b/briel/scratch/high_mass_physics/data/main_figure/"
folder_types = ["Eddington-limited", "GRMHD", "conservative"]
SFH_type = "IllustrisTNG"


title_mapping = {
    "no_kick": "",
    "low_kick": "-Low",
    "normal_kick": "-Normal",
}

mass_bins = np.linspace(40, 150, 101)

cm = Colormap("tol:vibrant")

colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

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
    total_mass = (
        filtered_population["S1_mass"].to_numpy()
        + filtered_population["S2_mass"].to_numpy()
    )

    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    kde = gaussian_kde(total_mass, weights=np.nansum(weights, axis=1) / volume)
    x_eval = np.linspace(40, 300, 500)
    kde_values = kde(x_eval)
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.plot(
        x_eval,
        kde_values * np.nansum(weights) / volume,
        lw=1,
        label="Eddington",
        color=colours[i],
    )


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

    total_mass = (
        filtered_population["S1_mass"].to_numpy()
        + filtered_population["S2_mass"].to_numpy()
    )
    print(total_mass)
    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    # h, _ = np.histogram(S1_mass,
    #                     bins=mass_bins,
    #                     weights=np.nansum(weights, axis=1)/volume,)

    kde = gaussian_kde(total_mass, weights=np.nansum(weights, axis=1) / volume)
    x_eval = np.linspace(40, 300, 500)
    kde_values = kde(x_eval)
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    # plt.step(mass_bins[:-1],
    #          h/np.diff(mass_bins),
    #          lw=1,
    #          label='Cons.'+label,
    #          color=colours[5],
    #          ls=linestyles[i])
    plt.plot(
        x_eval,
        kde_values * np.nansum(weights) / volume,
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
    total_mass = (
        filtered_population["S1_mass"].to_numpy()
        + filtered_population["S2_mass"].to_numpy()
    )

    print(filtered_population["metallicity"].value_counts())
    # plot KDE
    kde = gaussian_kde(total_mass, weights=np.nansum(weights, axis=1) / volume)
    x_eval = np.linspace(40, 300, 500)
    kde_values = kde(x_eval)
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.plot(
        x_eval,
        kde_values * np.nansum(weights) / volume,
        lw=1,
        label="GRMHD" + label,
        color=colours[3],
        ls=linestyles[i],
    )


# plt.ylim(0, 10)
plt.xlim(40, 300)
plt.xlabel(r"$M_\mathrm{total} \, (M_{\odot})$")
plt.ylabel("Rate density $(\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_{\odot}^{-1})$")
plt.legend(bbox_to_anchor=(0, -0.2), loc="upper left", ncol=2)
output_dir = "/home/users/b/briel/scratch/high_mass_physics/figures"
plt.savefig(
    f"{output_dir}/intrinsic_Mtotal_distribution.png", dpi=300, bbox_inches="tight"
)
plt.savefig(f"{output_dir}/intrinsic_Mtotal_distribution.pdf", bbox_inches="tight")
