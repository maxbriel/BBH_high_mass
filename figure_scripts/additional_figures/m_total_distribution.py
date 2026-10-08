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

MASS_CUTOFF = 39.76734837
Z_MIN = 0.15
Z_MAX = 0.25
Z_EVAL = 0.2

# LVK GWTC-5.0 BGP data at z = 0: the 1D dR/dm1 sets the rate and the 2D map,
# a pdf in (ln m1, q) bins, sets the shape in (m1, q)
data_file = "../../data/hm_dists_m1rates_gwtc5_39.h5"
with h5py.File(data_file, "r") as hf:
    m1_grid = hf["1D"]["mass1"][()]
    rate_m1 = np.median(hf["1D"]["p_mass1"][()], axis=0)
    m1_edges = hf["2D"]["mass1"][()]
    q_edges = hf["2D"]["mass_ratio"][()]
    p_lnm1_q = hf["2D"]["p_m1q"][()]

# scale to z = 0.2, as for the main figures
lvk_rate = np.trapz(rate_m1, m1_grid) * (1 + Z_EVAL) ** 2.7

# spread each (ln m1, q) bin uniformly over sub-points and bin in M_total = m1 (1 + q)
n_sub = 50
f_sub = (np.arange(n_sub) + 0.5) / n_sub
lnm1 = np.log(m1_edges[:-1])[:, None] + np.diff(np.log(m1_edges))[:, None] * f_sub
q = q_edges[:-1][:, None] + np.diff(q_edges)[:, None] * f_sub
bin_prob = p_lnm1_q * np.diff(np.log(m1_edges))[:, None] * np.diff(q_edges)[None, :]
M_total = np.exp(lnm1)[:, None, :, None] * (1 + q[None, :, None, :])
M_weights = np.broadcast_to((bin_prob / n_sub**2)[:, :, None, None], M_total.shape)

total_mass_bins = np.linspace(40, 300, 131)
h_lvk, _ = np.histogram(M_total.ravel(), bins=total_mass_bins, weights=M_weights.ravel())

fig, ax = plt.subplots(1, 1, figsize=(3.38, 2.535))

plt.stairs(
    lvk_rate * h_lvk / np.diff(total_mass_bins),
    total_mass_bins,
    baseline=None,
    color="black",
    linewidth=2,
    label=r"LVK $\texttt{GWTC-5.0}$",
)

# Define the data directory and folder types
data_dir = "../../data/main_figure/"
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
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)
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


# figure_2: conservative accretion with kicks; the no-kick run is main_figure/conservative.h5
figure_2_files = {
    "no_kick": "../../data/main_figure/conservative.h5",
    "low_kick": "../../data/kicks_conservative/low_kick.h5",
    "normal_kick": "../../data/kicks_conservative/normal_kick.h5",
}
folder_types = ["no_kick", "low_kick", "normal_kick"]
linestyles = ["solid", "dashed", "dotted"]

SFH_type = "IllustrisTNG"
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = figure_2_files[folder_type]
    data = Rates(co_contact_file, "BBH", SFH_type)
    print("Rates data loaded.")
    max_mass = np.maximum(
        data.population["S1_mass"].values, data.population["S2_mass"].values
    )
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]

    total_mass = (
        filtered_population["S1_mass"].to_numpy()
        + filtered_population["S2_mass"].to_numpy()
    )
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

data_dir = "../../data/kick_figure_GRMHD/"
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
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)
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
        label="GRRMHD" + label,
        color=colours[3],
        ls=linestyles[i],
    )


# plt.ylim(0, 10)
plt.xlim(40, 300)
plt.xlabel(r"$M_\mathrm{total} \, (M_{\odot})$")
plt.ylabel("Rate density $(\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_{\odot}^{-1})$")
plt.legend(bbox_to_anchor=(0, -0.2), loc="upper left", ncol=2)
output_dir = "../../figures"
plt.savefig(
    f"{output_dir}/png/intrinsic_Mtotal_distribution.png", dpi=300, bbox_inches="tight"
)
plt.savefig(f"{output_dir}/pdf/intrinsic_Mtotal_distribution.pdf", bbox_inches="tight")
