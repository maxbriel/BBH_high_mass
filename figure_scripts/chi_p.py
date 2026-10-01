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

def precession(theta_1, theta_2, a1, a2, m1, m2):
    """Effective precession spin (Gerosa+2021, Eq. 14), ordered by mass."""
    swap = m2 > m1
    m_hi, m_lo = np.where(swap, m2, m1), np.where(swap, m1, m2)
    a_hi, a_lo = np.where(swap, a2, a1), np.where(swap, a1, a2)
    t_hi, t_lo = np.where(swap, theta_2, theta_1), np.where(swap, theta_1, theta_2)
    q = m_lo / m_hi
    chi_p = np.maximum(a_hi * np.abs(np.sin(t_hi)),
                       q * (4*q + 3) / (4 + 3*q) * a_lo * np.abs(np.sin(t_lo)))
    return chi_p

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

# define limits

MASS_CUTOFF = 39.76734837  # Mass cutoff for the analysis
Z_MIN = 0.15
Z_MAX = 0.25

# Plot mean with filled uncertainty region
fig, ax = plt.subplots(1, 1, figsize=(3.38, 2.535*0.7))

# Define the data directory and folder types
data_dir = "../data/main_figure/"
SFH_type = "IllustrisTNG"


title_mapping = {
    "no_kick": "",
    "low_kick": "-Low kick",
    "normal_kick": "-Normal kick",
}

cm = Colormap("tol:vibrant")
colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

# define bins for histograms
chi_p_bins = np.linspace(-0.1, 1.1, 51)


def get_histogram_data(co_contact_file):
    data = Rates(co_contact_file, "BBH", SFH_type)
    print(co_contact_file)

    max_mass = np.maximum(
        data.population["S1_mass"].values, data.population["S2_mass"].values
    )
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    volume = get_shell_comoving_volume(Z_MIN, Z_MAX)

    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chi_p = precession(
        filtered_population["S1_spin_orbit_tilt"].to_numpy(),
        filtered_population["S2_spin_orbit_tilt"].to_numpy(),
        filtered_population["S1_spin"].to_numpy(),
        filtered_population["S2_spin"].to_numpy(),
        filtered_population["S1_mass"].to_numpy(),
        filtered_population["S2_mass"].to_numpy(),
    )

    # get histogram
    h, _ = np.histogram(
        chi_p,
        bins=chi_p_bins,
        weights=np.nansum(weights, axis=1) / volume,
    )

    return h / np.diff(chi_p_bins)


data_dir = "../data/kick_figure_GRMHD/"
folder_types = ["low_kick", "normal_kick"]
linestyles = ["solid", "dashed", "dotted"]

SFH_type = "IllustrisTNG"
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(
        data_dir,
        folder_type + ".h5",
    )

    h = get_histogram_data(co_contact_file)
    # already divided by bin width

    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type    

    plt.step(
        chi_p_bins[:-1],
        h,
        lw=2,
        where="post",
        label="GRRMHD" + label,
        color=colours[i + 1],
    )

data_dir = "../data/kicks_conservative/"
folder_types = ["low_kick", "normal_kick"]
SFH_type = "IllustrisTNG"
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(
        data_dir,
        folder_type + ".h5",
    )

    h = get_histogram_data(co_contact_file)

    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type

    plt.step(
        chi_p_bins[:-1],
        h,
        lw=2,
        where="post",
        label="Conservative" + label,
        color=colours[i + 4],
    )

plt.ylim(0)
plt.xlim(-0.03, 1.0)
plt.xlabel("$\chi_{p}$")
plt.ylabel("d$\mathcal{R}$/d$\chi_{p}$ [Gpc$^{-3}$ yr$^{-1}$]")
plt.legend(ncol=1, bbox_to_anchor=(0.95, 1.0), loc="upper right")

output_dir = "../figures"
plt.savefig(f"{output_dir}/png/intrinsic_chi_p.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/pdf/intrinsic_chi_p.pdf", bbox_inches="tight")
