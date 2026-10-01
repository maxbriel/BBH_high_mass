import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from posydon.popsyn.synthetic_population import Rates

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

# load the LVK GWTC-5.0 BGP data; the 2D map is a pdf in (ln m1, q) bins
data_file = "../../data/hm_dists_m1rates_gwtc5_39.h5"
with h5py.File(data_file, "r") as hf:
    m1_edges = hf["2D"]["mass1"][()]
    q_edges = hf["2D"]["mass_ratio"][()]
    p_lnm1_q = hf["2D"]["p_m1q"][()]

# pdf per unit m1 and q
m1_centres = np.sqrt(m1_edges[1:] * m1_edges[:-1])
pdf_m1q = p_lnm1_q / m1_centres[:, None]


def lvk_pdf(m1, q):
    """Binned BGP pdf p(m1, q), zero outside the BGP bins."""
    i = np.searchsorted(m1_edges, m1) - 1
    j = np.searchsorted(q_edges, q) - 1
    inside = (i >= 0) & (i < len(m1_edges) - 1) & (j >= 0) & (j < len(q_edges) - 1)
    pdf = np.zeros(np.shape(m1))
    pdf[inside] = pdf_m1q[i[inside], j[inside]]
    return pdf


high_res_mchirp_bins = np.linspace(1, 200, 2000)
high_res_q_bins = np.linspace(0.0001, 1, 2000)
X, Y = np.meshgrid(high_res_mchirp_bins, high_res_q_bins, indexing="ij")
high_res_m1_values = Y ** (-3 / 5) * (1 + Y) ** (1 / 5) * X
# p(Mchirp, q) = p(m1, q) dm1/dMchirp
Z = lvk_pdf(high_res_m1_values, Y) * Y ** (-3 / 5) * (1 + Y) ** (1 / 5)
# make into a density
Z = Z / np.sum(Z)
pdf_levels = [
    np.quantile(Z, 0.05),
    np.quantile(Z, 0.95),
    np.quantile(Z, 0.99),
]

cm_blues = Colormap("colorbrewer:Blues")
pdf_colours = cm_blues([0.2, 0.6, 1.0])
cm_grays = Colormap("colorbrewer:Greys")


# figure_2: conservative accretion with kicks; the no-kick run is main_figure/conservative.h5
figure_2_files = {
    "no_kick": "../../data/main_figure/conservative.h5",
    "low_kick": "../../data/kicks_conservative/low_kick.h5",
    "normal_kick": "../../data/kicks_conservative/normal_kick.h5",
}
folder_types = ["no_kick", "low_kick", "normal_kick"]
SFH_type = "IllustrisTNG"

mass_bins = np.linspace(10, 200, 51)
q_bins = np.linspace(0, 1, 51)
chi_bins = np.linspace(-0.2, 1, 51)

fig, axes = plt.subplots(2, 3, figsize=(3.38 * 2, 2.535 * 2))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = figure_2_files[folder_type]
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
    chi_eff = filtered_population["chi_eff"].to_numpy()

    # Top row: chirp mass vs mass ratio
    H0, xedges0, yedges0 = np.histogram2d(
        chirp_mass,
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

    axes[0, i].imshow(
        H0.T,
        origin="lower",
        extent=[xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]],
        cmap=cm_grays.to_mpl(),
        aspect="auto",
        interpolation="bilinear",
    )
    axes[0, i].contour(
        H0.T,
        levels=levels0,
        colors=cm_grays([0.2, 0.6, 1.0]),
        linewidths=[1.0, 1, 1],
        extent=[xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]],
    )

    axes[0, i].contour(
        X, Y, Z, levels=pdf_levels, colors=pdf_colours, linewidths=[1.5, 1.5, 1.5]
    )

    # Top bottom row: Chirp mass vs chi_eff
    H1, xedges1, yedges1 = np.histogram2d(
        chirp_mass,
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

    axes[1, i].imshow(
        H1.T,
        origin="lower",
        extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
        cmap=cm_grays.to_mpl(),
        aspect="auto",
        interpolation="bilinear",
    )

    axes[1, i].contour(
        H1.T,
        levels=levels1,
        colors=cm_grays([0.2, 0.6, 1.0]),
        linewidths=[1.0, 1, 1],
        extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
    )

    # add rate denisty
    rate_density = np.nansum(weights) / volume
    axes[1, i].text(
        0.95,
        0.95,
        r"$\mathcal{{R}}_{0{-}2}=$"
        + f"{rate_density:.1f}"
        + r"$\,\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$",
        transform=axes[1, i].transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            alpha=0.6,
            edgecolor="grey",
            lw=0.5,
        ),
    )

# axes and stuff
axes[0, 0].set_ylabel(r"$q=M_\mathrm{min}/M_\mathrm{max}$")
axes[1, 0].set_ylabel(r"$\chi_\mathrm{eff}$")


title_mapping = {
    "no_kick": "No kicks",
    "low_kick": "Low kicks",
    "normal_kick": "Normal kicks",
}

for ax in axes[0, :]:
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position("both")
    ax.set_ylim(1e-2, 1)
    ax.set_title(title_mapping[folder_types[axes[0, :].tolist().index(ax)]])

for ax in axes.flatten():
    ax.set_xlabel(r"$M_\mathrm{chirp}$ [$M_\odot$]")
    ax.set_xlim(10, 200)

for ax in axes[:, 1:].flatten():
    ax.set_yticklabels([])

for ax in axes.flatten():
    ax.grid(ls="--", alpha=0.5)

output_dir = "../../figures"
plt.savefig(f"{output_dir}/png/intrinsic_figure_2_kicks.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/pdf/intrinsic_figure_2_kicks.pdf", bbox_inches="tight")
