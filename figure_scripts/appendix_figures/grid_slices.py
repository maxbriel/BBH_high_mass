# Load the three different grids for different accretion efficiencies

# grid locations
grids_path = (
    "/srv/astro/projects/posydon/max/population_synthesis/250825_high_mass/grid_links"
)

grid_types = ["eddington_limited", "GRMHD", "conservative"]


import warnings
from pathlib import Path

import cmap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from posydon.config import PATH_TO_POSYDON, PATH_TO_POSYDON_DATA
from posydon.grids.psygrid import PSyGrid
from posydon.popsyn.synthetic_population import Population, Rates
from posydon.utils.common_functions import (
    convert_metallicity_to_string,
    inspiral_timescale_from_separation,
)

# Suppress warnings about missing ini parameters
warnings.filterwarnings("ignore", message="Missing ini parameter:.*")

# Constants
METALLICITY = 0.01
DPI = 300
DONOR_MASS = 33.8767  # 59.4328  # Fixed donor mass in solar masses
HUBBLE_TIME = 13.8e9  # years
MARKER_SIZE = 9
MASS_TOLERANCE = 2.0  # solar masses
CO_TYPE = "BBH"
SFH_IDENTIFIER = "IllustrisTNG"

inspiral_timescale_from_separation = np.vectorize(inspiral_timescale_from_separation)

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

# Color schemes
tol_vibrant = cmap.Colormap("tol:vibrant")(np.linspace(0, 1, 7))


def plot_donor_mass_analysis(
    M_donor, ax, grid, m1_initial, metallicity, hist2d_colour, marker_colours
):
    """
    Plot analysis for a specific donor mass on the given axis.

    Parameters:
    -----------
    M_donor : float
        Donor mass in solar masses
    ax : matplotlib.axes.Axes
        Axis to plot on
    grid : PSyGrid
        Grid object containing stellar evolution data
    m1_initial : array
        Initial masses of primary stars
    metallicity : float
        Metallicity value
    hist2d_colour : colormap
        Colormap for 2D histogram
    marker_colours : array
        Color array for plotting
    """
    # Find the closest donor mass in the grid
    unique_masses = np.unique(m1_initial)
    closest_mass = unique_masses[np.argmin(np.abs(unique_masses - M_donor))]
    donor_mass_mask = m1_initial == closest_mass

    print(f"    Using donor mass: {closest_mass:.4f} M☉ (requested: {M_donor:.1f} M☉)")

    smt_mask = (
        grid.final_values["termination_flag_1"] == "Primary has depleted central carbon"
    )
    unstable_mask_rate = (
        grid.final_values["termination_flag_1"]
        == "Reached maximum mass transfer rate: 1d-1"
    ) | (
        grid.final_values["termination_flag_1"]
        == "Reached maximum mass transfer rate: Exceeded photon trapping radius"
    )
    unstable_mask_L2 = (
        (
            grid.final_values["termination_flag_1"]
            == "overflow from L2 (D_L2) distance for q(=Macc/Mdon)>1, donor is star 1"
        )
        | (
            grid.final_values["termination_flag_1"]
            == "overflow from L2 (R_L2) surface for q(=Macc/Mdon)<1, donor is star 1"
        )
        | (
            grid.final_values["termination_flag_1"]
            == "overflow from L2 (R_L2) surface for q(=Macc/Mdon)>1, donor is star 1"
        )
    )

    # Calculate x and y axes
    x_axis = grid.initial_values["star_2_mass"] / grid.initial_values["star_1_mass"]
    y_axis = np.log10(grid.initial_values["period_days"])

    # Plot unstable mass transfer cases
    ax.scatter(
        x_axis[donor_mass_mask & unstable_mask_rate],
        y_axis[donor_mass_mask & unstable_mask_rate],
        color="none",
        s=MARKER_SIZE - 1,
        lw=0.9,
        edgecolor=marker_colours["Mdot"],
        marker="D",
        label="$\dot{M}_\mathrm{max}$",
        zorder=10,
    )

    ax.scatter(
        x_axis[donor_mass_mask & unstable_mask_L2],
        y_axis[donor_mass_mask & unstable_mask_L2],
        color="none",
        s=MARKER_SIZE - 1,
        lw=0.9,
        edgecolor=marker_colours["L2_overflow"],
        marker="D",
        label="$L_2$ overflow",
        zorder=10,
    )

    # Plot SMT systems with merger analysis
    S1_mass_f = grid.final_values["S1_SN_MODEL_v2_01_mass"][donor_mass_mask & smt_mask]
    S2_mass_f = grid.final_values["star_2_mass"][donor_mass_mask & smt_mask]
    separation_f = grid.final_values["binary_separation"][donor_mass_mask & smt_mask]

    if len(S1_mass_f) != 0:
        PISN_mask = S1_mass_f == 0
        if np.any(PISN_mask):
            ax.scatter(
                x_axis[donor_mass_mask & smt_mask][PISN_mask],
                y_axis[donor_mass_mask & smt_mask][PISN_mask],
                color="red",
                marker="*",
                s=10,
                label="PISN",
                edgecolors="none",
            )
            S1_mass_f = S1_mass_f[~PISN_mask]
            S2_mass_f = S2_mass_f[~PISN_mask]
            separation_f = separation_f[~PISN_mask]

        merger_times = inspiral_timescale_from_separation(
            S1_mass_f, S2_mass_f, separation_f, np.zeros(len(S1_mass_f))
        )
        no_merge = merger_times * 1e6 > HUBBLE_TIME

        # Create combined mask for non-PISN systems
        x_smt = x_axis[donor_mass_mask & smt_mask][~PISN_mask]
        y_smt = y_axis[donor_mass_mask & smt_mask][~PISN_mask]

        ax.scatter(
            x_smt[no_merge],
            y_smt[no_merge],
            color="none",
            marker="s",
            s=MARKER_SIZE + 1,
            label="SMT ($t_\mathrm{merge} > t_\mathrm{Hubble}$)",
            lw=1,
            edgecolor=marker_colours["SMT_no_merge"],
            zorder=10,
        )
        ax.scatter(
            x_smt[~no_merge],
            y_smt[~no_merge],
            s=MARKER_SIZE + 1,
            color="none",
            marker="s",
            label="SMT ($t_\mathrm{merge} \leq t_\mathrm{Hubble}$)",
            lw=1,
            edgecolor=marker_colours["SMT_merger"],
            zorder=10,
        )

    # Set plot properties
    ax.set_xscale("log")
    ax.set_xlim(0.05, 4)
    ax.set_ylim(-0.5, 3.5)

    return ax


def load_grid_data(path, metallicity):
    """
    Load the CO-HMS RLO grid data.

    Parameters
    ----------
    path : str or Path
        Path to the grid directory.
    metallicity : float
        Metallicity value.

    Returns
    -------
    tuple
        grid, m1_initial, m2_initial, p_initial
    """
    print("  Loading CO-HMS RLO grid...")
    str_met = convert_metallicity_to_string(metallicity)
    CO_HMS_RLO_file = Path(path) / "CO-HMS_RLO" / f"{str_met}_Zsun.h5"
    grid = PSyGrid(str(CO_HMS_RLO_file))
    print(f"    ✓ Loaded grid from {Path(path).name}")

    print("  Parsing initial conditions from grid...")
    m1_initial = np.zeros(len(grid.MESA_dirs))
    p_initial = np.zeros(len(grid.MESA_dirs))
    m2_initial = np.zeros(len(grid.MESA_dirs))

    for j, i in enumerate(grid.MESA_dirs):
        m2_initial[j] = float(str(i).split("_m2_")[1].split("_")[0])
        m1_initial[j] = float(str(i).split("_m1_")[1].split("_")[0])
        p_initial[j] = float(str(i).split("_days_")[1].split("_")[0])
    print(f"    ✓ Parsed initial conditions")

    return grid, m1_initial, m2_initial, p_initial


def create_figure(donor_mass, grid_paths, grid_labels, metallicity, output_folder):
    """
    Create and save the donor mass analysis figure for different accretion grids.

    Parameters
    ----------
    donor_mass : float
        Donor mass to analyze.
    grid_paths : list of str
        List of paths to the grid directories.
    grid_labels : list of str
        Labels for each grid.
    metallicity : float
        Metallicity value.
    output_folder : Path
        Output directory for the figure.
    """

    print(f"\nAnalyzing donor mass: {donor_mass} M☉ across {len(grid_paths)} grids")

    # Set up colormaps
    hist2d_colour = cmap.Colormap("crameri:grayc").to_mpl()
    marker_colours = {
        "SMT_no_merge": tol_vibrant[0],
        "SMT_merger": tol_vibrant[1],
        "Mdot": tol_vibrant[4],
        "L2_overflow": tol_vibrant[3],
    }

    fig, axes = plt.subplots(1, 3, figsize=(3.38 * 2, 2.535 * 0.7))
    plt.subplots_adjust(wspace=0.03, hspace=0.1)

    print("  Plotting grid analysis...")
    for grid_path, grid_label, ax in zip(grid_paths, grid_labels, axes):
        print(f"\n  Processing grid: {grid_label}")
        grid, m1_initial, m2_initial, p_initial = load_grid_data(grid_path, metallicity)
        ax = plot_donor_mass_analysis(
            donor_mass, ax, grid, m1_initial, metallicity, hist2d_colour, marker_colours
        )
        ax.set_title(grid_label)
        ax.set_xlabel("$M_\mathrm{acc}/M_\mathrm{donor}$")

    axes[0].set_ylabel("$\log_{10} P$ [days]")
    axes[1].set_yticklabels([])

    axes[-1].yaxis.set_ticks_position("right")
    axes[-1].set_ylabel("$\log_{10} P$ [days]")
    axes[-1].yaxis.set_label_position("right")
    axes[-1].tick_params(axis="y", which="both", left=True, right=True)

    # Save figure
    str_met = convert_metallicity_to_string(metallicity)

    # Create subdirectories for png and pdf
    png_folder = output_folder / "png"
    pdf_folder = output_folder / "pdf"
    png_folder.mkdir(parents=True, exist_ok=True)
    pdf_folder.mkdir(parents=True, exist_ok=True)

    output_file_png = (
        png_folder / f"1e-2Zun_COHMS_RLO_accretion_comparison_M{int(donor_mass)}.png"
    )
    output_file_pdf = (
        pdf_folder / f"1e-2Zun_COHMS_RLO_accretion_comparison_M{int(donor_mass)}.pdf"
    )

    # Get handles and labels from the first plot
    handles, labels = axes[0].get_legend_handles_labels()

    axes[1].legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, -0.23),
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    print(f"\n  Saving figure to: {output_file_png}")
    plt.savefig(output_file_png, bbox_inches="tight", dpi=DPI)
    plt.savefig(output_file_pdf, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  ✓ Figure saved successfully!")


# Main execution
print("\n" + "=" * 70)
print("CO-HMS Grid Slice Analysis - Accretion Efficiency Comparison")
print("=" * 70)

# Set up grid paths for different accretion efficiencies
grid_paths = [Path(grids_path) / grid_type / "POSYDON_data" for grid_type in grid_types]
grid_labels = ["Eddington-limited", "GRMHD", "Conservative"]

output_folder = Path("/home/users/b/briel/scratch/high_mass_physics/figures/")

print(f"\nMetallicity: {METALLICITY}")
print(f"Donor mass: {DONOR_MASS} M☉")
print(f"Accretion grids: {grid_types}")

# Load grid data and create figure
print("\n" + "-" * 70)
print("Generating figure...")
print("-" * 70)
create_figure(DONOR_MASS, grid_paths, grid_labels, METALLICITY, output_folder)

print("\n" + "=" * 70)
print("✓ Analysis complete!")
print("=" * 70 + "\n")
