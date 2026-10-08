# Evolution of one CO-HMS RLO grid point for the three BH accretion efficiencies:
# donor (M_1) and BH (M_2) mass, and the orbital period relative to the initial
# period of the grid point, against age, for a short-period grid point with q ~ 0.9.
# Eddington-limited is the standard POSYDON grid in PATH_TO_POSYDON_DATA,
# GRRMHD ('moderate') and conservative are in data/other_grids.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from matplotlib.lines import Line2D
from posydon.config import PATH_TO_POSYDON, PATH_TO_POSYDON_DATA
from posydon.grids.psygrid import PSyGrid
from posydon.utils.common_functions import (convert_metallicity_to_string,
                                            inspiral_timescale_from_separation)
from posydon.utils.constants import Zsun

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

METALLICITY = 0.01
DONOR_MASS = 33.8767  # Msun
HUBBLE_TIME = 13.8e9  # years

# grid point BH mass and initial period, as in the grid directory names
BH_MASS = 30.0000  # Msun, q = 0.89: closest to q = 0.9 at this period
PERIOD = 1.7433  # days, grid period closest to 10^0.2 days

str_met = convert_metallicity_to_string(METALLICITY)
grid_files = {
    "Eddington-limited": Path(PATH_TO_POSYDON_DATA) / "CO-HMS_RLO" / f"{str_met}_Zsun.h5",
    "GRRMHD": Path("../../data/other_grids") / f"{str_met}_Zsun_moderate.h5",
    "Conservative": Path("../../data/other_grids") / f"{str_met}_Zsun_conservative.h5",
}

# same colours as the accretion models in the other figures
cm = Colormap("tol:vibrant")
colours = cm([0.1, 0.5, 0.8])


def get_model(grid, bh_mass, period):
    """Return the run and final values of the grid point (DONOR_MASS, bh_mass, period)."""
    dirs = [str(d) for d in grid.MESA_dirs]
    m1 = np.array([float(d.split("_m1_")[1].split("_")[0]) for d in dirs])
    m2 = np.array([float(d.split("_m2_")[1].split("_")[0]) for d in dirs])
    p = np.array([float(d.split("_days_")[1].split("_")[0]) for d in dirs])
    # the conservative grid also contains some models at another metallicity
    match = (
        np.isclose(m1, DONOR_MASS)
        & np.isclose(m2, bh_mass)
        & np.isclose(p, period)
        & np.isclose(grid.initial_values["Z"], METALLICITY * Zsun)
    )
    index = np.where(match)[0]
    if len(index) != 1:
        raise ValueError(f"Found {len(index)} models for the grid point")
    print(f"  run {index[0]}, {grid.final_values['termination_flag_1'][index[0]]}")
    return grid[index[0]], grid.final_values[index[0]]


def merges_in_hubble_time(final_values):
    """Merger time of the BBH after the donor collapses, as in grid_slices.py."""
    S1_mass_f = final_values["S1_SN_MODEL_v2_01_mass"]
    if S1_mass_f == 0:  # PISN, no BH
        return False, np.inf
    merger_time = inspiral_timescale_from_separation(
        S1_mass_f, final_values["star_2_mass"], final_values["binary_separation"], 0.0
    )  # Myr
    return merger_time * 1e6 <= HUBBLE_TIME, merger_time


def create_figure(grids, bh_mass, period):
    fig, axes = plt.subplots(2, 1, figsize=(3.38, 2.535 * 1.4), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    for (label, grid), colour in zip(grids.items(), colours):
        print(f"{label}, M_2 = {bh_mass} Msun, P_i = {period} d")
        run, final_values = get_model(grid, bh_mass, period)
        history = run.binary_history
        age = history["age"] / 1e6  # Myr
        merges, merger_time = merges_in_hubble_time(final_values)
        print(f"  t_merge = {merger_time / 1e3:.3g} Gyr -> "
              f"{'merges' if merges else 'does not merge'} within the Hubble time")

        axes[0].plot(age, history["star_1_mass"], color=colour, lw=1.5)
        axes[0].plot(age, history["star_2_mass"], color=colour, lw=1.5, ls="--")
        # relative to the initial period of the grid point, not the first history entry
        axes[1].plot(age, history["period_days"] / period, color=colour, lw=1.5)

        if merges:
            marker = dict(marker="s", color="black", ms=4, zorder=10, ls="none")
            axes[0].plot(age[-1], history["star_1_mass"][-1], **marker)
            axes[0].plot(age[-1], history["star_2_mass"][-1], **marker)
            axes[1].plot(age[-1], history["period_days"][-1] / period, **marker)

    axes[0].set_ylabel(r"$M\,[\mathrm{M}_\odot]$")
    axes[1].set_ylabel(r"$P/P_\mathrm{i}$")
    axes[1].set_xlabel("Age [Myr]")
    axes[1].set_xlim(0, None)

    model_handles = [
        Line2D([], [], color=colour, lw=1.5, label=label)
        for label, colour in zip(grids, colours)
    ]
    star_handles = [
        Line2D([], [], color="gray", lw=1.5, label=r"$M_1$ (donor)"),
        Line2D([], [], color="gray", lw=1.5, ls="--", label=r"$M_2$ (BH)"),
    ]
    merge_handle = Line2D([], [], marker="s", color="black", ms=4, ls="none",
                          label=r"$t_\mathrm{merge} \leq t_\mathrm{Hubble}$")
    # both legends in the bottom panel, where there is space
    model_legend = axes[1].legend(handles=model_handles + [merge_handle], loc="upper left")
    axes[1].add_artist(model_legend)
    axes[1].legend(handles=star_handles, loc="lower right")
    axes[0].set_title(
        rf"$M_1={DONOR_MASS:.1f}\,\mathrm{{M}}_\odot$, "
        rf"$q={bh_mass / DONOR_MASS:.2f}$, "
        rf"$P_\mathrm{{i}}={period:.3g}\,$d"
    )

    output_dir = "../../figures"
    name = (f"{str_met}_Zsun_COHMS_RLO_model_evolution_M{int(DONOR_MASS)}"
            f"_q{bh_mass / DONOR_MASS:.2f}_P{period:.3g}d")
    plt.savefig(f"{output_dir}/png/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/pdf/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


grids = {label: PSyGrid(str(grid_file)) for label, grid_file in grid_files.items()}
create_figure(grids, BH_MASS, PERIOD)
