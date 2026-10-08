# Orbital response to stable mass transfer:
#   d ln a / d ln M_donor = 2 (f - 1),
#   f(beta, q) = beta / q + (1 / q + 1 / 2) (1 - beta) / (q + 1),
# with q = M_acc / M_donor and beta the fraction of the transferred mass accreted
# by the BH (the rest is lost from the vicinity of the BH, isotropic re-emission).
# The orbit shrinks for f > 1 and widens for f < 1. Over-plotted are the six
# CO-HMS RLO models of grid_model_evolution.py and grid_model_evolution_short_period.py,
# using MESA's instantaneous accreted fraction (xfer_fraction) as beta.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from matplotlib.lines import Line2D
from posydon.config import PATH_TO_POSYDON, PATH_TO_POSYDON_DATA
from posydon.grids.psygrid import PSyGrid
from posydon.utils.common_functions import convert_metallicity_to_string
from posydon.utils.constants import Zsun

plt.style.use(
    str(Path(PATH_TO_POSYDON) / "posydon" / "visualization" / "posydon.mplstyle")
)

METALLICITY = 0.01
DONOR_MASS = 33.8767  # Msun
# grid points (BH mass [Msun], initial period [days], line style)
MODELS = [
    (10.2485, 14.874, "-"),  # q = 0.30, grid_model_evolution.py
    (30.0000, 1.7433, "--"),  # q = 0.89, grid_model_evolution_short_period.py
]
MIN_LG_MT_RATE = -12  # only plot steps with ongoing mass transfer

str_met = convert_metallicity_to_string(METALLICITY)
grid_files = {
    "Eddington-limited": Path(PATH_TO_POSYDON_DATA) / "CO-HMS_RLO" / f"{str_met}_Zsun.h5",
    "GRRMHD": Path("../../data/other_grids") / f"{str_met}_Zsun_moderate.h5",
    "Conservative": Path("../../data/other_grids") / f"{str_met}_Zsun_conservative.h5",
}

# same colours as the accretion models in the other figures
cm = Colormap("tol:vibrant")
colours = cm([0.1, 0.5, 0.8])


def f_orbit(beta, q):
    return beta / q + (1 / q + 0.5) * (1 - beta) / (q + 1)


def get_history(grid, bh_mass, period):
    """Return the binary history of the grid point (DONOR_MASS, bh_mass, period)."""
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
    return grid[index[0]].binary_history


fig, ax = plt.subplots(1, 1, figsize=(3.38, 2.535))

q_grid = np.logspace(np.log10(0.2), np.log10(4), 500)
ax.plot(q_grid, f_orbit(1, q_grid), color="gray", lw=1.5, label=r"$\beta=1$ (conservative)")
ax.plot(q_grid, f_orbit(0, q_grid), color="gray", lw=1.5, ls=":",
        label=r"$\beta=0$ (isotropic re-emission)")
ax.axhline(1, color="black", ls="--", lw=1)
ax.text(3.8, 1.08, "shrinking", ha="right", va="bottom")
ax.text(3.8, 0.92, "widening", ha="right", va="top")

for (label, grid_file), colour in zip(grid_files.items(), colours):
    grid = PSyGrid(str(grid_file))
    for bh_mass, period, ls in MODELS:
        history = get_history(grid, bh_mass, period)
        mt = history["lg_mtransfer_rate"] > MIN_LG_MT_RATE
        q = history["star_2_mass"][mt] / history["star_1_mass"][mt]
        f = f_orbit(history["xfer_fraction"][mt], q)
        ax.plot(q, f, color=colour, lw=1.5, ls=ls)
        ax.plot(q[0], f[0], marker="s", color="black", ms=3, ls="none", zorder=10)

ax.set_xscale("log")
ax.set_xlim(q_grid[0], q_grid[-1])
ax.set_xticks([0.2, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"])
ax.set_ylim(0, None)
ax.set_xticks([], minor=True)
ax.set_xlabel(r"$q = M_\mathrm{acc}/M_\mathrm{donor}$")
ax.set_ylabel(r"$f(\beta, q)$")

theory_legend = ax.legend(loc="upper right", fontsize=7)
ax.add_artist(theory_legend)
model_handles = [
    Line2D([], [], color=colour, lw=1.5, label=label)
    for label, colour in zip(grid_files, colours)
] + [
    Line2D([], [], color="black", lw=1.5, ls="-", label=r"$q_\mathrm{i}=0.30$, $P_\mathrm{i}=14.9\,$d"),
    Line2D([], [], color="black", lw=1.5, ls="--", label=r"$q_\mathrm{i}=0.89$, $P_\mathrm{i}=1.74\,$d"),
]
ax.legend(handles=model_handles, loc="center right", fontsize=7)

output_dir = "../../figures"
plt.savefig(f"{output_dir}/png/orbital_response_f_beta_q.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/pdf/orbital_response_f_beta_q.pdf", bbox_inches="tight")
