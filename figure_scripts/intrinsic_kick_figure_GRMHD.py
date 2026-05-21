import numpy as np

import os
import h5py
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,LogLocator,NullFormatter)
from matplotlib.colors import LogNorm

from cmap import Colormap

from scipy.interpolate import RegularGridInterpolator


from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume

from posydon.config import PATH_TO_POSYDON

plt.style.use(str(Path(PATH_TO_POSYDON) / 'posydon' / 'visualization' / 'posydon.mplstyle'))

# Load BGP data from Anarya (send on slack)
data_file = "../data/hm_dists.h5"

with h5py.File(data_file, "r") as hf:
    matrix1 = hf["2D"]["p_m1q"][()]
    mbins = hf["2D"]["mass1"][()]
    qbins = hf["2D"]["mass_ratio"][()]
    
    matrix_m1chi = hf["2D"]["p_m1chi"][()]
    mbins = hf["2D"]["mass1"][()]
    chibins = hf["2D"]["chi_eff"][()]

# Define colormaps
cm = Colormap('tol:YlOrBr')
cm_grays = Colormap('colorbrewer:Greys')


data_dir = '../data/kick_figure_GRMHD/'
folder_types = ['no_kick', 'low_kick', 'normal_kick']
SFH_type = 'IllustrisTNG'

mass_bins = np.linspace(10, 200, 31)
q_bins = np.linspace(0, 1, 31)
chi_bins = np.linspace(-1, 1, 51)

fig, axes = plt.subplots(2, 3, figsize=(3.38*2, 2.535*1.7))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

def get_histogram_data(co_contact_file):
    
    data = Rates(co_contact_file, 'BBH', SFH_type)
    print("Rates data loaded.")

    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    M1_mass = max_mass[mask]
    mass_ratio = filtered_population['mass_ratio'].to_numpy()
    chi_eff = filtered_population['chi_eff'].to_numpy()

    return M1_mass, mass_ratio, chi_eff, weights, volume


for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    
    M1_mass, mass_ratio, chi_eff, weights, volume = get_histogram_data(os.path.join(data_dir, folder_type+'.h5'))
    
    # Top row: chirp mass vs mass ratio
    H0, xedges0, yedges0 = np.histogram2d(M1_mass,
                                        mass_ratio,
                                        bins=(mass_bins, q_bins),
                                        weights=np.nansum(weights, axis=1)/volume)
    H0 /= (np.diff(mass_bins)[:, None] * np.diff(q_bins))
    H0_sorted = np.sort(H0.flatten())[::-1]
    H0_cumsum = H0_sorted.cumsum()
    H0_total = H0_cumsum[-1]
    # Find contour levels for 1, 2, 3 sigma (containing 68.3%, 95.4%, 99.7% of the total)
    level0_1 = H0_sorted[np.searchsorted(H0_cumsum, 0.683 * H0_total)]
    level0_2 = H0_sorted[np.searchsorted(H0_cumsum, 0.954 * H0_total)]
    level0_3 = H0_sorted[np.searchsorted(H0_cumsum, 0.997 * H0_total)]
    levels0 = [level0_3, level0_2, level0_1]

    axes[0, i].contour(mass_bins[1:], q_bins[1:], H0.T,
                       levels=levels0,
                       colors=cm([0.2, 0.6, 1.0]),
                       linewidths=[2.0, 2.0, 2.0])
    
    axes[0, i].pcolor(mbins, qbins, matrix1.T, norm=LogNorm(vmin=matrix1[matrix1!=0].min(), vmax=matrix1.max()), cmap=cm_grays.to_mpl())

    # Top bottom row: Chirp mass vs chi_eff
    H1, xedges1, yedges1 = np.histogram2d(M1_mass,
                                            chi_eff,
                                            bins=(mass_bins, chi_bins),
                                            weights=np.nansum(weights, axis=1)/volume)
    H1 /= (np.diff(mass_bins)[:, None] * np.diff(chi_bins))
    H1_sorted = np.sort(H1.flatten())[::-1]
    H1_cumsum = H1_sorted.cumsum()
    H1_total = H1_cumsum[-1]
    # Find contour levels for 1, 2, 3 sigma (containing 68.3%, 95.4%, 99.7% of the total)
    level1_1 = H1_sorted[np.searchsorted(H1_cumsum, 0.683 * H1_total)]
    level1_2 = H1_sorted[np.searchsorted(H1_cumsum, 0.954 * H1_total)]
    level1_3 = H1_sorted[np.searchsorted(H1_cumsum, 0.997 * H1_total)]
    levels1 = [level1_3, level1_2, level1_1]

    axes[1, i].contour(mass_bins[1:],
                       chi_bins[1:], H1.T,
                       levels=levels1,
                       colors=cm([0.2, 0.6, 1.0]),
                       linewidths=[2.0, 2.0, 2.0])
    
    axes[1, i].pcolor(mbins,
                    chibins,
                    matrix_m1chi.T,
                    norm=LogNorm(vmin=matrix_m1chi[matrix_m1chi!=0].min(),vmax=matrix_m1chi[matrix_m1chi!=0].max()),cmap=cm_grays.to_mpl())

    # add rate denisty
    rate_density = np.nansum(weights)/volume
    axes[1, i].text(0.95,
                    0.15,
                    r'$\mathcal{{R}}_{0{-}2}=$'+f'{rate_density:.1f}'+r'$\,\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$',
                    transform=axes[1, i].transAxes,
                    ha='right',
                    va='top', 
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white',
                              alpha=0.6,
                              edgecolor='grey',
                              lw=0.5,
                              )
                    )

# axes and stuff
axes[0,0].set_ylabel(r'$q=M_\mathrm{min}/M_\mathrm{max}$')
axes[1,0].set_ylabel(r'$\chi_\mathrm{eff}$')

axes[0,0].set_title('No kick')
axes[0,1].set_title('Low kick')
axes[0,2].set_title('Normal kick')

# axes and stuff
for ax in axes.flatten():
    ax.set_xscale('log')
    ax.set_xlim(44, 200)
    ax.xaxis.set_ticks([50, 60, 80, 100, 150, 200])
    ax.set_xticklabels([])

# Top row: x-axis at top
for ax in axes[0, :]:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$M_{1}$')
    ax.xaxis.set_ticks([50, 60, 80, 100, 150, 200])
    ax.set_xticklabels([50, 60, 80, 100, 150, 200])
    ax.set_ylim(0.1, 1)


# Bottom row: x-axis at bottom
for ax in axes[1, :]:
    ax.set_xlabel(r'$M_{1}$')
    ax.xaxis.set_ticks([50, 60, 80, 100, 150, 200])
    ax.set_xticklabels([50, 60, 80, 100, 150, 200])

axes[0, 0].set_ylabel(r'$q = M_2/M_1$')
axes[1, 0].set_ylabel(r'$\chi_\mathrm{eff}$')

for ax in axes[:, 1]:
    ax.set_yticklabels([])

for ax in axes[:, 2]:
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    
output_dir = '../figures'
plt.savefig(f'{output_dir}/png/intrinsic_kick_GRMHD.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_dir}/pdf/intrinsic_kick_GRMHD.pdf', bbox_inches='tight')