import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from cmap import Colormap

from scipy.stats import gaussian_kde

from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume

plt.style.use(str(Path(PATH_TO_POSYDON) / 'posydon' / 'visualization' / 'posydon.mplstyle'))

MASS_CUTOFF = 39.76734837
# based on BGP bin

# load BGP data from Anarya (send on slack)
data_file = "../data/hm_dists_gwtc5_40.h5"

with h5py.File(data_file, "r") as hf:
    m1bins = hf['1D']['mass1'][:]
    pdf_m1 = hf['1D']['p_mass1'][:]

Rp_m1=np.array(pdf_m1) * (1+0.2)**2.7
Rpm_5 = np.percentile(Rp_m1,q=5,axis=0)
Rpm_95 = np.percentile(Rp_m1,q=95,axis=0)
R_pm_med = np.percentile(Rp_m1,q=50,axis=0)


# define colourmaps
cm = Colormap('tol:vibrant')
colours = cm([0.1,  0.5, 0.8])

# define bins
mass_bins = np.logspace(np.log10(40), np.log10(210), 51)

# setup figure
fig, axes = plt.subplots(1,3, figsize=(3.38*2, 2.535*0.8))

# plot BGP distribution
for ax in axes:
    ax.fill_between(m1bins,Rpm_5,Rpm_95,alpha=0.3,color="gray", label = "BGP")


# Define the data directory and folder types
data_dir = "../data/main_figure/"
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']
SFH_type = 'IllustrisTNG'

ax = axes[0]

def get_histogram_data(co_contact_file):
    
    data = Rates(co_contact_file, 'BBH', SFH_type)

    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > MASS_CUTOFF
    z_max = 0.25
    z_min = 0.15
    z_event_mask = (data.z_events >= z_min) & (data.z_events <= z_max)
    volume = get_shell_comoving_volume(z_min, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    S1_mass = np.where(
        (filtered_population['S1_mass'].to_numpy() 
            >= filtered_population['S2_mass'].to_numpy()),
                       filtered_population['S1_mass'].to_numpy(),
                       filtered_population['S2_mass'].to_numpy())
    
    print(co_contact_file)
    print('max:', np.max(S1_mass[S1_mass < 110]))
    print('rate', np.nansum(weights.to_numpy())/volume)
    # get histogram
    h, _ = np.histogram(S1_mass,
                    bins=mass_bins,
                    weights=np.nansum(weights, axis=1)/volume,)

    return h/np.diff(mass_bins)

# no kick population
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")

    co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
    h = get_histogram_data(co_contact_file)
    # set label
    label = folder_type
    if folder_type == 'conservative':
        label = 'Conservative'

    ax.step(mass_bins[:-1],
             h,
             lw=2,
             label=label,
             color=colours[i],
             where='post')


# low kicked populations
co_contact_file = '../data/' + 'kicks_Eddington-limited/low_kick.h5'
h = get_histogram_data(co_contact_file)
axes[1].step(mass_bins[:-1],
             h,
             lw=2,
             label='Eddington-limited',
             color=colours[0],
             where='post')

co_contact_file = '../data/' + 'kick_figure_GRMHD/low_kick.h5'
h = get_histogram_data(co_contact_file)
axes[1].step(mass_bins[:-1],
             h,
             lw=2,
             label='GRRMHD',
             color=colours[1],
             where='post')

co_contact_file = '../data/' + 'kicks_conservative/low_kick.h5'
h = get_histogram_data(co_contact_file)
axes[1].step(mass_bins[:-1],
             h,
             lw=2,
             label='Conservative',
             color=colours[2],
             where='post')

axes[1].set_title('Low kick')

# normal kick population

co_contact_file = '../data/' + 'kicks_Eddington-limited/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(mass_bins[:-1],
             h,
             lw=2,
             label='Eddington-limited',
             color=colours[0],
             where='post')

co_contact_file = '../data/' + 'kick_figure_GRMHD/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(mass_bins[:-1],
             h,
             lw=2,
             label='GRRMHD',
             color=colours[1],
             where='post')

co_contact_file = '../data/' + 'kicks_conservative/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(mass_bins[:-1],
             h,
             lw=2,
             label='Conservative',
             color=colours[2],
             where='post')

axes[2].set_title('Normal kick')

axes[0].set_title('No kick')

for ax in axes:
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([40, 50, 60, 80,  100, 150, 200])
    ax.set_xticklabels(['', 50, 60, 80,  100, 150, 200])
    ax.set_xticks([70, 90, 110, 120, 130, 140, 160, 170, 180, 190], minor=True)
    ax.set_xticklabels([], minor=True)

    ax.set_xlim(MASS_CUTOFF, 200)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel(r'$\mathrm{M}_1 \, [\mathrm{M}_{\odot}]$')


axes[1].set_yticks([])
axes[2].set_yticks([])

axes[0].set_ylabel('$d\mathcal{R}/d\mathrm{M}_1$ $[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,\mathrm{M}_{\odot}^{-1}]$')


axes[0].legend(bbox_to_anchor=(0.1, -0.25), loc='upper left', ncol=4)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

output_dir = "../figures"
plt.savefig(f'{output_dir}/png/intrinsic_m1_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/pdf/intrinsic_m1_distribution.pdf', bbox_inches='tight')