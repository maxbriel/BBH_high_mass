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

# load BGP data from Anarya (send on slack)
data_file = "../data/hm_dists.h5"

with h5py.File(data_file, "r") as hf:
    chieff_bins_model = hf['1D']['chi_eff'][:]
    pdf_chieff = hf['1D']['p_chi_eff'][:]
    
    

Rp_chieff=np.array(pdf_chieff) / np.trapz(np.array(pdf_chieff),chieff_bins_model,axis=1)[:,None]
Rpm_5 = np.percentile(Rp_chieff,q=5,axis=0)
Rpm_95 = np.percentile(Rp_chieff,q=95,axis=0)
R_pm_med = np.percentile(Rp_chieff,q=50,axis=0)


# define colourmaps
cm = Colormap('tol:vibrant')
colours = cm([0.1,  0.5, 0.8])

# define bins
chieff_bins = np.linspace(-1, 1.1, 36)

# setup figure
fig, axes = plt.subplots(1,3, figsize=(3.38*2, 2.535*0.8))

# plot BGP distribution
for ax in axes:
    ax.fill_between(chieff_bins_model,Rpm_5,Rpm_95,alpha=0.3,color="gray", label = "BGP")


# Define the data directory and folder types
data_dir = "../data/main_figure/"
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']
SFH_type = 'IllustrisTNG'

ax = axes[0]

def get_histogram_data(co_contact_file):
    
    data = Rates(co_contact_file, 'BBH', SFH_type)
    print("Rates data loaded.")

    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 44.2
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chief_eff = filtered_population['chi_eff'].to_numpy()

    # get histogram
    h, _ = np.histogram(chief_eff,
                    bins=chieff_bins,
                    weights=np.nansum(weights, axis=1)/volume,)
    
    one_weights = np.sum(weights, axis=1)
    print(f'% negative chi_eff: {np.sum(one_weights[chief_eff < 0])/np.sum(one_weights) * 100:.2f}%')

    return h/np.diff(chieff_bins)

# no kick population
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")

    co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
    h = get_histogram_data(co_contact_file)
    # set label
    label = folder_type
    if folder_type == 'conservative':
        label = 'Conservative'

    ax.step(chieff_bins[:-1],
             h,
             lw=2,
             label=label,
             color=colours[i],
             where='post')  


# low kicked populations
co_contact_file = '../data/' + 'kicks_Eddington-limited/low_kick.h5'
h = get_histogram_data(co_contact_file )
axes[1].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='Eddington-limited',
             color=colours[0],
             where='post')

co_contact_file = '../data/' + 'kick_figure_GRMHD/low_kick.h5'
h = get_histogram_data(co_contact_file)
axes[1].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='GRMHD',
             color=colours[1],
             where='post')

co_contact_file = '../data/' + 'kicks_conservative/low_kick.h5'
h = get_histogram_data(co_contact_file)
axes[1].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='Conservative',
             color=colours[2],
             where='post')
axes[1].set_title('Low kick')

# normal kick population

co_contact_file = '../data/' + 'kicks_Eddington-limited/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='Eddington-limited',
             color=colours[0],
             where='post')

co_contact_file = '../data/' + 'kick_figure_GRMHD/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='GRMHD',
             color=colours[1],
             where='post')

co_contact_file = '../data/' + 'kicks_conservative/normal_kick.h5'
h = get_histogram_data(co_contact_file)
axes[2].step(chieff_bins[:-1],
             h / np.sum(h * np.diff(chieff_bins)),
             lw=2,
             label='Conservative',
             color=colours[2],
             where='post')

axes[2].set_title('Normal kick')

ax.set_title('No kick')

for ax in axes:
    ax.set_yscale('log')
    ax.set_xlim(-0.999,0.999)
    ax.set_ylim(1e-3, 2e2)
    ax.set_xlabel(r'$\chi_\mathrm{eff}$')
#     ax.set_xticks([50, 60, 80,  100, 150, 200])
#     ax.set_xticklabels([50, 60,  80,  100, 150, 200])
#     ax.set_xticks([70, 90, 110, 120, 130, 140, 160, 170, 180, 190], minor=True)
#     ax.set_xticklabels([], minor=True)

#     ax.set_xlim(44.2, 200)
#     ax.set_ylim(1e-4, 2)
    #ax.set_xlabel(r'$M_1 \, [M_{\odot}]$')


axes[1].set_yticks([])
axes[2].set_yticks([])

axes[0].set_ylabel('PDF')


axes[0].legend(bbox_to_anchor=(0.1, -0.25), loc='upper left', ncol=4)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

output_dir = "../figures"
plt.savefig(f'{output_dir}/png/intrinsic_chi_eff_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/pdf/intrinsic_chi_eff_distribution.pdf', bbox_inches='tight')