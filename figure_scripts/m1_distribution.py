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
data_file = "/home/users/b/briel/scratch/high_mass_physics/data/hm_dists.h5"


with h5py.File(data_file, "r") as hf:
    m1bins = hf['1D']['mass1'][:]
    pdf_m1 = hf['1D']['p_mass1'][:]

Rp_m1=np.array(pdf_m1)
Rpm_5 = np.percentile(Rp_m1,q=5,axis=0)
Rpm_95 = np.percentile(Rp_m1,q=95,axis=0)
R_pm_med = np.percentile(Rp_m1,q=50,axis=0)


# define colourmaps
cm = Colormap('tol:vibrant')
colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

# define bins
mass_bins = np.linspace(40, 200, 71)

# setup figure
fig, ax = plt.subplots(1,1, figsize=(3.38, 2.535))

# plot BGP distribution
ax.fill_between(m1bins,Rpm_5,Rpm_95,alpha=0.3,color="gray", label = "BGP")


# Define the data directory and folder types
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/main_figure/'
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']
SFH_type = 'IllustrisTNG'

title_mapping = {
    'no_kick': '',
    'low_kick': '-Low',
    'normal_kick': '-Normal',
}

for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")

    co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
    data = Rates(co_contact_file, 'BBH', SFH_type)
    print("Rates data loaded.")

    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    S1_mass = np.where(
        (filtered_population['S1_mass'].to_numpy() 
            >= filtered_population['S2_mass'].to_numpy()),
                       filtered_population['S1_mass'].to_numpy(),
                       filtered_population['S2_mass'].to_numpy())
    
    print(filtered_population['metallicity'].value_counts())
    
    # get histogram
    h, _ = np.histogram(S1_mass,
                    bins=mass_bins,
                    weights=np.nansum(weights, axis=1)/volume,)
    
    # set label
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(mass_bins[:-1],
             h/np.diff(mass_bins),
             lw=2,
             label=label,
             color=colours[i])   

ax.set_yscale('log')
plt.xlim(44.2, 190)
ax.set_ylim(1e-4, 1)
plt.xlabel(r'$M_1 \, (M_{\odot})$')
plt.ylabel('Rate density $(\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_{\odot}^{-1})$')
plt.legend(ncol=2)

output_dir = '/home/users/b/briel/scratch/high_mass_physics/figures'
plt.savefig('intrinsic_m1_distribution.png', dpi=300, bbox_inches='tight')
#plt.savefig(f'{output_dir}/intrinsic_m1_distribution.pdf', bbox_inches='tight')