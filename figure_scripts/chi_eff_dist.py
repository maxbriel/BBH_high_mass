import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from cmap import Colormap

from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume

plt.style.use(str(Path(PATH_TO_POSYDON) / 'posydon' / 'visualization' / 'posydon.mplstyle'))

# load spline
spline_file = '/home/users/b/briel/scratch/high_mass_physics/data/spline_data/ppd_pdfs_mean_m1qzchieff_mmin40_collector_only_10000w_10000s_rng129_zspline.h5'

file = h5py.File(spline_file, 'r')

chi_eff_pdf = file['chi_eff_pdf'][:]
chi_grid = file['chi_grid'][:]

# Calculate statistics across iterations
mean_pdf = np.mean(chi_eff_pdf, axis=0)
std_pdf = np.std(chi_eff_pdf, axis=0)
# Use 95% confidence interval (2.5th to 97.5th percentile)
percentile_2_5 = np.percentile(chi_eff_pdf, 2.5, axis=0)
percentile_97_5 = np.percentile(chi_eff_pdf, 97.5, axis=0)

# Plot mean with filled uncertainty region
fig, ax = plt.subplots(1,1, figsize=(3.38, 2.535))

plt.plot(chi_grid, mean_pdf, color='black', linewidth=2, label='Spline')
plt.fill_between(chi_grid, percentile_2_5, percentile_97_5, 
                 color='black',
                 alpha=0.2,
                 edgecolor='none')


# Define the data directory and folder types
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/main_figure/'
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']
SFH_type = 'IllustrisTNG'


title_mapping = {
    'no_kick': 'No kicks',
    'low_kick': 'Low kicks',
    'normal_kick': 'Normal kicks'
}

cm = Colormap('tol:vibrant')

colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

chi_eff_bins = np.linspace(-0.6, 1, 51)

for i, folder_type in enumerate(folder_types[:-1]):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
    #co_contact_file = os.path.join(folder_path, 'CO_contact.h5')
    data = Rates(co_contact_file, 'BBH', SFH_type)
    print("Rates data loaded.")
    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chirp_mass = filtered_population['chirp_mass'].to_numpy()
    mass_ratio = filtered_population['mass_ratio'].to_numpy()
    chi_eff = filtered_population['chi_eff'].to_numpy()
    
    H0, xedges= np.histogram(chi_eff,
                             bins=chi_eff_bins, 
                             weights=np.nansum(weights, axis=1)/volume,
                             density=True,
                             )
    
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(xedges[:-1], H0, where='post', lw=1, label=label, color=colours[i])
    

data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/figure_2/'
folder_types = ['no_kick', 'low_kick', 'normal_kick']
SFH_type = 'IllustrisTNG'
for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(data_dir, folder_type+'.h5',)
    #co_contact_file = os.path.join(folder_path, 'CO_contact.h5')
    data = Rates(co_contact_file, 'BBH', SFH_type)
    print("Rates data loaded.")
    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 40
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    weights = data.weights[z_event_mask][mask]
    filtered_population = data.population[mask]
    chirp_mass = filtered_population['chirp_mass'].to_numpy()
    mass_ratio = filtered_population['mass_ratio'].to_numpy()
    chi_eff = filtered_population['chi_eff'].to_numpy()
    
    H0, xedges= np.histogram(chi_eff,
                             bins=chi_eff_bins, 
                             weights=np.nansum(weights, axis=1)/volume,
                             density=True,
                             )
    
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(xedges[:-1], H0, where='post', lw=1, label=label, color=colours[i+3])

plt.ylim(0, 10)
plt.xlim(-0.6, 1.0)
plt.xlabel(r'$\chi_{\mathrm{eff}}$')
plt.ylabel('PDF')
plt.legend()

output_dir = '/home/users/b/briel/scratch/high_mass_physics/figures'
plt.savefig(f'{output_dir}/intrinsic_chi_eff.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/intrinsic_chi_eff.pdf', bbox_inches='tight')
