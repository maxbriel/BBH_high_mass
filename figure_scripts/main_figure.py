import numpy as np
import matplotlib.pyplot as plt
import os
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume

# Set up the figure with 3 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(10, 5))

# Define the data directory and folder types
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/main_figure/'
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']

mass_bins = np.linspace(10, 100, 51)
q_bins = np.linspace(0, 1, 51)
chi_bins = np.linspace(-0.2, 1, 51)

# Loop through each folder type
for i, folder_type in enumerate(folder_types):
    
    print(f"Processing folder: {folder_type}")
    folder_path = os.path.join(data_dir, folder_type)
    
    # Find CO_contact files in the folder
    co_contact_file = os.path.join(folder_path, 'CO_contact.h5')

    # Load the data
    data = Rates(co_contact_file, 'BBH', 'IllustrisTNG')
    print("Rates data loaded.")

    # Apply selection: max(S1_mass, S2_mass) > 40
    max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
    mask = max_mass > 40
    
    z_max = 2
    z_event_mask = data.z_events <= z_max
    volume = get_shell_comoving_volume(0, z_max)
    
    # Filter the population data
    filtered_population = data.population[mask]
    chirp_mass = filtered_population['chirp_mass'].to_numpy()
    mass_ratio = filtered_population['mass_ratio'].to_numpy()
    chi_eff = filtered_population['chi_eff'].to_numpy()
    
    # Get weights
    weights = data.weights[z_event_mask][mask].to_numpy()
    
    # Top row: Chirp mass vs chi_eff
    H1, xedges1, yedges1 = np.histogram2d(chirp_mass,
                                          chi_eff,
                                          bins=(mass_bins, chi_bins),
                                          weights=np.nansum(weights, axis=1)/volume)
    H1 /= np.diff(mass_bins)[:, None] / np.diff(chi_bins)
    H1_sorted = np.sort(H1.flatten())[::-1]
    H1_cumsum = H1_sorted.cumsum()
    H1_total = H1_cumsum[-1]
    # Find contour levels for 1, 2, 3 sigma (containing 68.3%, 95.4%, 99.7% of the total)
    level1_1 = H1_sorted[np.searchsorted(H1_cumsum, 0.683 * H1_total)]
    level1_2 = H1_sorted[np.searchsorted(H1_cumsum, 0.954 * H1_total)]
    level1_3 = H1_sorted[np.searchsorted(H1_cumsum, 0.997 * H1_total)]
    levels1 = [level1_3, level1_2, level1_1]
    
    axes[0, i].imshow(H1.T, origin='lower', extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], 
                        cmap='gray_r', aspect='auto', interpolation='gaussian')
    #axes[0, i].contour(H1.T, levels=levels1, colors=['black', 'dimgray', 'lightgray'],
    #                   linewidths=[2.0, 1.4, 0.9],
    #                   extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
                       
                       
    axes[0, i].set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]')
    axes[0, i].set_ylabel(r'$\chi_{\rm eff}$')
    axes[0, i].set_title(f'{folder_type}')
    axes[0, i].grid(True, alpha=0.3)
    #axes[0, i].text(0.95, 0.95, f'N={total_sum:.1f}', transform=axes[0, i].transAxes, 
    #                ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom row: Mass ratio vs chi_eff
    # H2, xedges2, yedges2 = np.histogram2d(chirp_mass, mass_ratio, bins=(mass_bins, q_bins), weights=weights)
    # H2_sorted = np.sort(H2.flatten())[::-1]
    # H2_cumsum = H2_sorted.cumsum()
    # H2_total = H2_cumsum[-1]
    # level2_1 = H2_sorted[np.searchsorted(H2_cumsum, 0.683 * H2_total)]
    # level2_2 = H2_sorted[np.searchsorted(H2_cumsum, 0.954 * H2_total)]
    # level2_3 = H2_sorted[np.searchsorted(H2_cumsum, 0.997 * H2_total)]
    # levels2 = [level2_3, level2_2, level2_1]
    
    # axes[1, i].imshow(H2.T, origin='lower', extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], 
    #                     cmap='gray_r', aspect='auto', interpolation='gaussian')
    # axes[1, i].contour(H2.T, levels=levels2, colors=['black', 'dimgray', 'lightgray'],
    #                    linewidths=[2.0, 1.4, 0.9],
    #                    extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
    # axes[1, i].set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]')
    # axes[1, i].set_ylabel(r'$q$ (mass ratio)')
    # axes[1, i].grid(True, alpha=0.3)
    break

# Save the figure
output_path = '/home/users/b/briel/scratch/high_mass_physics/figures/main_figure.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

# Show the figure
plt.show()
