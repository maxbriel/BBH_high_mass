import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from posydon.popsyn.synthetic_population import Rates

# Set up the figure with 3 rows and 3 columns
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Define the data directory and folder types
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/main_figure/'
folder_types = ['limited', 'GRMHD', 'conservative']

mass_bins = np.linspace(10, 100, 51)
q_bins = np.linspace(0, 1, 51)
chi_bins = np.linspace(-0.1, 1, 51)
redshift_bins = np.linspace(0, 10, 51)

# Loop through each folder type
for i, folder_type in enumerate(folder_types):
    folder_path = os.path.join(data_dir, folder_type)
    
    # Find CO_contact files in the folder
    co_contact_file = os.path.join(folder_path, 'CO_contact.h5')

    # Load the data
    if co_contact_file.endswith('.csv'):
        data = pd.read_csv(co_contact_file)
    elif co_contact_file.endswith('.h5'):
        data = Rates(co_contact_file, 'BBH', 'IllustrisTNG')

        # Apply selection: max(S1_mass, S2_mass) > 40
        max_mass = np.maximum(data.population['S1_mass'].values, data.population['S2_mass'].values)
        mask = max_mass > 40
        
        # Filter the population data
        filtered_population = data.population[mask].reset_index(drop=True)
        filtered_z_events = data.z_events[mask].reset_index(drop=True)
        filtered_weights = data.weights[mask].reset_index(drop=True)

        # Get z_events and flatten it
        redshift = filtered_z_events.to_numpy().flatten()
        
        # Get weights and flatten them
        weights = filtered_weights.to_numpy().flatten()
        
        # Calculate total sum for this column
        total_sum = np.sum(weights)
        
        # Repeat each row of population data to match z_events length
        n_repeats = filtered_z_events.shape[1]  # Should be 138
        
        # Calculate chirp mass: M_chirp = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        chirp_mass = np.repeat(filtered_population['chirp_mass'].values, n_repeats)
        
        mass_ratio = np.repeat(filtered_population['mass_ratio'].values, n_repeats)
        
        # Get chi_eff
        chi_eff = np.repeat(filtered_population['chi_eff'].values, n_repeats)
        
        # Top row: Chirp mass vs chi_eff
        H1, xedges1, yedges1 = np.histogram2d(chirp_mass, chi_eff, bins=(mass_bins, chi_bins), weights=weights)
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
        axes[0, i].contour(0.5*(xedges1[:-1]+xedges1[1:]), 0.5*(yedges1[:-1]+yedges1[1:]), H1.T, 
                          levels=levels1, colors=['yellow', 'orange', 'red'], linewidths=1.5)
        axes[0, i].set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]')
        axes[0, i].set_ylabel(r'$\chi_{\rm eff}$')
        axes[0, i].set_title(f'{folder_type}')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].text(0.95, 0.95, f'N={total_sum:.1f}', transform=axes[0, i].transAxes, 
                       ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom row: Mass ratio vs chi_eff
        H2, xedges2, yedges2 = np.histogram2d(chirp_mass, mass_ratio, bins=(mass_bins, q_bins), weights=weights)
        H2_sorted = np.sort(H2.flatten())[::-1]
        H2_cumsum = H2_sorted.cumsum()
        H2_total = H2_cumsum[-1]
        level2_1 = H2_sorted[np.searchsorted(H2_cumsum, 0.683 * H2_total)]
        level2_2 = H2_sorted[np.searchsorted(H2_cumsum, 0.954 * H2_total)]
        level2_3 = H2_sorted[np.searchsorted(H2_cumsum, 0.997 * H2_total)]
        levels2 = [level2_3, level2_2, level2_1]
        
        axes[1, i].imshow(H2.T, origin='lower', extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], 
                         cmap='gray_r', aspect='auto', interpolation='gaussian')
        axes[1, i].contour(0.5*(xedges2[:-1]+xedges2[1:]), 0.5*(yedges2[:-1]+yedges2[1:]), H2.T, 
                          levels=levels2, colors=['yellow', 'orange', 'red'], linewidths=1.5)
        axes[1, i].set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]')
        axes[1, i].set_ylabel(r'$q$ (mass ratio)')
        axes[1, i].grid(True, alpha=0.3)
        
        # Third row: Total mass vs redshift
        H3, xedges3, yedges3 = np.histogram2d(chirp_mass, redshift, bins=(mass_bins, redshift_bins), weights=weights)
        H3_sorted = np.sort(H3.flatten())[::-1]
        H3_cumsum = H3_sorted.cumsum()
        H3_total = H3_cumsum[-1]
        level3_1 = H3_sorted[np.searchsorted(H3_cumsum, 0.683 * H3_total)]
        level3_2 = H3_sorted[np.searchsorted(H3_cumsum, 0.954 * H3_total)]
        level3_3 = H3_sorted[np.searchsorted(H3_cumsum, 0.997 * H3_total)]
        levels3 = [level3_3, level3_2, level3_1]
        
        axes[2, i].imshow(H3.T, origin='lower', extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]], 
                         cmap='gray_r', aspect='auto', interpolation='gaussian')
        axes[2, i].contour(0.5*(xedges3[:-1]+xedges3[1:]), 0.5*(yedges3[:-1]+yedges3[1:]), H3.T, 
                          levels=levels3, colors=['yellow', 'orange', 'red'], linewidths=1.5)
        axes[2, i].set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]')
        axes[2, i].set_ylabel(r'Redshift $z$')
        axes[2, i].grid(True, alpha=0.3)

# Save the figure
output_path = '/home/users/b/briel/scratch/high_mass_physics/figures/main_figure.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

# Show the figure
plt.show()
