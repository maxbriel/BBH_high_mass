import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from cmap import Colormap
from scipy.stats import gaussian_kde

from posydon.config import PATH_TO_POSYDON
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume


def precession(theta_1, theta_2, a1, a2, m1, m2):
    """Calculate the effective spin precession.
    
    Following the formulation in Gerosa+2021, which 
    is used in LIGO/Virgo analyses.
    
    Parameters
    ----------
    theta_1 : float or np.ndarray
        Tilt angle of the primary spin (in radians).
    theta_2 : float or np.ndarray
        Tilt angle of the secondary spin (in radians).
    a1 : float or np.ndarray
        Dimensionless spin magnitude of the primary.
    a2 : float or np.ndarray
        Dimensionless spin magnitude of the secondary.
    m1 : float or np.ndarray
        Mass of the primary.
    m2 : float or np.ndarray
        Mass of the secondary.
    """
    q = m2/m1
    a_1_perp = np.abs(a1 * np.sin(theta_1))
    a_2_perp = q * ((4*q + 3)/(4+3*q)) * a2 * np.sin(theta_2)
    chi_p = np.maximum(a_1_perp, a_2_perp)
    return chi_p

plt.style.use(str(Path(PATH_TO_POSYDON) / 'posydon' / 'visualization' / 'posydon.mplstyle'))

# Plot mean with filled uncertainty region
fig, ax = plt.subplots(1,1, figsize=(3.38, 2.535))

# Define the data directory and folder types
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/main_figure/'
folder_types = ['Eddington-limited', 'GRMHD', 'conservative']
SFH_type = 'IllustrisTNG'


title_mapping = {
    'no_kick': '',
    'low_kick': '-Low',
    'normal_kick': '-Normal',
}

cm = Colormap('tol:vibrant')

colours = cm([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])

chi_p_bins = np.linspace(-0.6, 1, 51)


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
    chi_p = precession(filtered_population['S1_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S2_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S1_spin'].to_numpy(),
                        filtered_population['S2_spin'].to_numpy(),
                        filtered_population['S1_mass'].to_numpy(),
                        filtered_population['S2_mass'].to_numpy())

    # get histogram
    h, _ = np.histogram(chi_p,
                    bins=chi_p_bins,
                    weights=np.nansum(weights, axis=1)/volume,)

    return h/np.diff(chi_p_bins)


for i, folder_type in enumerate(folder_types[:-2]):
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

    chi_p = precession(filtered_population['S1_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S2_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S1_spin'].to_numpy(),
                        filtered_population['S2_spin'].to_numpy(),
                        filtered_population['S1_mass'].to_numpy(),
                        filtered_population['S2_mass'].to_numpy())
    
    print(filtered_population['metallicity'].value_counts())
    
    
    
    # plot KDE
    h, _ = np.histogram(chi_p,
                        bins=chi_p_bins,
                        weights=np.nansum(weights, axis=1)/volume,
                        density=True)
    #kde = gaussian_kde(chi_p, weights=np.nansum(weights, axis=1)/volume)
    #x_eval = np.linspace(-0.6, 1.0, 500)
    #kde_values = kde(x_eval)
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(chi_p_bins[:-1],
             h,
             lw=1,
             label='Eddington',
             color=colours[i])
    

data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/figure_2/'
folder_types = ['no_kick', 'low_kick', 'normal_kick']
linestyles = ['solid', 'dashed', 'dotted']

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
    
    chi_p = precession(filtered_population['S1_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S2_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S1_spin'].to_numpy(),
                        filtered_population['S2_spin'].to_numpy(),
                        filtered_population['S1_mass'].to_numpy(),
                        filtered_population['S2_mass'].to_numpy())
    
    print(filtered_population['metallicity'].value_counts())
    # plot KDE
    #kde = gaussian_kde(chi_p, weights=np.nansum(weights, axis=1)/volume)
    #x_eval = np.linspace(-0.6, 1.0, 500)
    #kde_values = kde(x_eval)
    h, _ = np.histogram(chi_p,
                        bins=chi_p_bins,
                        weights=np.nansum(weights, axis=1)/volume,
                        density=True)
    
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(chi_p_bins[:-1],
             h,
             lw=1,
             label="Cons."+label,
             color=colours[5],
             ls=linestyles[i])
    
data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/kick_figure_GRMHD/'
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
    
    chi_p = precession(filtered_population['S1_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S2_spin_orbit_tilt'].to_numpy(),
                        filtered_population['S1_spin'].to_numpy(),
                        filtered_population['S2_spin'].to_numpy(),
                        filtered_population['S1_mass'].to_numpy(),
                        filtered_population['S2_mass'].to_numpy())
    
    print(filtered_population['metallicity'].value_counts())
    # plot KDE
    h, _ = np.histogram(chi_p,
                        bins=chi_p_bins,
                        weights=np.nansum(weights, axis=1)/volume,
                        density=True)
    #kde = gaussian_kde(chi_p, weights=np.nansum(weights, axis=1)/volume)
    #x_eval = np.linspace(-0.6, 1.0, 500)
    #kde_values = kde(x_eval)
    if folder_type in tuple(title_mapping.keys()):
        label = title_mapping[folder_type]
    else:
        label = folder_type
    
    plt.step(chi_p_bins[:-1],
             h,
             lw=1,
             label='GRMHD'+label,
             color=colours[3],
             ls=linestyles[i])
    

plt.ylim(0, 10)
plt.xlim(-0.6, 1.0)
plt.xlabel(r'$\chi_{\mathrm{p}}$')
plt.ylabel('PDF')
plt.legend(ncol=2)

output_dir = '/home/users/b/briel/scratch/high_mass_physics/figures'
plt.savefig(f'{output_dir}/intrinsic_chi_p.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/intrinsic_chi_p.pdf', bbox_inches='tight')