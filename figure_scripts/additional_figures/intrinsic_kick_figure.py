import numpy as np
import matplotlib.pyplot as plt
import os
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from pathlib import Path
from posydon.config import PATH_TO_POSYDON
import h5py
from cmap import Colormap
from scipy.interpolate import RegularGridInterpolator

plt.style.use(str(Path(PATH_TO_POSYDON) / 'posydon' / 'visualization' / 'posydon.mplstyle'))

# load the spline data
spline_file = '/home/users/b/briel/scratch/high_mass_physics/data/spline_data/ppd_pdfs_mean_m1qzchieff_mmin40_collector_only_10000w_10000s_rng129_zspline.h5'
file = h5py.File(spline_file, 'r')

m1_bins_pdf = file['mgrid'][:]
q_bins_pdf = file['qgrid'][:]
pdf_spline = file['mass_pdf'][:]

spline_interpolator = RegularGridInterpolator(
    (m1_bins_pdf, q_bins_pdf),
    pdf_spline.T,
    method='linear',
    bounds_error=False,
    fill_value=0
)


high_res_mchirp_bins = np.linspace(1, 200, 2000)
high_res_q_bins = np.linspace(0.0001, 1, 2000)
X,Y  = np.meshgrid(high_res_mchirp_bins, high_res_q_bins, indexing='ij')
high_res_m1_values = Y**(-3/5) * (1+Y)**(1/5) * X
points = np.array([high_res_m1_values.ravel(), Y.ravel()]).T
Z = spline_interpolator(points).reshape(X.shape)* Y**(-3/5) * (1+Y)**(1/5)
# make into a density
Z = Z / np.sum(Z)
pdf_levels = [np.quantile(Z, 0.05),
              np.quantile(Z, 0.95),
              np.quantile(Z, 0.99),
              ]

cm_blues = Colormap('colorbrewer:Blues')
pdf_colours = cm_blues([0.2, 0.6, 1.0])
cm_grays = Colormap('colorbrewer:Greys')


data_dir = '/home/users/b/briel/scratch/high_mass_physics/data/figure_2/'
folder_types = ['no_kick', 'low_kick', 'normal_kick']
SFH_type = 'IllustrisTNG'

mass_bins = np.linspace(10, 200, 51)
q_bins = np.linspace(0, 1, 51)
chi_bins = np.linspace(-0.2, 1, 51)

fig, axes = plt.subplots(2, 3, figsize=(3.38*2, 2.535*2))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for i, folder_type in enumerate(folder_types):
    print(f"Processing folder: {folder_type}")
    co_contact_file = os.path.join(data_dir, folder_type+'.h5')
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
    
    # Top row: chirp mass vs mass ratio
    H0, xedges0, yedges0 = np.histogram2d(chirp_mass,
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

    axes[0, i].imshow(H0.T,
                      origin='lower',
                      extent=[xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]],
                      cmap=cm_grays.to_mpl(),
                      aspect='auto',
                      interpolation='bilinear')
    axes[0, i].contour(H0.T,
                       levels=levels0,
                       colors=cm_grays([0.2, 0.6, 1.0]),
                       linewidths=[1.0, 1, 1],
                       extent=[xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]])

    axes[0,i].contour(X, Y, Z,
                      levels=pdf_levels,
                      colors=pdf_colours,
                      linewidths=[1.5, 1.5, 1.5])

    # Top bottom row: Chirp mass vs chi_eff
    H1, xedges1, yedges1 = np.histogram2d(chirp_mass,
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

    axes[1, i].imshow(H1.T,
                      origin='lower',
                      extent=[xedges1[0], xedges1[-1],
                              yedges1[0], yedges1[-1]], 
                      cmap=cm_grays.to_mpl(),
                      aspect='auto',
                      interpolation='bilinear')

    axes[1, i].contour(H1.T,
                       levels=levels1,
                       colors=cm_grays([0.2, 0.6, 1.0]),
                       linewidths=[1.0, 1, 1],
                       extent=[xedges1[0], xedges1[-1],
                               yedges1[0], yedges1[-1]])

    # add rate denisty
    rate_density = np.nansum(weights)/volume
    axes[1, i].text(0.95,
                    0.95,
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


title_mapping = {
    'no_kick': 'No kicks',
    'low_kick': 'Low kicks',
    'normal_kick': 'Normal kicks'
}

for ax in axes[0,:]:
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    ax.set_ylim(1e-2,1)
    ax.set_title(title_mapping[folder_types[axes[0,:].tolist().index(ax)]])
    
for ax in axes.flatten():
    ax.set_xlabel(r'$M_\mathrm{chirp}$ [$M_\odot$]')
    ax.set_xlim(10, 200)

for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])
    
for ax in axes.flatten():
    ax.grid(ls='--', alpha=0.5)
    
output_dir = '/home/users/b/briel/scratch/high_mass_physics/figures'
plt.savefig(f'{output_dir}/intrinsic_figure_2_kicks.png', bbox_inches='tight')