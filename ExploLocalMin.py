#!/usr/bin/env python3
"""
Tracking of neutral Ra of harmonic degree corresponding to square cells
         and of neutral Ra of mode l=1
through problems space.
"""
import heapq
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer
from physics import PhysicalProblem, visco_Arrhenius

# Font and markers size
FTSZ = 14
MSIZE = 5

ngamma = 200
gamma_min = 0.3
gamma_max = 0.85
phi_top = None
phi_bot = 1.e-2
eta_c = 10**5

ana = LinearAnalyzer(
    phys=PhysicalProblem(
        gamma=gamma_max,
        phi_top=phi_top,
        phi_bot=phi_bot,
        freeslip_top=True,
        freeslip_bot=True),
    ncheb=50)

_, (l_max, _) = ana.ran_l_mins()
l_max = max(l_max, 3)

gams = np.linspace(gamma_min, gamma_max, ngamma)
l_vals = np.arange(1, l_max+1)
rans_l = np.zeros((l_max, ngamma))

for l_harm in l_vals:
    ran_p = 600
    for igam, gamma in enumerate(gams):
        ana.phys.gamma = gamma
        if eta_c is not None:
            ana.phys.eta_r = visco_Arrhenius(eta_c, gamma)
        ran_p = ana.neutral_ra(l_harm, ra_guess=ran_p)
        rans_l[l_harm-1, igam] = ran_p * (1 - gamma)**3

# define discrete colormap
cmap = plt.cm.jet
norm = mpl.colors.BoundaryNorm(np.append(l_vals, l_max + 1) - 0.5, cmap.N)

# plot Ra_n(gamma) for each harmonic degree
fig, axis = plt.subplots(1, 2, gridspec_kw={'width_ratios': [24, 1]})
for l_harm in l_vals:
    axis[0].plot(gams, rans_l[l_harm-1], label=r'$l={}$'.format(l_harm),
              linestyle='dotted', color=cmap(norm(l_harm)))

# plot Ra_c(gamma)
l_mins = itertools.groupby(np.argmin(rans_l, 0) + 1)
i_s = 0
i_e = 0
for l_harm, l_dum in l_mins:
    i_e = i_s + len(list(l_dum))
    i_f = min(ngamma, i_e+1)
    axis[0].plot(gams[i_s:i_f], rans_l[l_harm-1, i_s:i_f],
                 color=cmap(norm(l_harm)))
    i_s = i_e

cbar = mpl.colorbar.ColorbarBase(axis[1], cmap=cmap, norm=norm, ticks=l_vals)
cbar.set_label(r'   $l$', rotation=0, fontsize=FTSZ)

axis[0].set_xlabel(r'$\gamma$', fontsize=FTSZ)
axis[0].set_ylabel(r'$\mathrm{Ra}_n(1-\gamma)^3$', fontsize=FTSZ)

axis[0].set_xlim([gamma_min, gamma_max])
ra_min = np.min(rans_l, 0)
ra_2ndmax = heapq.nsmallest(2, np.max(rans_l, 0))[1]
axis[0].set_ylim([np.min(ra_min),
                  max(np.max(ra_min), ra_2ndmax)
                 ])

axis[0].tick_params(labelsize=FTSZ)
axis[1].tick_params(labelsize=FTSZ)

if phi_bot is not None:
    title = '$\Phi^-={}$, '.format(phi_bot)
else:
    title = 'Closed bottom, '
if phi_top is not None:
    title += '$\Phi^+={}$, '.format(phi_top)
else:
    title += 'Closed top'
axis[0].set_title(title)

plt.tight_layout()
plt.savefig('track_lmin.pdf', format='PDF')
