#!/usr/bin/env python3
"""
Tracking of neutral Ra of harmonic degree corresponding to square cells
         and of neutral Ra of mode l=1
through problems space.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer
from physics import PhysicalProblem

# Font and markers size
FTSZ = 14
MSIZE = 5

ngamma = 200

ana = LinearAnalyzer(
    phys=PhysicalProblem(
        gamma=0.7,
        phi_top=None,
        phi_bot=1.e-2,
        #eta_r = lambda r: np.exp(10*r-10),
        freeslip_top=True,
        freeslip_bot=True),
    ncheb=20)

gams = np.linspace(0.2, 0.85, ngamma)

ran1s = []
ran2s = []
l_stars = []

for gamma in gams:
    ana.phys.gamma = gamma
    (_, ran1), (l_star, ran2) = ana.ran_l_mins()
    ran1s.append(ran1 * (1-gamma)**3)
    ran2s.append(ran2 * (1-gamma)**3)
    l_stars.append(l_star)

l_stars = np.array(l_stars)
l_vals = np.unique(l_stars)
ran2s = np.array(ran2s)

# define discrete colormap
cmap = plt.cm.jet
lmax = np.max(l_stars)
norm = mpl.colors.BoundaryNorm(np.append(l_vals, lmax + 1) - 0.5, cmap.N)

# set of line segments colored individually
points = np.array([gams, ran2s]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(l_stars)

fig, axis = plt.subplots(1, 1)

axis.plot(gams, ran1s, label=r'$l=1$', linestyle='dotted',
          color=cmap(norm(1)))
#axis.plot(gams, ran2s, label=r'$l_\star$')
cax = axis.add_collection(lc)
cbar = plt.colorbar(cax, ticks=l_vals)
cbar.set_label(r'   $l_\star$', rotation=0, fontsize=FTSZ)

#axis.set_xscale('log')
#axis.set_yscale('log')

axis.set_xlabel(r'$\gamma$', fontsize=FTSZ)
axis.set_ylabel(r'$\mathrm{Ra}_n(1-\gamma)^3$', fontsize=FTSZ)
axis.legend(loc='upper right', fontsize=FTSZ)

axis.tick_params(labelsize=FTSZ)
plt.tight_layout()
plt.savefig('track_lmin.pdf', format='PDF')
