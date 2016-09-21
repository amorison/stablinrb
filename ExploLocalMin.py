#!/usr/bin/env python3
"""
Tracking of neutral Ra of harmonic degree corresponding to square cells
         and of neutral Ra of mode l=1
through problems space.
"""
import numpy as np
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

gams = np.linspace(0.2, 0.99, ngamma)

ran1s = []
ran2s = []

for gamma in gams:
    ana.phys.gamma = gamma
    (_, ran1), (_, ran2) = ana.ran_l_mins()
    ran1s.append(ran1 * (1-gamma)**3)
    ran2s.append(ran2 * (1-gamma)**3)

fig, axis = plt.subplots(1, 1)
axis.plot(gams, ran1s, label=r'$l=1$')
axis.plot(gams, ran2s, label=r'$l_*$')
axis.set_xlabel(r'$\gamma$', fontsize=FTSZ)
axis.set_ylabel(r'$\mathrm{Ra}_n(1-\gamma)^3$', fontsize=FTSZ)
axis.legend(loc='upper right', fontsize=FTSZ)
axis.tick_params(labelsize=FTSZ)
plt.tight_layout()
plt.savefig('track_lmin.pdf', format='PDF')
