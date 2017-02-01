#!/usr/bin/env python3
"""
Finds critical Rayleigh number.

Plots the Critical Ra as function of wave-number and finds the minimum.
Based on the matlab code provided by Thierry Alboussiere.
Can do both the no-slip and free-slip BCs, applying to both boundaries.
Also treats the phase change boundary conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo
from plotting import plot_fastest_mode, plot_ran_harm
from misc import normalize_modes

# Font and markers size
FTSZ = 14
MSIZE = 5

phit = 1e2
phib = 1e2

pblm = PhysicalProblem(
    gamma=None,
    phi_top=phit,
    phi_bot=phib,
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=True)

ana = LinearAnalyzer(pblm, ncheb=30)

epsilon = np.linspace(0, 1.5, 4)
wkn = np.power(10, np.linspace(-2, 2, 100))
sigma = np.zeros(wkn.shape)
ana.phys.ref_state_translation = True
axe = plt.subplot()
for eps in epsilon:
    rtr = 12*(phib+phit)
    ran = rtr*(1+eps)
    for i, kxn in enumerate(wkn):
        sigma[i] = ana.eigval(kxn, ran)

    axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2f$' %(eps))

axe.set_ylim(-50, 100)

axe.set_xlabel(r'$k$', fontsize=FTSZ)
axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
plt.legend(loc='upper right', fontsize=FTSZ)

plt.savefig('sigmaRaN' + np.str(ana._ncheb) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')

