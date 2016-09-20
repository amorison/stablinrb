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

pblm = PhysicalProblem(
    gamma=None,
    phi_top=None,
    phi_bot=1e4,
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=False)

phit = 0.01
phib = 0.01
ana.phys.phi_top = phit
ana.phys.phi_bot = phib
# epsilon = np.flipud(np.linspace(0, 1, 4))
epsilon = np.array([5])
wkn = np.power(10, np.linspace(-1, 4, 100))
sigma = np.zeros(wkn.shape)
ana.phys.ref_state_translation = True
axe = plt.subplot()
for j, eps in enumerate(epsilon):
    rtr = 12*(phib+phit)
    ran = rtr*(1+eps)
    for i, kxn in enumerate(wkn):
        sigma[i] = ana.eigval(kxn, ran)

    axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2f$' %(eps))
    axe.set_xlabel(r'$k$', fontsize=FTSZ)
    axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
    plt.legend(loc='upper right', fontsize=FTSZ)
    # axe.set_ylim((-500, 1500))
    # axe.set_ylim(bottom=-500)

plt.savefig('sigmaRa' + np.str(eps) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')

