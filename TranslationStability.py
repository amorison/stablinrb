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
# import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo
from plotting import plot_fastest_mode, plot_ran_harm
from misc import normalize_modes

# Font and markers size
FTSZ = 14
MSIZE = 5

phit = 1e-1
phib = 1e-1

pblm = PhysicalProblem(
    gamma=None,
    phi_top=phit,
    phi_bot=phib,
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=False)

ana = LinearAnalyzer(pblm, ncheb=20)

# first determine the linear stability without translation
ra_def, kguess = ana.critical_ra()

ana.ref_state_translation = True
epsilon = np.linspace(0, 0.001, 4)
wkn = np.power(10, np.linspace(-3, 0, 100))
sigma = np.zeros(wkn.shape)
ana.phys.ref_state_translation = True
axe = plt.subplot()
kmax = np.zeros(epsilon.shape)
sigmax = np.zeros(epsilon.shape)
kmax1 = np.zeros(epsilon.shape)
sigmax1 = np.zeros(epsilon.shape)

def find_max_k(ran, guess=1):
    """Find the maximum growth rate"""
    eps = [0.1, 0.01]
    harms = np.linspace(guess * (1 - 2 * eps[0]), guess * (1 + eps[0]), 3)
    sig  = [ana.eigval(h, ran) for h in harms]
    # fit a degree 2 polynomial
    pol = np.polyfit(harms, sig, 2)
    # minimum value
    exitloop = False
    kmax = -0.5*pol[1]/pol[0]
    for i, err in enumerate([0.03, 1.e-3]):
        while np.abs(kmax-harms[1]) > err*kmax and not exitloop:
            harms = np.linspace(kmax * (1 - eps[i]), kmax * (1 + eps[i]), 3)
            sig  = [ana.eigval(h, ran) for h in harms]
            pol = np.polyfit(harms, sig, 2)
            kmax = -0.5*pol[1]/pol[0]
            sigmax = sig[1]

    return kmax, sigmax


for j, eps in enumerate(epsilon):
    rtr = 12*(phib+phit)
    ran = rtr*(1+eps)
    sigmax[j], kmax[j] = ana.fastest_mode(ran, harm=kguess)
    for i, kxn in enumerate(wkn):
        sigma[i] = ana.eigval(kxn, ran)

    axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2e$' %(eps))

axe.semilogx(kmax, np.real(sigmax), 'o')

axe.set_ylim(-0.05, 0.05)

axe.set_xlabel(r'$k$', fontsize=FTSZ)
axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
plt.legend(loc='upper center', ncol=2, fontsize=FTSZ)
axe.xaxis.grid(True, 'major')
axe.yaxis.grid(True, 'major')

plt.savefig('sigmaRaN' + np.str(ana._ncheb) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')


def neutral_k():
    """Finds the value of k with a zero growth rate for translation"""
    
    return
