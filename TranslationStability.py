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
import numpy.ma as ma

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
kplus = np.zeros(epsilon.shape)
kminus = np.zeros(epsilon.shape)

for j, eps in enumerate(epsilon):
    rtr = 12*(phib+phit)
    ran = rtr*(1+eps)
    sigmax[j], kmax[j], kminus[j], kplus[j] = ana.critical_harm(ran, kguess)
    # sigmax[j], kmax[j] = ana.fastest_mode(ran, harm=kguess)
    for i, kxn in enumerate(wkn):
        sigma[i] = ana.eigval(kxn, ran)

    axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2e$' %(eps))

axe.semilogx(kmax, np.real(sigmax), 'o', c='k')
kpma = ma.array(kplus, mask=np.real(sigmax)<0)
axe.semilogx(kpma, np.zeros(kpma.shape), 'o', c='r')
kmma = ma.array(kminus, mask=np.real(sigmax)<0)
axe.semilogx(kmma, np.zeros(kmma.shape), 'o', c='b')

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
