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
from physics import PhysicalProblem

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
    ref_state_translation=False)

ana = LinearAnalyzer(pblm, ncheb=20)

# first determine the linear stability without translation
ra_def, kguess = ana.critical_ra()

# turn on the translation mode
ana.phys.ref_state_translation = True

# compute the range of exploration in ra
# min is the critical Ra for translation
# max is Ra beyond which translation is always stable
rtr = 12 * (phit + phib)
smax0, kmax0, kminus0, kplus0 = ana.critical_harm(rtr, kguess)
ramax, hmax, ramin, hmin, smin = ana.max_ra_trans_instab(hguess=kguess)

# range of exploration
epsmax = (ramax - ramin) / ramin
neps = 30
epsilon = np.linspace(0,epsmax, neps)
wnmin = np.floor(np.log10(kminus0))
wnmax = np.ceil(np.log10(kplus0))
wkn = np.power(10, np.linspace(wnmin, wnmax, 100))

# initialisation of arrays
sigma = np.zeros(wkn.shape)
axe = plt.subplot()
kmax = np.zeros(epsilon.shape)
sigmax = np.zeros(epsilon.shape) - 1
kplus = np.zeros(epsilon.shape)
kminus = np.zeros(epsilon.shape)
ran = np.zeros(epsilon.shape)

# plot growth rate as function of wavenumber
PLOT_KSIGMA = False
if PLOT_KSIGMA:
    fig1, axe = plt.subplots(1, 1)

# loop on the Rayleigh number values
for j, eps in enumerate(epsilon):
    ran[j] = rtr*(1+eps)
    if ran[j] != ramax:
        sigmax[j], kmax[j], kminus[j], kplus[j] = ana.critical_harm(ran[j], kguess)
    else:
        sigmax[j] = 0
        kmax[j] = hmax
        kminus[j] = hmax
        kplus[j] = hmax
    if PLOT_KSIGMA:
        for i, kxn in enumerate(wkn):
            sigma[i] = ana.eigval(kxn, ran[j])
        axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2e$' %(eps))
    if np.real(sigmax[j]) < 0:
        break

ktot = np.concatenate((kminus, np.flipud(kplus)))
ratot = np.concatenate((ran, np.flipud(ran)))
eptot = np.concatenate((epsilon, np.flipud(epsilon)))

if PLOT_KSIGMA:
    axe.semilogx(kmax, np.real(sigmax), 'o', c='k')
    axe.semilogx(kplus, np.zeros(kplus.shape), 'o', c='r')
    axe.semilogx(kminus, np.zeros(kminus.shape), 'o', c='b')

    axe.set_ylim(-0.05, 0.05)

    axe.set_xlabel(r'$k_x$', fontsize=FTSZ)
    axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
    plt.legend(loc='upper center', ncol=2, fontsize=FTSZ)
    axe.xaxis.grid(True, 'major')
    axe.yaxis.grid(True, 'major')

    plt.savefig('sigmaRaN' + np.str(ana._ncheb) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')
    plt.close(fig1)

fig2, axe2 = plt.subplots(1, 1)

axe2.semilogx(kmax, epsilon, c='k', label='Max. growth rate')
axe2.fill_between(ktot, 0, eptot, alpha=0.5, label='Unstable translation')
axe2.set_xlabel(r'$k_x$', fontsize=FTSZ)
axe2.set_ylabel(r'$(Ra-Ra_c)/Ra_c$', fontsize=FTSZ)
plt.legend(loc='upper left', fontsize=FTSZ)


plt.savefig('kxEpsilonTransN' + np.str(ana._ncheb) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')
plt.close(fig2)

fig3, axe3 = plt.subplots(1, 1)

axe3.semilogx(kmax, ran, c='k', label='Max. growth rate')
axe3.fill_between(ktot, rtr, ratot, alpha=0.5, label='Unstable translation')
axe3.set_xlabel(r'$k_x$', fontsize=FTSZ)
axe3.set_ylabel(r'$Ra$', fontsize=FTSZ)
plt.legend(loc='upper left', fontsize=FTSZ)


plt.savefig('kxRaTransN' + np.str(ana._ncheb) +
            'Top' + np.str(phit).replace('.', '-') +
            'Bot' + np.str(phib).replace('.', '-') + '.pdf')
plt.close(fig3)

