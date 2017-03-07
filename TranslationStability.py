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
FTSZ = 8
MSIZE = 3

# range of exploration in phi
PLOT_K_RA = False
GREYSCALE = False
nphi = 50
phitot = np.power(10, np.flipud(np.linspace(-2, 2, nphi)))
phimax = np.max(phitot)
phimin = np.min(phitot)
cmax = 0.5
cmin = 0.2

def gscale(ppp):
    """grey color scale depending on phi"""
    ccc = cmin
    ccc += (np.log10(ppp) - np.log10(phimin)) / (np.log10(phimax) - np.log10(phimin)) * (cmax - cmin)
    return np.str(ccc)

kplus0 = np.zeros(phitot.shape)
kminus0 = np.zeros(phitot.shape)
kmax0 = np.zeros(phitot.shape)
smax0 = np.zeros(phitot.shape)
ramax = np.zeros(phitot.shape)
hmax = np.zeros(phitot.shape)

pblm = PhysicalProblem(
    gamma=None,
    phi_top=phitot[0],
    phi_bot=phitot[0],
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=False)

ana = LinearAnalyzer(pblm, ncheb=20)


PLOT_KSIGMA = False
neps = 100
nwkn = 200

# figure kx - epsilon
fig2, axe2 = plt.subplots(1, 1)

# figure kx - Ra
# fig3, axe3 = plt.subplots(1, 1)


for n, phi in enumerate(phitot):
    # compute the range of exploration in ra
    # min is the critical Ra for translation
    # max is Ra beyond which translation is always stable
    
    print('phi =', phi)
    phit = phi
    phib = phi
    ana.phys.phi_top = phit
    ana.phys.phi_bot = phib

    # turn off the translation mode to find initial guess
    ana.phys.ref_state_translation = False
    # first determine the linear stability without translation
    ra_def, kguess = ana.critical_ra()
    # turn on the translation mode
    ana.phys.ref_state_translation = True

    rtr = 12 * (phit + phib)
    smax0[n], kmax0[n], kminus0[n], kplus0[n] = ana.critical_harm(rtr, kguess)
    ramax[n], hmax[n], ramin, hmin, smin = ana.max_ra_trans_instab(hguess=kguess, eps=1e-3)

    if PLOT_K_RA:
        # range of exploration in Ra
        epsmax = (ramax[n] - ramin) / ramin
        epsilon = np.concatenate((np.linspace(0, epsmax/10, neps/2),\
                                  np.linspace(epsmax/10, epsmax, neps/2)))
        kmax = np.zeros(epsilon.shape)
        sigmax = np.zeros(epsilon.shape) - 1
        kplus = np.zeros(epsilon.shape)
        kminus = np.zeros(epsilon.shape)
        ran = np.zeros(epsilon.shape)


    # initialisation of arrays
    if PLOT_KSIGMA:
        # range of exploration in wave-number
        wnmin = np.max([np.floor(np.log10(kminus0[n])), -4])
        wnmax = np.ceil(np.log10(kplus0[n]))
        wkn = np.power(10, np.linspace(wnmin, wnmax, nwkn))
        sigma = np.zeros(wkn.shape)
        # plot growth rate as function of wavenumber
        fig1, axe = plt.subplots(1, 1)

    # loop on the Rayleigh number values
    if PLOT_KSIGMA or PLOT_K_RA:
        for j, eps in enumerate(epsilon):
            print('j, eps =', j, eps)
            ran[j] = rtr*(1+eps)
            if PLOT_K_RA:
                if ran[j] != ramax[n]:
                    sigmax[j], kmax[j], kminus[j], kplus[j] = ana.critical_harm(ran[j], kguess)
                else:
                    sigmax[j] = 0
                    kmax[j] = hmax[n]
                    kminus[j] = hmax[n]
                    kplus[j] = hmax[n]
            if PLOT_KSIGMA:
                for i, kxn in enumerate(wkn):
                    sigma[i] = ana.eigval(kxn, ran[j])
                axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2e$' %(eps))
            if np.real(sigmax[j]) < 0:
                break

    if PLOT_KSIGMA:
        # axe.semilogx(kmax, np.real(sigmax), 'o', c='k')
        # axe.semilogx(kplus, np.zeros(kplus.shape), 'o', c='r')
        # axe.semilogx(kminus, np.zeros(kminus.shape), 'o', c='b')

        # axe.set_ylim(-0.05, 0.05)

        axe.set_xlabel(r'$k_x$', fontsize=FTSZ)
        axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
        plt.legend(loc='upper center', ncol=2, fontsize=FTSZ)
        axe.xaxis.grid(True, 'major')
        axe.yaxis.grid(True, 'major')
        axe.set_ylim([-0.2, 0.4])
        plt.savefig('sigmaRaN' + np.str(ana._ncheb) +\
                        'Top' + np.str(phit).replace('.', '-') +\
                        'Bot' + np.str(phib).replace('.', '-') + '.pdf')
        plt.close(fig1)

    if PLOT_K_RA:
        ktot = np.concatenate((kminus, np.flipud(kplus)))
        ratot = np.concatenate((ran, np.flipud(ran)))
        eptot = np.concatenate((epsilon, np.flipud(epsilon)))

        axe2.semilogx(kmax, epsilon, '--', c='k')
        if GREYSCALE:
            axe2.fill_between(ktot, 0, eptot, color=gscale(phi), alpha=0.5, label='Unstable translation, $\phi=%.2f$' %phi)
        else:
            axe2.fill_between(ktot, 0, eptot, alpha=0.5, label='Unstable translation, $\phi=%.2f$' %phi)

    # axe3.semilogx(kmax, ran, c='k')
    # axe3.fill_between(ktot, rtr, ratot, alpha=0.5, label='Unstable translation, $\phi=%.1e$' %phi)

if PLOT_K_RA:
    axe2.set_xlim([1e-2, 4e0])
    axe2.set_xlabel(r'$k_x$', fontsize=FTSZ)
    axe2.set_ylabel(r'$(Ra-Ra_c)/Ra_c$', fontsize=FTSZ)
    axe2.legend(loc='upper left', fontsize=FTSZ)
    plt.savefig('kxEpsilonTransN' + np.str(ana._ncheb) + '.pdf')
    plt.close(fig2)

# axe3.set_xlabel(r'$k_x$', fontsize=FTSZ)
# axe3.set_ylabel(r'$Ra$', fontsize=FTSZ)
# axe3.legend(loc='upper left', fontsize=FTSZ)
# plt.savefig('kxRaTransN' + np.str(ana._ncheb) + '.pdf')

# plt.close(fig3)

with open('Phi_sigma_Ra.dat', 'w') as fich:
    fmt = '{:15.3e}'*7 + '\n'
    for i in range(nphi):
        fich.write(fmt.format(phitot[i], kmax0[i], smax0[i], kminus0[i], kplus0[i], hmax[i], ramax[i]))

fig, axe = plt.subplots(3, 1, sharex=True)
axe[0].loglog(phitot, kmax0, 'o', markersize=MSIZE, label=r'Fastest growing mode at $\varepsilon=0$')
# axe[0].loglog(phitot, hmax, 'o', label=r'$k_x$ for maximum $\varepsilon$')
# axe[0].loglog(phitot, kminus0, 'o', label=r'Minimum $k_x$ for instability at $\varepsilon=0$')
axe[0].loglog(phitot, kplus0, 'o', markersize=MSIZE, label=r'Maximum $k_x$ for instability at $\varepsilon=0$')
axe[0].legend(loc='upper left', fontsize=FTSZ)
axe[0].set_ylabel(r'$k_x$', fontsize=FTSZ)
axe[1].loglog(phitot, smax0, 'o', markersize=MSIZE, label=r'Maximum growth rate at $\varepsilon=0$')
axe[1].legend(loc='upper left', fontsize=FTSZ)
axe[1].set_ylabel('$Re(\sigma)$', fontsize=FTSZ)
axe[2].loglog(phitot, ramax, 'o', markersize=MSIZE, label=r'Maximum $Ra$ for instability')
axe[2].set_ylabel(r'$Ra_{max}$', fontsize=FTSZ)
axe[2].set_xlabel(r'$\Phi^+=\Phi^-$', fontsize=FTSZ)
plt.savefig('Phi_kx_smax_RamaxN' + np.str(ana._ncheb) + '.pdf')
plt.close(fig)


