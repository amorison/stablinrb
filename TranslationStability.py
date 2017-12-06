#!/usr/bin/env python3
"""
Finds critical Rayleigh number.

Plots the Critical Ra as function of wave-number and finds the minimum.
Based on the matlab code provided by Thierry Alboussiere.
Can do both the no-slip and free-slip BCs, applying to both boundaries.
Also treats the phase change boundary conditions.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

mpl.rcParams['pdf.fonttype'] = 42

# Font and markers size
FTSZ = 12
MSIZE = 3

# range of exploration in phi
PLOT_K_RA = False
PLOT_KSIGMA = True
PLOT_RAMAX = False
# type of colorscale
GREYSCALE = False

# scientific format for text
def fmt(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    if b != 0:
        return r'${} \times 10^{{{}}}$'.format(a, b)
    else:
        return r'${}$'.format(a)

nphi = 3
if nphi == 1:
    phitot = np.array([1])
else:
    phitot = np.power(10, np.flipud(np.linspace(-2, 0, nphi)))

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
epsm = np.zeros(phitot.shape)
hmax = np.zeros(phitot.shape)

pblm = PhysicalProblem(
    gamma=None,
    phi_top=phitot[0],
    phi_bot=phitot[0],
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=False)

ana = LinearAnalyzer(pblm, ncheb=20)

# number epsilon values to plot
neps = 4
# number of wavenumber values for the plot
nwkn = 200

# figure kx - epsilon
fig2, axe2 = plt.subplots(1, 1)


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
    epsm[n] = (ramax[n] - 24 * phi) / (24 * phi)

    if PLOT_K_RA or PLOT_KSIGMA:
        # range of exploration in Ra
        epsmax = (ramax[n] - ramin) / ramin
        if neps > 5:
            epsilon = np.concatenate((np.linspace(0, epsmax/10, neps/2),\
                                    np.linspace(epsmax/10, epsmax, neps/2)))
        else:
            epsilon = np.linspace(0, epsmax, neps)
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
                axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = $'+fmt(eps))
            if np.real(sigmax[j]) < 0:
                break

    if PLOT_KSIGMA:

        axe.set_xlabel(r'$k$', fontsize=FTSZ+2)
        axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ+2)
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
        with open('kx_epsmax_phi'+np.str(phi).replace('.', '-')+'.dat', 'w') as fich:
            fmt = '{:15.6e}'*2 + '\n'
            for i in range(nphi):
                fich.write(fmt.format(ktot[i], eptot[i]))

        axe2.semilogx(kmax, epsilon, '--', c='k')
        if GREYSCALE:
            axe2.fill_between(ktot, 0, eptot, color=gscale(phi), alpha=0.5, label=r'Unstable translation, $\Phi=%.2f$' %phi)
        else:
            axe2.fill_between(ktot, 0, eptot, alpha=0.5, label=r'Unstable translation, $\Phi=%.2f$' %phi)

if PLOT_K_RA:
    axe2.set_xlim([1e-2, 1e0])
    axe2.set_xlabel(r'$k$', fontsize=FTSZ+2)
    axe2.set_ylabel(r'$(\mbox{\textit{Ra}}-\mbox{\textit{Ra}}_c)/\mbox{\textit{Ra}}_c$', fontsize=FTSZ+2)
    axe2.legend(loc='upper left', fontsize=FTSZ)
    plt.savefig('kxEpsilonTransN' + np.str(ana._ncheb) + '.pdf')
    plt.close(fig2)

if PLOT_RAMAX:
    with open('Phi_sigma_Ra.dat', 'w') as fich:
        fmt = '{:15.6e}'*8 + '\n'
        for i in range(nphi):
            fich.write(fmt.format(phitot[i], kmax0[i], smax0[i], kminus0[i], kplus0[i], hmax[i], ramax[i], epsm[i]))

    fig, axe = plt.subplots(3, 1, sharex=True)
    axe[0].loglog(phitot, kmax0, 'o', markersize=MSIZE, label=r'Fastest growing mode at $\varepsilon=0$')
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


