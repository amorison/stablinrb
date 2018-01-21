#!/usr/bin/env python3
"""
Weakly non linear analysis
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo
from plotting import plot_fastest_mode, plot_ran_harm
from misc import normalize_modes

# plt.rc('font', family='Helvetica')
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

mpl.rcParams['pdf.fonttype'] = 42

# scientific format for text
def fmt(x):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if a=='1':
        return r'$10^{{{}}}$'.format(b)
    if b != 0:
        return r'${} \times 10^{{{}}}$'.format(a, b)
    else:
        return r'${}$'.format(a)


# Font and markers size
FTSZ = 16
MSIZE = 3

# Controls
PLOT_NURA = False
PLOT_COEF_NU = False
PLOT_COEFT = False
PLOT_BOTTOP = False
PLOT_BOT_ONLY = False
PLOT_TEMP = False
PLOT_BOTTOP_BOTONLY = True
COMPUTE = False

# file to read from
fich = 'NonLin29.dat'

pblm = PhysicalProblem(
    gamma=None,
    phi_top=None,
    phi_bot=None,
    freeslip_top=True,
    freeslip_bot=True,
    ref_state_translation=False)

NNONLIN = 2
ana = NonLinearAnalyzer(pblm, ncheb=20, nnonlin=NNONLIN)

# standard case (infinite phi_top and phi_bot)
rac = 27*np.pi**4/4
kxc = np.pi/np.sqrt(2)
coef = 2
# initialize table
if COMPUTE:
    maxlogphi = 5
    minlogphi = -2
    ndeca = maxlogphi - minlogphi
    # make sure it goes through integer powers of 10
    nphi = ndeca  * 4 + 1
    phinum = np.flipud(np.power(10, np.linspace(minlogphi, maxlogphi, nphi)))
else:
    nphi = sum(1 for line in open(fich))

# both BC open
kx_c = np.zeros(nphi)
ray = np.zeros((nphi, NNONLIN+1))
meant = np.zeros((nphi, NNONLIN+1))
qtop = np.zeros((nphi, NNONLIN+1))
coef_nu = np.zeros(nphi)
# only bottom open
kx_cb = np.zeros(nphi)
rayb = np.zeros((nphi, NNONLIN+1))
meantb = np.zeros((nphi, NNONLIN+1))
qtopb = np.zeros((nphi, NNONLIN+1))
coef_nub = np.zeros(nphi)

if COMPUTE:
    for i, phi in enumerate(phinum):
        # both boundaries open
        print('---- i = %i over nphi = %i, phi = %.3e' %(i, nphi, phi))
        print('Both open')
        ana.phys.phi_top = phi
        ana.phys.phi_bot = phi
        kx_c[i], ray[i], _, meant[i], qtop[i] = ana.nonlinana()
        print('ray =', ray[i])
        print('qtop =', qtop[i])
        # leading order coefficient: Nu = 1 + A (Ra-Rac)/Rac
        coef_nu[i] = ray[i, 0] * qtop[i, 2] / ray[i, 2]
        print('CoefNu = ', coef_nu[i])
        # only bottom open
        print('Only bottom')
        ana.phys.phi_top = None
        kx_cb[i], rayb[i], _, meantb[i], qtopb[i] = ana.nonlinana()
        print('ray =', rayb[i])
        print('qtop =', qtopb[i])
        # leading order coefficient: Nu = 1 + A (Ra-Rac)/Rac
        coef_nub[i] = rayb[i, 0] * qtopb[i, 2] / rayb[i, 2]
        print('CoefNu = ', coef_nub[i])

        # save
        with open('NonLin' + np.str(nphi) + '.dat', 'w') as fich:
            fmt = '{:13.4e}' * 17 + '\n'
            for i in range(nphi):
                fich.write(fmt.format(phinum[i], kx_c[i], ray[i, 0], ray[i, 2], meant[i, 0], meant[i, 2],
                                      qtop[i, 0], qtop[i, 2], coef_nu[i], kx_cb[i], rayb[i, 0], rayb[i, 2],
                                      meantb[i, 0], meantb[i, 2], qtopb[i, 0], qtopb[i, 2], coef_nub[i]))
else:
    # read - only done for NNONLIN == 2 for now
    print('reading from file: ' + fich)
    phinum, kx_c, ray[:, 0], ray[:, 2], meant[:, 0], meant[:, 2], qtop[:, 0],\
      qtop[:, 2], coef_nu, kx_cb, rayb[:, 0], rayb[:, 2], meantb[:, 0], \
      meantb[:, 2], qtopb[:, 0], qtopb[:, 2], coef_nub = np.loadtxt(fich, unpack=True)
    
if PLOT_BOTTOP:
    fig, axe = plt.subplots(1, 2, dpi=300)
    fig.set_size_inches(9, 4)
    # left : coef
    # theory
    lowf = lambda x : 71680 / (21504 - 688 * x + 9 * x ** 2)
    lowphi = np.power(10, np.linspace(-2, 0, nphi))
    axe[0].semilogx([1e2, 1e5], [coef, coef], c='k', label=r'$\Phi^\pm \longrightarrow \infty$')
    axe[0].semilogx(lowphi, lowf(lowphi), '--', c='k', label=r'$\Phi^\pm \longrightarrow 0$')
    axe[0].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[0].set_ylim([1.9, 3.5])
    axe[0].set_xlabel(r'$\Phi^+ = \Phi^-$', fontsize=FTSZ)
    axe[0].set_ylabel(r'Heat flux coefficient, $A$', fontsize=FTSZ)
    # numerical results
    axe[0].semilogx(phinum, coef_nu, 'o', markersize=MSIZE)#, label=r'varying $\Phi^+=\Phi^-$')
    axe[0].legend(loc='upper right', fontsize=FTSZ)
    # right Nu - Ra for a subset
    maxnu = 5
    for i, phi in reversed(list(enumerate(phinum))):
        if np.log10(phi) - np.int(np.log10(phi)) == 0 and np.log10(phi) <= 3:
            rayl = np.array([1, 1 + (maxnu - 1) / coef_nu[i]]) * ray[i, 0]
            nutab = np.array([1, maxnu])
            axe[1].plot(rayl, nutab, label=r'$\Phi = $' + fmt(phi))
    axe[1].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[1].set_ylabel(r'$\mbox{\textit{Nu}}$', fontsize=FTSZ)
    axe[1].yaxis.set_label_position("right")
    axe[1].yaxis.tick_right()
    axe[1].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[1].legend(loc='lower right', fontsize=FTSZ-2)

    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    plt.savefig('HFcoeff_NuRa_BotTop.pdf')
    plt.close(fig)

if PLOT_BOT_ONLY:
    fig, axe = plt.subplots(1, 2, dpi=300)
    fig.set_size_inches(9, 4)
    # left : coef
    # theory
    axe[0].semilogx([1e2, 1e5], [coef, coef], c='k', label=r'$\Phi^\pm \longrightarrow \infty$')
    axe[0].tick_params(axis='both', which='major', labelsize=FTSZ)
    # axe[0].set_ylim([1, 2.2])
    axe[0].set_xlabel(r'$\Phi^-$', fontsize=FTSZ)
    axe[0].set_ylabel(r'Heat flux coefficient, $A$', fontsize=FTSZ)
    # numerical results
    axe[0].semilogx(phinum, coef_nub, 'o', markersize=MSIZE)#, label=r'varying $\Phi^+=\Phi^-$')
    axe[0].legend(loc='lower right', fontsize=FTSZ)
    # right Nu - Ra for a subset
    maxnu = 5
    for i, phi in reversed(list(enumerate(phinum))):
        if np.log10(phi) - np.int(np.log10(phi)) == 0 and np.log10(phi) <= 3:
            rayl = np.array([1, 1 + (maxnu - 1) / coef_nub[i]]) * rayb[i, 0]
            nutab = np.array([1, maxnu])
            axe[1].plot(rayl, nutab, label=r'$\Phi = $' + fmt(phi))
    axe[1].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[1].set_ylabel(r'$\mbox{\textit{Nu}}$', fontsize=FTSZ)
    axe[1].yaxis.set_label_position("right")
    axe[1].yaxis.tick_right()
    axe[1].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[1].legend(loc='lower right', fontsize=FTSZ-2)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('HFcoeff_NuRa_BotOnly.pdf')
    plt.close(fig)

if PLOT_TEMP:
    coef_t = meantb[:, 2] * ray[:, 0] / ray[:, 2]
    fig, axe = plt.subplots(1, 2, dpi=300)
    fig.set_size_inches(9, 4)
    # left : coef
    # theory
    axe[0].semilogx([1e2, 1e5], [0, 0], c='k', label=r'$\Phi^\pm \longrightarrow \infty$')
    axe[0].tick_params(axis='both', which='major', labelsize=FTSZ)
    # axe[0].set_ylim([1, 2.2])
    axe[0].set_xlabel(r'$\Phi^-$', fontsize=FTSZ)
    axe[0].set_ylabel(r'$\langle T \rangle$ coefficient, $B$', fontsize=FTSZ)
    # numerical results
    axe[0].semilogx(phinum, coef_t, 'o', markersize=MSIZE)#, label=r'varying $\Phi^+=\Phi^-$')
    axe[0].legend(loc='upper right', fontsize=FTSZ)
    # right Nu - Ra for a subset
    maxnu = 5
    for i, phi in reversed(list(enumerate(phinum))):
        if np.log10(phi) - np.int(np.log10(phi)) == 0 and np.log10(phi) <= 3:
            raylb = np.array([1, 1 + (maxnu - 1) / coef_nub[i]]) * rayb[i, 0]
            meant = np.array([0, coef_t[i] * (raylb[1] - rayb[i, 0]) / rayb[i, 0]]) + 0.5
            axe[1].plot(raylb, meant, label=r'$\Phi = $' + fmt(phi))
    axe[1].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[1].set_ylabel(r'$\langle T\rangle$', fontsize=FTSZ)
    axe[1].yaxis.set_label_position("right")
    axe[1].yaxis.tick_right()
    axe[1].legend(loc='upper right', fontsize=FTSZ)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('Tcoeff_NuT_BotOnly.pdf')
    plt.close(fig)

if PLOT_NURA:
    # maxnu = 2
    coef_t = meantb[:, 2] * ray[:, 0] / ray[:, 2]
    # Ra for <T>=1 at the smallest Phi
    ramax = rayb[-1, 0] * (1 + 0.5 / coef_t[-1])
    print('ramax = ', ramax)
    # corresponding Nu
    maxnu = 1 + coef_nub[-1] * (ramax - rayb[-1, 0]) / rayb[-1, 0]
    # first, the case for both boundaries with a phase change
    fig, axe = plt.subplots(1, 2, sharey=True, dpi=300)
    fig.set_size_inches(9, 4)
    print('plotting Nu-Ra - both BC')
    # also the mean temperature
    figt, axt = plt.subplots()
    axt.set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axt.set_ylabel(r'$\langle T\rangle$', fontsize=FTSZ)

    leg = []
    lab = []
    for i, phi in reversed(list(enumerate(phinum))):
        rayl = np.array([1, 1 + (maxnu - 1) / coef_nu[i]]) * ray[i, 0]
        nutab = np.array([1, maxnu])
        l, = axe[0].plot(rayl, nutab, label=r'$\Phi = $' + fmt(phi))
        leg.append(l)
        lab.append(r'$\Phi = $' + fmt(phi))
    axe[0].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[0].set_ylabel(r'$\mbox{\textit{Nu}}$', fontsize=FTSZ)
    # then, the case for a phase change only at the bottom
    print('plotting Nu-Ra - bottom only')
    for i, phi in reversed(list(enumerate(phinum))):
        raylb = np.array([1, 1 + (maxnu - 1) / coef_nub[i]]) * rayb[i, 0]
        nutab = np.array([1, maxnu])
        axe[1].plot(raylb, nutab, label=r'$\Phi = $' + fmt(phi))
        meant = np.array([0, coef_t[i] * (raylb[1] - rayb[i, 0]) / rayb[i, 0]]) + 0.5
        axt.plot(raylb, meant, label=r'$\Phi = $' + fmt(phi))
    axe[1].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)

    lgd = fig.legend(handles = leg, labels=lab, loc="upper center",
                   bbox_to_anchor=(0., 1.02, 1., .102), borderaxespad=0., #mode="expand",
                   ncol=3, fancybox=True, shadow=False, fontsize=FTSZ-2)  #, borderaxespad=0. 

    fig.savefig('Nu-Ra.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)
    axt.legend(loc='upper right', fontsize=FTSZ-2)
    figt.savefig('meanT-Ra.pdf', bbox_inches='tight')
    plt.close(figt)


if PLOT_COEF_NU:
    lowf = lambda x : 71680 / (21504 - 688 * x + 9 * x ** 2)
    lowphi = np.power(10, np.linspace(-2, 0, nphi))

    fig, axe = plt.subplots()
    plt.semilogx([1e2, 1e5], [coef, coef], c='k', label=r'$\Phi^\pm \longrightarrow \infty$')
    plt.semilogx(lowphi, lowf(lowphi), '--', c='k', label=r'$\Phi^\pm \longrightarrow 0$')
    axe.tick_params(axis='both', which='major', labelsize=FTSZ)
    axe.set_ylim([0.9, 3.5])
    plt.xlabel(r'$\Phi^+, \Phi^-$', fontsize=FTSZ)
    plt.ylabel(r'Heat flux coefficient', fontsize=FTSZ)
    axe.semilogx(phinum, coef_nu, 'o', markersize=MSIZE, label=r'varying $\Phi^+=\Phi^-$')
    plt.legend(loc='lower right', fontsize=FTSZ)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('HF_coeff_BotTop.pdf')
    axe.semilogx(phinum, coef_nub, 'o', markersize=MSIZE, label=r'$\Phi^+=\infty, \quad \Phi^-$ varying')
    plt.legend(loc='lower right', fontsize=FTSZ)
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('HF_coeff_both.pdf')
    plt.close(fig)

if PLOT_COEFT:
    # plot the coefficient of epsilon in mean T as function of phi
    coef_t = meantb[:, 2] * ray[:, 0] / ray[:, 2]
    fig, axe = plt.subplots()
    axe.tick_params(axis='both', which='major', labelsize=FTSZ)
    plt.xlabel(r'$\Phi^-$', fontsize=FTSZ)
    plt.ylabel(r'$\langle T \rangle$ coefficient', fontsize=FTSZ)
    axe.semilogx(phinum, coef_t, 'o', markersize=MSIZE)
    plt.savefig('meanT_coeff.pdf')
    plt.close(fig)
    
if PLOT_BOTTOP_BOTONLY:
    fig, axe = plt.subplots(1, 3, dpi=300)
    fig.set_size_inches(15, 4)
    # left: Nu - Ra for a subset - Bot and Top open
    maxnu = 5
    for i, phi in reversed(list(enumerate(phinum))):
        if np.log10(phi) - np.int(np.log10(phi)) == 0 and np.log10(phi) <= 3:
            rayl = np.array([1, 1 + (maxnu - 1) / coef_nu[i]]) * ray[i, 0]
            nutab = np.array([1, maxnu])
            axe[0].plot(rayl, nutab, label=r'$\Phi^\pm = $' + fmt(phi))
    axe[0].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[0].set_ylabel(r'$\mbox{\textit{Nu}}$', fontsize=FTSZ)
    axe[0].set_title(r'Both boundaries open', fontsize=FTSZ)
    axe[0].yaxis.set_label_position("left")
    axe[0].yaxis.tick_left()
    axe[0].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[0].legend(loc='lower right', fontsize=FTSZ-3)
    
    # middle : coef for both
    lowf = lambda x : 71680 / (21504 - 688 * x + 9 * x ** 2)
    lowphi = np.power(10, np.linspace(-2, 0, nphi))
    axe[1].semilogx([1e2, 1e5], [coef, coef], c='k')#, label=r'$\Phi^\pm \longrightarrow \infty$')
    axe[1].semilogx(lowphi, lowf(lowphi), '--', c='k')#, label=r'$\Phi^\pm \longrightarrow 0$')
    axe[1].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[1].set_ylim([0.9, 3.5])
    axe[1].set_xlabel(r'$\Phi^+, \Phi^-$', fontsize=FTSZ)
    axe[1].set_ylabel(r'Heat flux coefficient, $A$', fontsize=FTSZ)
    axe[1].set_title(r'$A$', fontsize=FTSZ)
    axe[1].semilogx(phinum, coef_nu, 'o', markersize=MSIZE, label=r'varying $\Phi^+=\Phi^-$')
    axe[1].semilogx(phinum, coef_nub, 'o', markersize=MSIZE, label=r'$\Phi^+=\infty, \quad \Phi^-$ varying')
    axe[1].legend(loc='lower right', fontsize=FTSZ-3)
    # theory
    # axe[0].semilogx([1e2, 1e5], [coef, coef], c='k', label=r'$\Phi^\pm \longrightarrow \infty$')
    # axe[0].tick_params(axis='both', which='major', labelsize=FTSZ)
    # axe[0].set_ylim([1, 2.2])
    # axe[0].set_xlabel(r'$\Phi^-$', fontsize=FTSZ)
    # axe[0].set_ylabel(r'Heat flux coefficient, $A$', fontsize=FTSZ)
    # numerical results
    # axe[0].semilogx(phinum, coef_nub, 'o', markersize=MSIZE)#, label=r'varying $\Phi^+=\Phi^-$')
    # axe[0].legend(loc='lower right', fontsize=FTSZ)

    # right: Nu-Ra for bot only
    for i, phi in reversed(list(enumerate(phinum))):
        if np.log10(phi) - np.int(np.log10(phi)) == 0 and np.log10(phi) <= 3:
            rayl = np.array([1, 1 + (maxnu - 1) / coef_nub[i]]) * rayb[i, 0]
            nutab = np.array([1, maxnu])
            axe[2].plot(rayl, nutab, label=r'$\Phi^- = $' + fmt(phi))
    axe[2].set_xlabel(r'$\mbox{\textit{Ra}}$', fontsize=FTSZ)
    axe[2].set_ylabel(r'$\mbox{\textit{Nu}}$', fontsize=FTSZ)
    axe[2].yaxis.set_label_position("left")
    axe[2].yaxis.tick_left()
    axe[2].set_title(r'One boundary open', fontsize=FTSZ)
    axe[2].tick_params(axis='both', which='major', labelsize=FTSZ)
    axe[2].legend(loc='lower right', fontsize=FTSZ-3)
    print('last coef_nub = ', coef_nub[-1])
    print('last rayb = ', rayb[-1, 0])
    print('d Nub/dRab = ', coef_nub[-1]/rayb[-1, 0])
    print('closed rayb = ', rayb[0, 0])
    print('closed d Nub/dRab = ', coef_nub[0]/rayb[0, 0])

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('HFcoeff_NuRa_BotOnly_Both.pdf')
    plt.close(fig)
