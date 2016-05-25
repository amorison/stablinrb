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
from analyzer import PhysicalProblem, LinearAnalyzer
from plotting import plot_fastest_mode, plot_ran_harm

pblm = PhysicalProblem(
    gamma=0.5,
    phi_top=None,
    phi_bot=None,
    freeslip_top=False,
    freeslip_bot=True,
    ref_state_translation=False)

ana = LinearAnalyzer(pblm)
name = ana.phys.name()

ra_c, harm_c = ana.critical_ra()

plot_fastest_mode(name, ana, harm_c, ra_c)
plot_ran_harm(name, ana, harm_c)


# Explore phi space
COMPUTE_PHASECHANGE = False

if COMPUTE_PHASECHANGE:
    nphi = 100
    phinum = np.flipud(np.power(10, np.linspace(-3, 5, nphi)))
    # Limit case for infinite phi
    rac = 27*np.pi**4/4
    kxc = np.pi/np.sqrt(2)
    # NCHEB = 30

    EQUAL_PHI = False
    # Computes properties as function of phi, equal for both boundaries
    if EQUAL_PHI:
        # First unstable mode
        # rrm, kkx = ra_ks(500, 2, NCHEB,
                        # phase_change_top=True, phase_change_bot=True,
                        # phitop=phinum[0], phibot=phinum[0])
        rrm, kkx, pmx, umx, wmx, tmx = findplot_rakx(NCHEB,
                        'PhaseChangeTop',
                        plotfig=False,
                        output_rakx=False,
                        phase_change_top=True, phase_change_bot=True,
                        phibot=phinum[0], phitop=phinum[0],
                        output_eigvec=True)
        ram = [rrm]
        kwn = [kkx]
        pmax = [pmx]
        umax = [umx]
        wmax = [wmx]
        tmax = [tmx]
        print(phinum[0], ram, kwn)
        for i, phi in enumerate(phinum[1:]):
            # rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
            #             phase_change_top=True, phase_change_bot=True,
            #             phitop=phi, phibot=phi)
            rrm, kkx, pmx, umx, wmx, tmx = findplot_rakx(NCHEB,
                        'PhaseChangeTop',
                        plotfig=False,
                        output_rakx=False,
                        phase_change_top=True, phase_change_bot=True,
                        phibot=phi, phitop=phi,
                        output_eigvec=True)
            print(i, phi, rrm, kkx)
            ram = np.append(ram, rrm)
            kwn = np.append(kwn, kkx)
            pmax = np.append(pmax, pmx)
            umax = np.append(umax, umx)
            wmax = np.append(wmax, wmx)
            tmax = np.append(tmax, tmx)
        # save
        with open('EqualTopBotPhi.dat', 'w') as fich:
            fmt = '{:13}'*7 + '\n'
            fich.write(fmt.format(' phi', 'kx', 'Ra', 'Pmax', 'Umax', 'Tmax', 'Wmax'))
            fmt = '{:15.3e}'*6 + '{:15.3}' + '\n'
            for i in range(nphi):
                fich.write(fmt.format(phinum[i], kwn[i], ram[i], pmax[i],
                                      umax[i], tmax[i], wmax[i]))

        # plot kx and ra as function of phi
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Theoretical prediction for translation
        axe[0].semilogx(phinum, 24*phinum, '--', c='k', label=r'Translation mode')
        # Theoretical for low phi development
        axe[0].semilogx(phinum, 24*phinum-81*phinum**2/256, '-', c='k',
                        label=r'Small $\Phi$ prediction')
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                        label=r'$\frac{27\pi^4}{4}$')
        # col0 = p0.get_color()
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE,
                              label=r'Fastest growing mode')
        col1 = p1.get_color()
        # General case
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=4)
        axe[1].loglog([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Small phi prediction
        axe[1].loglog(phinum, np.sqrt(9*phinum/32), '-', c='k',
                    label=r'Small $\Phi$ prediction')
        axe[1].loglog(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'Fastest growing mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^+=\Phi^-$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_EqualPhi.pdf", format='PDF')
        plt.close(fig)

        # plot Pmax, Umax, Wmax and (24phi - Ra) as function of phi
        fig, axe = plt.subplots(4, 1, sharex=True)
        axe[0].loglog(phinum, np.abs(pmax), marker='.', linestyle='-')
        axe[0].loglog(phinum, 13*np.sqrt(13)/8*phinum, linestyle='--', c='k',
                        label=r'$13\sqrt{13}\Phi/8$')
        axe[0].legend(loc='upper left', fontsize=FTSZ)
        axe[0].set_ylabel(r'$P_{max}$', fontsize=FTSZ)
        axe[1].loglog(phinum, np.abs(umax), marker='.', linestyle='-')
        axe[1].loglog(phinum, 3*np.sqrt(phinum/2), linestyle='--', c='k',
                        label=r'$3\sqrt{\Phi/2}$')
        axe[1].legend(loc='upper left', fontsize=FTSZ)
        axe[1].set_ylabel(r'$U_{max}$', fontsize=FTSZ)
        axe[2].semilogx(phinum, np.abs(wmax), marker='.', linestyle='-')
        axe[2].semilogx(phinum, 8*np.ones(phinum.shape), linestyle='--', c='k',
                        label=r'$8$')
        axe[2].set_ylim(ymin=7)
        axe[2].legend(loc='upper left', fontsize=FTSZ)
        axe[2].set_ylabel(r'$W_{max}$', fontsize=FTSZ)
        axe[3].loglog(phinum, 24*phinum-ram, marker='.', linestyle='-')
        axe[3].loglog(phinum, 81*phinum**2/256,  linestyle='--', c='k',
                        label=r'$81\Phi^2/256$')
        axe[3].set_ylabel(r'$24\Phi-Ra$', fontsize=FTSZ)
        axe[3].legend(loc='upper left', fontsize=FTSZ)
        axe[3].set_xlabel(r'$\Phi$', fontsize=FTSZ)
        # axe[3].set_ylim([1.e-6, 1.e2])
        plt.savefig('Phi_ModeMax.pdf', format='PDF')
        plt.close(fig)

    PHASETOPBOT = False
    if PHASETOPBOT:
        phib = 1.e-3
        phit = 1.e-3
        ramin, kxmin, pmax, umax, wmax, tmax = findplot_rakx(NCHEB,
                      'PhaseChangeTop' + np.str(phit).replace('.', '-') + 'Bot' + np.str(phib).replace('.', '-'),
                      plotfig=True,
                      output_rakx=True,
                      phase_change_top=True, phase_change_bot=True,
                      phibot=phib, phitop=phit,
                      output_eigvec=True)
        print('max =', ramin, kxmin, pmax, umax, wmax, tmax)

    PHASEBOTONLY = False
    if PHASEBOTONLY:
        # phib = 1.e-2
        # ramin, kxmin, pmax, umax, wmax, tmax = findplot_rakx(NCHEB,
                      # 'FreeTopPhaseChangeBot' + np.str(phib).replace('.', '-'),
                      # plotfig=True,
                      # output_rakx=True,
                      # phase_change_top=False, phase_change_bot=True,
                      # bcsu=np.array([[0, 1, 0], [0, 1, 0]], float),
                      # phibot=phib,
                      # output_eigvec=True)
        # print('max =', ramin, kxmin, pmax, umax, wmax, tmax)
        nphi = 20
        phinum = np.flipud(np.power(10, np.linspace(-2, 4, nphi)))
        ram = np.zeros(phinum.shape)
        kwn = np.zeros(phinum.shape)
        pmax = np.zeros(phinum.shape)
        umax = np.zeros(phinum.shape)
        wmax = np.zeros(phinum.shape)
        tmax = np.zeros(phinum.shape)
        # print(ram, kwn)
        for i, phi in enumerate(phinum):
            ram[i], kwn[i], pmax[i], umx, wmax[i], tmax[i] = findplot_rakx(
                NCHEB, 'PhaseChangeBot', plotfig=False, output_rakx=False,
                phase_change_top=False, phase_change_bot=True, phibot=phi,
                bcsu=np.array([[0, 1, 0], [0, 1, 0]], float),
                output_eigvec=True)
            umax[i] = np.imag(umx)
            print(i, phi, ram[i], kwn[i])
            # save in file
        with open('FreeTopBotPhase.dat', 'w') as fich:
            fmt = '{:15}'*7 + '\n'
            fich.write(fmt.format(' phi', 'kx', 'Ra', 'Pmax', 'Umax', 'Tmax', 'Wmax'))
            fmt = '{:15.3e}'*6 + '{:15.3}' + '\n'
            for i in range(nphi):
                fich.write(fmt.format(phinum[i], kwn[i], ram[i], pmax[i],
                                      umax[i], tmax[i], wmax[i]))
        # Now plot
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Ra
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                               label=r'$\frac{27\pi^4}{4}$')
        # general case
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE, label=r'$\Phi^+=\infty$, varying $\Phi^-$')
        col1 = p1.get_color()
        # axe[0].semilogx(phinum, ram2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=7)
        # kx
        # classical RB case
        axe[1].semilogx([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Free top, phase change at bottom
        axe[1].semilogx(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'$\Phi^+=\infty$, varying $\Phi^-$')
        # axe[1].loglog(phinum, kwn2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^-$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_VaryingPhiBotFreeTop.pdf", format='PDF')
        plt.close(fig)
        # plot Pmax, Umax, Wmax and (24phi - Ra) as function of phi
        fig, axe = plt.subplots(3, 1, sharex=True)
        axe[0].semilogx(phinum, np.abs(pmax), marker='.', linestyle='-')
        axe[0].legend(loc='upper left', fontsize=FTSZ)
        axe[0].set_ylabel(r'$P_{max}$', fontsize=FTSZ)
        axe[1].semilogx(phinum, np.abs(umax), marker='.', linestyle='-')
        axe[1].legend(loc='upper left', fontsize=FTSZ)
        axe[1].set_ylabel(r'$U_{max}$', fontsize=FTSZ)
        axe[2].semilogx(phinum, np.abs(wmax), marker='.', linestyle='-')
        axe[2].legend(loc='upper left', fontsize=FTSZ)
        axe[2].set_ylabel(r'$W_{max}$', fontsize=FTSZ)
        axe[2].set_xlabel(r'$\Phi$', fontsize=FTSZ)
        # axe[3].set_ylim([1.e-6, 1.e2])
        plt.savefig('Phi_ModeMaxFreeTopPhaseBot.pdf', format='PDF')
        plt.close(fig)

    DIFFERENT_PHI = False
    # Compute solution properties as function of both phitop and phibot
    if DIFFERENT_PHI:
        # Botphase = False and varying phitop
        rrm, kkx = ra_ks(500, 2, NCHEB,
                        phase_change_top=True, phase_change_bot=False,
                        bcsu=np.array([[1, 0, 0], [0, 1, 0]], float),
                        phitop=phinum[0])
        ram = [rrm]
        kwn = [kkx]
        print(ram, kwn)
        for i, phi in enumerate(phinum[1:]):
            rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
                        phase_change_top=True, phase_change_bot=False,
                        bcsu=np.array([[1, 0, 0], [0, 1, 0]], float),
                        phitop=phi)
            print(i, phi, rrm, kkx)
            ram = np.append(ram, rrm)
            kwn = np.append(kwn, kkx)
        # now keep top to the lowest value and change phibot
        ram2 = [ram[-1]]
        kwn2 = [kwn[-1]]
        print(ram2, kwn2)
        for i, phi in enumerate(phinum):
            rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
                        phase_change_top=True, phase_change_bot=True,
                        phitop=phinum[-1], phibot=phi)
            print(i, phi, rrm, kkx)
            ram2 = np.append(ram2, rrm)
            kwn2 = np.append(kwn2, kkx)

        # Now plot
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Ra
        # Theoretical prediction for translation
        axe[0].semilogx(phinum, 24*phinum, '--', c='k', label='Translation mode')
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                               label=r'$\frac{27\pi^4}{4}$')
        # general case
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE, label=r'$\Phi^-=\infty$, varying $\Phi^+$')
        p2, = axe[0].semilogx(phinum, ram2[1:], 'o', markersize=MSIZE, label='Varying $\Phi^-$, $\Phi^+=10^{-2}$')
        col1 = p1.get_color()
        col2 = p2.get_color()
        # axe[0].semilogx(phinum, ram2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=7)
        # kx
        # classical RB case
        axe[1].semilogx([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Free bottom, phase change at top
        axe[1].semilogx(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'$\Phi^-=\infty$, varying $\Phi^+$')
        # Gradually openning bottom
        axe[1].semilogx(phinum, kwn2[1:], 'o', markersize=MSIZE, c=col2,
                    label='Varying $\Phi^-$, $\Phi^+=10^{-2}$')
        # axe[1].loglog(phinum, kwn2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^-,\quad \Phi^+$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_VaryingPhiBotTop.pdf", format='PDF')
        plt.close(fig)

STAB_TRANSLATION = False
# Computes the linear stability of the steady translation mode
if STAB_TRANSLATION:
    phib = 0.01
    phit = phib
    # epsilon = np.flipud(np.linspace(0, 1, 4))
    epsilon = np.array([5])
    wkn = np.power(10, np.linspace(-1, 4, 100))
    sigma = np.zeros(wkn.shape)
    NCHEB = 30
    # rmin, kmin = ra_ks(ran, wkn, NCHEB,
                   # phase_change_top=True,
                   # phase_change_bot=True,
                   # phitop = phit,
                   # phibot = phib,
                   # translation=True)
    # print('eps = ', (rmin-rtr)/rtr, 'kmin = ', kmin)
    # rao = search_ra(wkn, ran, NCHEB,
                   # phase_change_top=True,
                   # phase_change_bot=True,
                   # phitop = phit,
                   # phibot = phib,
                   # translation=True)
    # print('eps = ', (rao-rtr)/rtr)
    axe = plt.subplot()
    for j, eps in enumerate(epsilon):
        rtr = 12*(phib+phit)
        ran = rtr*(1+eps)
        for i, kxn in enumerate(wkn):
            sigma[i] = eigval_cartesian(kxn, ran, NCHEB,
                                        phase_change_top=True,
                                        phase_change_bot=True,
                                        phitop = phit,
                                        phibot = phib,
                                        translation=True)

        axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2f$' %(eps))
        axe.set_xlabel(r'$k$', fontsize=FTSZ)
        axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        # axe.set_ylim((-500, 1500))
        # axe.set_ylim(bottom=-500)

    plt.savefig('sigmaRa'+np.str(eps)+'N'+np.str(NCHEB)+'Top'+np.str(phit).replace('.', '-') + 'Bot' + np.str(phib).replace('.', '-')+'.pdf')

