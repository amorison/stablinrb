#!/usr/bin/env python3
"""
Finds critical Rayleigh number for only one case.
"""
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo, visco_Arrhenius
from misc import normalize_modes
import plotting

# Font and markers size
FTSZ = 11
MSIZE = 2
gamma = 0.7
eta_c = None

PHI = 1e-2

pblm = PhysicalProblem(
    gamma=gamma,
    phi_top=PHI,
    phi_bot=PHI,
    freeslip_top=True,
    freeslip_bot=True,
    eta_r = visco_Arrhenius(eta_c, gamma) if eta_c is not None else None,
    ref_state_translation=False)

NON_LINEAR = False
ra_comp = None

if NON_LINEAR:
    ana = NonLinearAnalyzer(pblm, ncheb=30, nnonlin=2)
    harm_c, ray, mode, moyt, qtop = ana.nonlinana()
    print('kc = ', harm_c, np.pi / np.sqrt(2))
    print('Rayleigh = ', ray)
    print('moyt = ', moyt)
    print('qtop = ', qtop)
    print('qtop2 =', qtop[2], 0.25 / (np.pi ** 2 + harm_c ** 2))
    print('coef qtop2 = ', ray[0] * qtop[2] / ray[2])
    nterms = qtop.shape[0]
    eps = np.linspace(0, 10, num=30)
    vdm = np.vander(eps, nterms, increasing=True)
    rayl = np.dot(vdm, ray)
    nuss = np.dot(vdm, qtop)
    meant = np.dot(vdm, moyt)
    fig, axe = plt.subplots(2, 1, sharex=True)
    axe[0].plot(rayl, nuss)
    axe[0].set_ylabel('Nusselt number', fontsize=FTSZ)
    axe[0].set_xlabel('Rayleigh number', fontsize=FTSZ)
    axe[1].plot(rayl, meant)
    axe[1].set_xlabel('Rayleigh number', fontsize=FTSZ)
    axe[1].set_ylabel('Mean T', fontsize=FTSZ)
    plt.savefig('Ra-Nu-Tmean.pdf', format='PDF')
else:
    ana = LinearAnalyzer(pblm, ncheb=20)
    ra_c, harm_c = ana.critical_ra(ra_comp=ra_comp)
    print('Rac, kc = ', ra_c, harm_c)
    plotting.plot_fastest_mode(ana, harm_c, ra_c, ra_comp, plot_theory=False)
    plotting.plot_ran_harm(ana, harm_c, ra_comp)
    if eta_c is not None:
        plotting.plot_viscosity(pblm)

