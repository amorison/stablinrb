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
MSIZE = 3
# gamma = 0.7
eta_c = None
pblm = PhysicalProblem(
    gamma=None,
    phi_top=1e1,
    phi_bot=1e1,
    freeslip_top=True,
    freeslip_bot=False,
    eta_r = visco_Arrhenius(eta_c, gamma) if eta_c is not None else None,
    ref_state_translation=False)

NON_LINEAR = False
ra_comp = None

if NON_LINEAR:
    ana = NonLinearAnalyzer(pblm, ncheb=20, nnonlin=4)
    harm_c, ray, mode, moyt, qtop = ana.nonlinana()
    print('Rayleigh = ', ray)
    print('moyt = ', moyt)
    print('qtop = ', qtop)
    nterms = qtop.shape[0]
    eps = np.linspace(0, 1, num=10)
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
    plotting.plot_fastest_mode(ana, harm_c, ra_c, ra_comp, plot_theory=True)
    plotting.plot_ran_harm(ana, harm_c, ra_comp)
    plotting.plot_viscosity(pblm)
