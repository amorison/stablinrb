#!/usr/bin/env python3
"""
Finds critical Rayleigh number for only one case.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo, visco_Arrhenius
from misc import normalize_modes
import plotting

# Font and markers size
FTSZ = 14
MSIZE = 5
gamma = 0.7
eta_c = 10**7
pblm = PhysicalProblem(
    gamma=None,
    phi_top=1e4,
    phi_bot=1e4,
    freeslip_top=True,
    freeslip_bot=True,
    eta_r = visco_Arrhenius(eta_c, gamma) if eta_c is not None else None,
    ref_state_translation=False)

NON_LINEAR = True
ra_comp = None

if NON_LINEAR:
    ana = NonLinearAnalyzer(pblm, ncheb=20)
    harm_c, ray, modec, mode20, mode22, glob_val = ana.nonlinana()
    print('Ra_c, Ra2 = ', ray)
    print('globval', glob_val)
else:
    ana = LinearAnalyzer(pblm, ncheb=20)
    ra_c, harm_c = ana.critical_ra(ra_comp=ra_comp)
    print('Rac, kc = ', ra_c, harm_c)
    plotting.plot_fastest_mode(ana, harm_c, ra_c, ra_comp)
    plotting.plot_ran_harm(ana, harm_c, ra_comp)
    plotting.plot_viscosity(pblm)
