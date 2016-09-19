#!/usr/bin/env python3
"""
Finds critical Rayleigh number for only one case.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo
from plotting import plot_fastest_mode, plot_ran_harm
from misc import normalize_modes

# Font and markers size
FTSZ = 14
MSIZE = 5

pblm = PhysicalProblem(
    gamma=0.7,
    phi_top=None,
    phi_bot=1e-2,
    freeslip_top=True,
    freeslip_bot=True,
    eta_r = None,  # lambda r: 1,
    ref_state_translation=False)

NON_LINEAR = False
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
    plot_fastest_mode(ana, harm_c, ra_c, ra_comp)
    plot_ran_harm(ana, harm_c, ra_comp)
