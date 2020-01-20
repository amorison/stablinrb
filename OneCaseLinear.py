#!/usr/bin/env python3
"""
Finds critical Rayleigh number for only one case.

Minimal example which computes the minimum critical Rayleigh 
number and the associated wavenumber. 
Additionnally, it plots the critical Ra as function of wavenumber 
around the minimal value and the fastest growing mode.
"""

from stablinrb.analyzer import LinearAnalyzer
from stablinrb.physics import PhysicalProblem
import stablinrb.plotting as plotting

# Define the physical problem.
# Adjust the parameters and conditions here.
# See physics.py for all the options
pblm = PhysicalProblem(
    gamma = None, # aspect ratio for spherical shell. None for cartesian (default).
    freeslip_top=True, # mechanical BC at top. True is default.
    freeslip_bot=True # mechanical BC at bottom. True is default.
    )

ana = LinearAnalyzer(pblm, ncheb=10)
# Find the critical Ra and wavenumber and print.
ra_c, harm_c = ana.critical_ra()
print('Rac, kc = ', ra_c, harm_c)

# Compute figures if needed.
plotting.plot_fastest_mode(ana, harm_c, ra_c)
plotting.plot_ran_harm(ana, harm_c)

