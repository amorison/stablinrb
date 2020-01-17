#!/usr/bin/env python3
"""
Finds critical Rayleigh number for only one case.

Minimal example which computes the minimum critical Rayleigh 
number and the associated wavenumber. 
Additionnally, it plots the critical Ra as function of wavenumber 
around the minimal value and the fastest growing mode.
"""

from analyzer import LinearAnalyzer
from physics import PhysicalProblem
import plotting

# Define the physical problem.
# Adjust the parameters and conditions here.
pblm = PhysicalProblem(
    freeslip_top=True,
    freeslip_bot=True)

ana = LinearAnalyzer(pblm, ncheb=10)
# Find the critical Ra and wavenumber and print.
ra_c, harm_c = ana.critical_ra()
print('Rac, kc = ', ra_c, harm_c)

# Compute figures if needed.
plotting.plot_fastest_mode(ana, harm_c, ra_c)
plotting.plot_ran_harm(ana, harm_c)

