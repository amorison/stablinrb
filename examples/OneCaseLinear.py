#!/usr/bin/env python3
"""Find critical Rayleigh number and associated mode.

Minimal example which computes the minimum critical Rayleigh number and the
associated wavenumber in cartesian geometry with free-slip boundaries.
Additionnally, it plots the critical Ra as function of wavenumber around the
minimal value and the fastest growing mode.
"""

import stablinrb.plotting as plotting
from stablinrb.analyzer import LinearAnalyzer
from stablinrb.geometry import Cartesian
from stablinrb.physics import FreeSlip, PhysicalProblem

pblm = PhysicalProblem(
    geometry=Cartesian(),
    bc_mom_top=FreeSlip(),
    bc_mom_bot=FreeSlip(),
)

ana = LinearAnalyzer(pblm, chebyshev_degree=10)
# Find the critical Ra and wavenumber and print.
ra_c, harm_c = ana.critical_ra()
print("Rac, kc = ", ra_c, harm_c)

plotting.plot_fastest_mode(ana, harm_c, ra_c)
plotting.plot_ran_harm(ana, harm_c)
