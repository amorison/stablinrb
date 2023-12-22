"""Find critical Rayleigh number and associated mode.

Minimal example which computes the minimum critical Rayleigh number and the
associated wavenumber in cartesian geometry with free-slip boundaries.
Additionnally, it plots the critical Ra as function of wavenumber around the
minimal value and the fastest growing mode.
"""

import numpy as np

import stablinrb.plotting as plotting
from stablinrb.cartesian import CartStability
from stablinrb.physics import FreeSlip
from stablinrb.ref_prof import DiffusiveProf, Dirichlet
from stablinrb.rheology import Arrhenius

Tscale = 18.420680743952367
vscale = np.exp(-Tscale)

ana = CartStability(
    chebyshev_degree=50,
    bc_mom_top=FreeSlip(),
    bc_mom_bot=FreeSlip(),
    rheology=Arrhenius(
        vscale, Tscale, 1, DiffusiveProf(bcs_top=Dirichlet(0), bcs_bot=Dirichlet(1))
    ),
)

# Find the critical Ra and wavenumber and print.
ra_c, harm_c = ana.critical_ra()
print("Rac, kc = ", ra_c, harm_c)

plotting.plot_fastest_mode_cart(ana, harm_c, ra_c)
plotting.plot_ran_harm_cart(ana, harm_c)
