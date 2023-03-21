"""Find critical Rayleigh number and associated mode.

Minimal example which computes the minimum critical Rayleigh number and the
associated wavenumber in spherical geometry with free-slip boundaries.
Additionnally, it plots the critical Ra as function of wavenumber around the
minimal value and the fastest growing mode.
"""

import stablinrb.plotting as plotting
from stablinrb.spherical import SphStability

ana = SphStability(
    chebyshev_degree=10,
    gamma=0.5,
)

# Find the critical Ra and wavenumber and print.
ra_c, harm_c = ana.critical_ra()
print("Rac, kc = ", ra_c, harm_c)

plotting.plot_fastest_mode_sph(ana, harm_c, ra_c)
plotting.plot_ran_harm_sph(ana, harm_c)
