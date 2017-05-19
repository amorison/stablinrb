#!/usr/bin/env python3
"""
Calculation of stability from an initial composition and temperature profile.
"""
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import LinearAnalyzer, NonLinearAnalyzer
from physics import PhysicalProblem, compo_smo
from plotting import plot_fastest_mode, plot_ran_harm
from misc import normalize_modes

# Font and markers size
FTSZ = 14
MSIZE = 5

# all in base units
r_earth = 6371e3
d_crystal = 1500e3
r_int = r_earth - d_crystal
r_ext = lambda h: r_int + h
rho = 4e3
g = 10
alpha = 1e-5  # dV/dT /V
beta = -3.58 / 5.74  # dV/dc /V
kappa = 1e-6
tau = lambda s, h: r_ext(h)**2 / (kappa * s)
# composition
part = 0.8  # partitioning coefficient
lam = 0.999  # R_eutectic / R_earth
c_0 = ((1 - lam**3) * r_earth**3 / (r_earth**3 - r_int**3))**(1 - part)
# always in case r<R_eutectic
composition = lambda r, h, c_0=c_0: \
    c_0 * ((r_earth**3 - r_int**3)
           / (r_earth**3 - r**3))**(1-part)
dtmelt_dp = 1e-7
dtmelt_dc = -1e3
grad_temp = lambda r, h: -rho * g * dtmelt_dp + \
    dtmelt_dc * (composition(r, h) * 3 * (1 - part) *
                 r**2 / (r_earth**3 - r**3))
delta_temp = lambda h: -integ.quad(
    lambda r: grad_temp(r, h), r_int, r_ext(h))[0]

eta_vals = np.array(range(15, 19))
h_crystal_vals = np.linspace(100e3, 1300e3, 50)

# non-dimensional variables
gamma = lambda h: r_int / r_ext(h)
rayleigh = lambda h, eta: \
    rho * g * alpha * delta_temp(h) * r_ext(h)**3 / (eta * kappa)
ra_comp = lambda h, eta: \
    rho * g * beta * r_ext(h)**3 / (eta * kappa)
phi_top = 1e-2
phi_bot = 1e-2

ana = LinearAnalyzer(PhysicalProblem(phi_top=phi_top,
                                     phi_bot=phi_bot),
                     ncheb=150)

fig, axis = plt.subplots(1, 1)
tau_vals = {}
harm_vals = {}
for eta in eta_vals:
    tau_vals[eta] = []
    harm_vals[eta] = []
    for h_crystal in h_crystal_vals:
        ana.phys.gamma = gamma(h_crystal)
        ana.phys.composition = lambda r: composition(r * r_ext(h_crystal),
                                                     h_crystal)
        ana.phys.grad_ref_temperature = \
            lambda r: grad_temp(r * r_ext(h_crystal), h_crystal) * \
                r_ext(h_crystal) / delta_temp(h_crystal)
        sigma, harm = ana.fastest_mode(rayleigh(h_crystal, 10**eta),
                                       ra_comp(h_crystal, 10**eta))
        tau_vals[eta].append(np.real(tau(sigma, h_crystal)))
        harm_vals[eta].append(harm)
    axis.semilogy(h_crystal_vals/1e3, np.array(tau_vals[eta])/1e6,
                     label=r'$\eta=10^{%d}$, $l=%d$' %(eta, harm_vals[eta][0]))
axis.set_xlabel(r'Crystallized mantle thickness $h$ (km)', fontsize=FTSZ)
axis.set_ylabel(r'Destabilization time scale $\tau$ (Myr)', fontsize=FTSZ)
axis.legend(loc='upper right', fontsize=FTSZ)
axis.tick_params(labelsize=FTSZ)
plt.tight_layout()
plt.savefig('DestabilizationTime.pdf', format='PDF')
