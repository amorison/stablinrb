#!/usr/bin/env python3
"""
Calculation of stability from an initial composition and temperature profile.
"""
import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt
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
h_crystal_max = 100e3
t_crystal = 3500  # temperature solidus as d_crystal
r_int = r_earth - d_crystal
r_ext = lambda h: r_int + h
rho = 4e3
g = 10
alpha = 1e-5  # dV/dT /V
beta = (3.58 - 5.74) / 3.58  # dV/dc /V
heat_capacity = 1e3  # in SMO
isen = alpha * g / heat_capacity  # in SMO
latent_heat = 4e5  # J/kg
emissivity = 1.0  # black body
stefan_boltzmann = 5.67e-8
temp_inf = 255
kappa = 1e-6
tau = lambda s, h: h**2 / (kappa * s)
# composition
part = 0.8  # partitioning coefficient
lam = 0.999  # R_eutectic / R_earth
c_0 = ((1 - lam**3) * r_earth**3 / (r_earth**3 - r_int**3))**(1 - part)
# always in case r<R_eutectic
composition = lambda r, h, c_0=c_0: \
    c_0 * ((r_earth**3 - r_int**3)
           / (r_earth**3 - r**3))**(1-part)
dtmelt_dp = 2e-8
dtmelt_dc = -1e2
grad_temp = lambda r, h: -rho * g * dtmelt_dp + \
    dtmelt_dc * (composition(r, h) * 3 * (1 - part) *
                 r**2 / (r_earth**3 - r**3))
delta_temp = lambda h: -integ.quad(
    lambda r: grad_temp(r, h), r_int, r_ext(h))[0]

eta_vals = np.array(range(16, 18))
h_crystal_vals = np.logspace(np.log10(5) + 3, np.log10(h_crystal_max), 50)
phi_vals = [(None, None), (None, 1e-2), (1e-2, 1e-2)]

# non-dimensional variables
gamma = lambda h: r_int / r_ext(h)
rayleigh = lambda h, eta: \
    rho * g * alpha * delta_temp(h) * h**3 / (eta * kappa)
ra_comp = lambda h, eta: \
    rho * g * beta * h**3 / (eta * kappa)

# compute time to crystallize given thickness of solid mantle
def surf_temp(h):
    """Surface temperature determined by fixing
    the boundary layer Ra# at top of the SMO"""
    temp_bot_smo = t_crystal - delta_temp(h)
    temp_surf_pot = temp_bot_smo * np.exp(- isen * (r_earth - r_ext(h)))
    ra_bnd = 1e3
    eta_smo = 1e-1
    tsurf_func = lambda ts: ((temp_surf_pot - ts)**(4/3) -
                             emissivity * stefan_boltzmann /
                             (kappa * rho * heat_capacity) *
                             (kappa * eta_smo * ra_bnd / alpha * rho * g)**(1/3) *
                             (ts**4 - temp_inf**4))
    return opt.fsolve(tsurf_func, (temp_surf_pot - temp_inf) / 2)[0]

crystallized = [0]
time = [0]
dtime = None
while crystallized[-1] < h_crystal_max:
    dtime = 1e5 if dtime is None else 3e8  # to have first point earlier
    h = crystallized[-1]
    temp_top = t_crystal - delta_temp(h)
    gtemp_top = grad_temp(r_ext(h), h)
    heat_to_extract = rho * heat_capacity * (gtemp_top + temp_top * isen)
    expis = np.exp(isen * (r_ext(h) - r_earth))
    heat_to_extract *= (r_earth**2 * expis - r_ext(h)**2) / isen + \
        2 * (r_earth * expis - r_ext(h)) / isen**2 + \
        2 * (expis - 1) / isen**3
    heat_to_extract += rho * latent_heat * r_ext(h)**2 + \
        rho * heat_capacity * temp_top * r_ext(h)**2
    gray_body = emissivity * stefan_boltzmann * r_earth**2 * \
        (surf_temp(h)**4 - temp_inf**4)
    drad = gray_body * dtime / heat_to_extract
    crystallized.append(h + drad)
    time.append(time[-1] + dtime)
    if len(crystallized) % 1000 == 0:
        print(surf_temp(h), temp_top, crystallized[-1]/1e3, time[-1]/3.15e7)

fig, axis = plt.subplots(1, 1)

for phi_bot, phi_top in phi_vals:
    ana = LinearAnalyzer(PhysicalProblem(phi_top=phi_top,
                                         phi_bot=phi_bot),
                         ncheb=15)
    if phi_bot is None:
        phi_str = r'$\Phi^+=10^{-2}$' if phi_top else 'closed'
        col = 'g' if phi_top else 'b'
    else:
        phi_str = r'$\Phi^+=\Phi^-=10^{-2}$'
        col = 'r'

    tau_vals = {}
    for eta in eta_vals:
        harm = 1
        style = '-' if eta == 17 else '--'
        tau_vals[eta] = []
        for h_crystal in h_crystal_vals:
            ana.phys.gamma = gamma(h_crystal)
            ana.phys.composition = lambda r: composition(r * h_crystal,
                                                         h_crystal)
            ana.phys.grad_ref_temperature = \
                lambda r: grad_temp(r * h_crystal, h_crystal) * \
                    h_crystal / delta_temp(h_crystal)
            sigma, harm = ana.fastest_mode(rayleigh(h_crystal, 10**eta),
                                           ra_comp(h_crystal, 10**eta),
                                           harm)
            #plot_fastest_mode(ana, harm, rayleigh(h_crystal, 10**eta),
            #                  ra_comp(h_crystal, 10**eta), name='N150')
            #sigma = ana.eigval(harm, rayleigh(h_crystal, 10**eta),
            #                   ra_comp(h_crystal, 10**eta))
            print(phi_top, phi_bot, eta, sigma, harm, rayleigh(h_crystal, 10**eta))
            tau_vals[eta].append(np.real(tau(sigma, h_crystal)))
        axis.semilogy(h_crystal_vals/1e3, np.array(tau_vals[eta]),
                      label=r'$\eta=10^{%d}$, %s' %(eta, phi_str),
                      color=col, linestyle=style)
axis.semilogy(np.array(crystallized) / 1e3, np.array(time), color='k')
for duration, name in [(86400, '1d'), (3.15e7, '1y'),
                       (3.15e10, '1ky'), (3.15e13, '1My')]:
    axis.semilogy([0, h_crystal_max/1e3], [duration]*2,
                  color='k', linestyle=':', linewidth=1)
    axis.text(-h_crystal_max/2e4, duration, name, va='center')
axis.set_xlabel(r'Crystallized mantle thickness $h$ (km)', fontsize=FTSZ)
axis.set_ylabel(r'Destabilization time scale $\tau$ (s)', fontsize=FTSZ)
axis.legend(
    [plt.Line2D((0,1),(0,0), color='k'),
     plt.Line2D((0,1),(0,0), color='b', linestyle='--'),
     plt.Line2D((0,1),(0,0), color='b', linestyle='-'),
     plt.Line2D((0,1),(0,0), color='b'),
     plt.Line2D((0,1),(0,0), color='g'),
     plt.Line2D((0,1),(0,0), color='r')],
    ['Cooling time', '$\eta=10^{16}$', '$\eta=10^{17}$',
     'closed', r'$\Phi^+=10^{-2}$', r'$\Phi^+=\Phi^-=10^{-2}$'],
    loc='upper right', fontsize=FTSZ)
axis.tick_params(labelsize=FTSZ)
plt.tight_layout()
plt.savefig('DestabilizationTimeCryst.pdf', format='PDF')
