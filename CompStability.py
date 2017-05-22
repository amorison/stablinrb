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
t_crystal = 3500  # temperature solidus as d_crystal
r_int = r_earth - d_crystal
r_ext = lambda h: r_int + h
rho = 4e3
g = 10
alpha = 1e-5  # dV/dT /V
beta = (3.58 - 5.74) / 3.58  # dV/dc /V
heat_capacity = 1e3  # in SMO
latent_heat = 4e5  # J/kg
emissivity = 0.8
stefan_boltzmann = 5.67e-8
temp_inf = 255
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
dtmelt_dp = 2e-8
dtmelt_dc = -1e2
grad_temp = lambda r, h: -rho * g * dtmelt_dp + \
    dtmelt_dc * (composition(r, h) * 3 * (1 - part) *
                 r**2 / (r_earth**3 - r**3))
delta_temp = lambda h: -integ.quad(
    lambda r: grad_temp(r, h), r_int, r_ext(h))[0]

eta_vals = np.array(range(15, 17))  # 19
h_crystal_vals = np.linspace(100e3, 1300e3, 10)  # 50

# non-dimensional variables
gamma = lambda h: r_int / r_ext(h)
rayleigh = lambda h, eta: \
    rho * g * alpha * delta_temp(h) * r_ext(h)**3 / (eta * kappa)
ra_comp = lambda h, eta: \
    rho * g * beta * r_ext(h)**3 / (eta * kappa)
phi_top = None
phi_bot = None

# compute time to crystallize given thickness of solid mantle
def surf_temp(h):
    """Surface temperature determined by fixing
    the boundary layer Ra# at top of the SMO"""
    temp_bot_smo = t_crystal - delta_temp(h)
    temp_surf_pot = temp_bot_smo * np.exp(- alpha * g * (r_earth - r_ext(h))
                                          / heat_capacity)
    ra_bnd = 1e3
    eta_smo = 1e-1
    tsurf_func = lambda ts: ((temp_surf_pot - ts)**(4/3) -
                             emissivity * stefan_boltzmann /
                             (kappa * rho * heat_capacity) *
                             (kappa * eta_smo * ra_bnd / alpha * rho * g)**(1/3) *
                             (ts**4 - temp_inf**4))
    return opt.fsolve(tsurf_func, (temp_surf_pot - temp_inf) / 2)[0]

#for h_crystal in h_crystal_vals:
#    print(t_crystal - delta_temp(h_crystal), surf_temp(h_crystal))

#crystallized = [0]
#time = [0]
#dtime = 1e3
#temp_top = t_crystal
#while crystallized[-1] < 1300e3:
#    h = crystallized[-1]
#    dtop = ((rho * latent_heat * (r_ext(h)**3 - r_int**3)/3 -
#             emissivity * stefan_boltzmann * r_earth**2 *
#             (surf_temp(h)**4 - temp_inf**4)) * dtime /
#            (rho * heat_capacity * (r_earth**3 - r_ext(h)**3)))
#    temp_top += dtop
#    print(temp_top)
#    crystallized.append(opt.fsolve(lambda h:delta_temp(h) - (t_crystal - temp_top), h)[0])
#    temp_top = t_crystal - delta_temp(crystallized[-1])
#    print(temp_top, crystallized[-1])
#    time.append(time[-1] + dtime)

#crystal_time = []
#for h in h_crystal_vals:
#    crystal_time.append(
#      (rho * latent_heat * (r_ext(h)**3 - r_int**3)/3 -
#         emissivity * stefan_boltzmann * r_earth**2 *
#         (surf_temp(h)**4 - temp_inf**4)) /
#        (rho * heat_capacity * (r_earth**3 - r_ext(h)**3)
#         * (t_crystal - delta_temp(h))))

ana = LinearAnalyzer(PhysicalProblem(phi_top=phi_top,
                                     phi_bot=phi_bot),
                     ncheb=50)

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
    axis.semilogy(h_crystal_vals/1e3, np.array(tau_vals[eta])/(86400),
                     label=r'$\eta=10^{%d}$, $l=%d$' %(eta, harm_vals[eta][0]))
#axis.semilogy(h_crystal_vals/1e3, 1/np.array(crystal_time)/86400)
axis.set_xlabel(r'Crystallized mantle thickness $h$ (km)', fontsize=FTSZ)
axis.set_ylabel(r'Destabilization time scale $\tau$ (days)', fontsize=FTSZ)
axis.legend(loc='upper right', fontsize=FTSZ)
axis.tick_params(labelsize=FTSZ)
plt.tight_layout()
plt.savefig('DestabilizationTime.pdf', format='PDF')
