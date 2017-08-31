#!/usr/bin/env python3
"""
Crystallization and cumulate destabilization time scales of a SMO
"""
import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt
import matplotlib.pyplot as plt
from misc import savefig
from analyzer import LinearAnalyzer
from physics import PhysicalProblem

# Font and markers size
FTSZ = 14
MSIZE = 5

# all in base units
r_earth = 6371e3
r_cmb = 3400e3
h_mantle = r_earth - r_cmb
d_crystal = 1500e3
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
tau = lambda s: h_mantle**2 / (kappa * s)
# composition
part = 0.6  # partitioning coefficient
c_feo_liq0 = 0.1  # part & c_feo_liq0 from Andrault et al, 2011
c_0 = part * c_feo_liq0
# r at which c = 1
r_eut = (r_int**3 * c_0**(1 / (1-part)) +
         r_earth**3 * (1 - c_0**(1 / (1-part))))**(1/3)
composition = lambda r: \
    (c_0 * ((r_earth**3 - r_int**3)
            / (r_earth**3 - r**3))**(1-part)
     if r < r_eut else 1)
dtmelt_dp = 2e-8
dtmelt_dc = -1e3


def grad_temp(r, rm_isen=True):
    """Temperature gradient

    Total if rm_isen is False,
    superadiabatic otherwise"""
    total = -rho * g * dtmelt_dp + \
        dtmelt_dc * (composition(r) * 3 * (1 - part) *
                     r**2 / (r_earth**3 - r**3)
                     if r < r_eut else 0)
    if rm_isen:
        total += t_crystal * isen * np.exp(- isen * (r - r_int))
    return total


delta_temp = lambda h, rm_isen=True: -integ.quad(
    lambda r: grad_temp(r, rm_isen), r_int, r_ext(h))[0]

h_crystal_max = r_eut - r_int - 150e3

eta_vals = np.array(range(16, 18))
h_crystal_vals = np.logspace(np.log10(5) + 3, np.log10(h_crystal_max), 50)
phi_vals = [(None, None), (None, 1e-2), (1e-2, 1e-2)]

# dimensional radius from non-dimensional one
radim = lambda r, h: (r - 1) * h + r_int
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
    temp_bot_smo = t_crystal - delta_temp(h, False)
    temp_surf_pot = temp_bot_smo * np.exp(- isen * (r_earth - r_ext(h)))
    ra_bnd = 1e3
    eta_smo = 1e-1
    tsurf_func = lambda ts: ((temp_surf_pot - ts)**(4/3) -
                             emissivity * stefan_boltzmann /
                             (kappa * rho * heat_capacity) *
                             (kappa * eta_smo * ra_bnd / alpha * rho * g)**(1/3) *
                             (ts**4 - temp_inf**4))
    return opt.fsolve(tsurf_func, (temp_surf_pot - temp_inf) / 2)[0]

def update_ana_thickness(ana, h_crystal):
    """Update analyzer with new thickness"""
    ana.phys.gamma = gamma(h_crystal)
    ana.phys.composition = np.vectorize(
        lambda r: composition(radim(r, h_crystal)))
    ana.phys.grad_ref_temperature = np.vectorize(
        lambda r: grad_temp(radim(r, h_crystal)) *
            h_crystal / delta_temp(h_crystal))

def cooling_time():
    """Compute time to evacuate latent heat and cool down SMO

    Based on grey body radiation and fixed boundary layer Ra
    """
    crystallized = [0]
    time = [0]
    dtime = None
    while crystallized[-1] < h_crystal_max:
        dtime = 1e5 if dtime is None else 3e8  # to have first point earlier
        h = crystallized[-1]
        temp_top = t_crystal - delta_temp(h, False)
        gtemp_top = grad_temp(r_ext(h), False)
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
    return np.array(crystallized), np.array(time)


def _phi_col_lbl(phi_top, phi_bot):
    if phi_bot is None:
        phi_str = r'$\Phi^+=10^{-2}$' if phi_top else 'closed'
        col = 'g' if phi_top else 'b'
    else:
        phi_str = r'$\Phi^+=\Phi^-=10^{-2}$'
        col = 'r'
    return col, phi_str


def plot_destab(ana, crystallized, time):
    """Plot destabilization time scale as function of solid thickness"""
    figt, axt = plt.subplots(1, 1)
    figl, axl = plt.subplots(1, 1)

    for phi_bot, phi_top in phi_vals:
        ana.phys.phi_top = phi_top
        ana.phys.phi_bot = phi_bot
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)

        tau_vals = {}
        harm_vals = {}
        for eta in eta_vals:
            harm = 1
            style = '-' if eta == 17 else '--'
            tau_vals[eta] = []
            harm_vals[eta] = []
            for h_crystal in h_crystal_vals:
                update_ana_thickness(ana, h_crystal)
                sigma, harm = ana.fastest_mode(rayleigh(h_crystal, 10**eta),
                                               ra_comp(h_crystal, 10**eta),
                                               harm)
                print(phi_top, phi_bot, eta, sigma, harm, rayleigh(h_crystal, 10**eta))
                tau_vals[eta].append(np.real(tau(sigma)))
                harm_vals[eta].append(harm)
            axt.semilogy(h_crystal_vals/1e3, np.array(tau_vals[eta]),
                         label=r'$\eta=10^{%d}$, %s' %(eta, phi_str),
                         color=col, linestyle=style)
            axl.semilogy(h_crystal_vals/1e3, np.array(harm_vals[eta]),
                         label=r'$\eta=10^{%d}$, %s' %(eta, phi_str),
                         color=col, linestyle=style)
    axt.semilogy(crystallized / 1e3, time, color='k')
    for duration, name in [(86400, '1d'), (3.15e7, '1y'),
                           (3.15e10, '1ky'), (3.15e13, '1My')]:
        axt.semilogy([0, h_crystal_max/1e3], [duration]*2,
                     color='k', linestyle=':', linewidth=1)
        axt.text(-h_crystal_max/2e4, duration, name, va='center')
    axt.set_ylabel(r'Destabilization time scale $\tau$ (s)', fontsize=FTSZ)
    axl.set_ylabel(r'Harmonic degree $l$', fontsize=FTSZ)
    for axis in (axt, axl):
        axis.set_xlabel(r'Crystallized mantle thickness $h$ (km)', fontsize=FTSZ)
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
    savefig(figt, 'DestabilizationTimeCryst.pdf')
    savefig(figl, 'DestabilizationTimeCryst_l.pdf')


def _sigma(h_cryst, eta, ana):
    """Return dimensionless growth rate"""
    update_ana_thickness(ana, h_cryst)
    sigma, _ = ana.fastest_mode(rayleigh(h_cryst, 10**eta),
                                ra_comp(h_cryst, 10**eta))
    return sigma


def _time_diff(h_cryst, eta, ana, crystallized, time):
    """Compute time difference between destab and cooling time"""
    sigma = _sigma(h_cryst, eta, ana)
    destab = np.real(tau(sigma))
    cooling = np.interp(h_cryst, crystallized, time)
    return cooling - destab


def plot_min_time(ana, crystallized, time, eps=1e-3):
    """Research of time at which destab = cooling time"""
    fig, axt = plt.subplots(1, 1)
    eta_logs = np.linspace(15, 18, 10)
    for phi_bot, phi_top in phi_vals:
        ana.phys.phi_top = phi_top
        ana.phys.phi_bot = phi_bot
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)
        tau_vals = []
        for eta in eta_logs:
            h_min = 20e3
            while _sigma(h_min, eta, ana) < 0:
                h_min *= 4
            while _time_diff(h_min, eta, ana, crystallized, time) > 0:
                h_min /= 2
                if _sigma(h_min, eta, ana) < 0:
                    # could be finer
                    raise ValueError(
                        'Destab=cooling time has no solution {} {}'
                        .format(phi_str, eta))
            h_max = h_min * 2
            while _time_diff(h_max, eta, ana, crystallized, time) < 0:
                h_max *= 2
            while (h_max - h_min) / h_max > eps:
                h_cryst = np.sqrt(h_min * h_max)
                if _time_diff(h_cryst, eta, ana, crystallized, time) < 0:
                    h_min = h_cryst
                else:
                    h_max = h_cryst
            tau_val = np.interp(h_cryst, crystallized, time)
            tau_vals.append(tau_val)
            print(eta, tau_val)
        axt.loglog(10**eta_logs, np.array(tau_vals) / 3.15e10,
                   color=col, label=phi_str)
    axt.set_xlabel(r'Viscosity $\eta$')
    axt.set_ylabel(r'Time (kyr)')
    axt.legend()
    tmin, tmax = axt.get_ylim()
    hmin = np.interp(tmin*3.15e10, time, crystallized)
    hmax = np.interp(tmax*3.15e10, time, crystallized)
    axh = axt.twinx()
    axh.set_yscale('log')
    axh.set_ylim(hmin * 1e-3, hmax * 1e-3)
    axh.set_ylabel(r'Crystallized thickness (km)')
    savefig(fig, 'CoolingDestab.pdf')


def plot_cooling_time(crystallized, time):
    """Plot cooling time"""
    crt = crystallized / 1000
    tim = time / 3.15e10
    fig, axis = plt.subplots(1, 1)
    axis.plot(crt, tim)
    axis.plot([crt[0], crt[-1]], [tim[0], tim[-1]],
              linestyle='--', lw=0.5, color='k')
    slope_init = (tim[1] - tim[0]) / (crt[1] - crt[0])
    t_final = slope_init * (crt[-1] - crt[0]) + tim[0]
    axis.plot([crt[0], crt[-1]], [tim[0], t_final],
              linestyle='--', lw=0.5, color='k')
    axis.set_xlabel('Crystallized thickness (km)')
    axis.set_ylabel('Time (kyr)')
    savefig(fig, 'CoolingTime.pdf')


if __name__ == '__main__':
    crystallized, time = cooling_time()
    plot_cooling_time(crystallized, time)

    gam_smo = gamma(crystallized[1:])
    gamt_smo = (crystallized[1:] / h_mantle)
    w_smo = ((crystallized[1:] - crystallized[:-1]) / (time[1:] - time[:-1]) *
             crystallized[1:] / kappa)
    gamt_f = lambda g: np.interp(g, gam_smo, gamt_smo)
    w_f = lambda g: np.interp(g, gam_smo, w_smo)
    ana = LinearAnalyzer(
        PhysicalProblem(cooling_smo=(gamt_f, w_f)),
        ncheb=24)

    plot_min_time(ana, crystallized, time)
    plot_destab(ana, crystallized, time)
