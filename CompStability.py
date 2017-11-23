#!/usr/bin/env python3
"""
Crystallization and cumulate destabilization time scales of a SMO
"""
import numpy as np
import matplotlib.pyplot as plt
from misc import savefig
from analyzer import LinearAnalyzer
from physics import PhysicalProblem
from planets import EARTH as pnt

# Font and markers size
FTSZ = 14
MSIZE = 5

COMPO_EFFECTS = True

# compute time to crystallize given thickness of solid mantle

def update_thickness(ana, pnt, h_crystal):
    """Update analyzer and planet with new thickness"""
    pnt.h_crystal = h_crystal
    ana.phys.gamma = pnt.gamma
    if COMPO_EFFECTS:
        ana.phys.composition = np.vectorize(
            lambda r: pnt.composition(pnt.radim(r)))
    ana.phys.grad_ref_temperature = np.vectorize(
        lambda r: pnt.grad_temp(pnt.radim(r)) *
            pnt.h_crystal / pnt.delta_temp())


def _phi_col_lbl(phi_top, phi_bot):
    if phi_bot is None:
        phi_str = r'$\Phi^+=10^{-2}$' if phi_top else 'closed'
        col = 'g' if phi_top else 'b'
    else:
        phi_str = r'$\Phi^+=\Phi^-=10^{-2}$'
        col = 'r'
    return col, phi_str


def plot_destab(pnt, ana, crystallized, time):
    """Plot destabilization time scale as function of solid thickness"""
    figt, axt = plt.subplots(1, 1)
    figl, axl = plt.subplots(1, 1)
    pnt.eta = 1e18

    for phi_bot, phi_top in phi_vals:
        ana.phys.phi_top = phi_top
        ana.phys.phi_bot = phi_bot
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)

        tau_vals = []
        harm_vals = []
        harm = 1
        for h_crystal in h_crystal_vals:
            update_thickness(ana, pnt, h_crystal)
            sigma, harm = ana.fastest_mode(pnt.rayleigh, pnt.ra_comp, harm)
            print(phi_top, phi_bot, sigma, harm, pnt.rayleigh)
            tau_vals.append(np.real(pnt.tau(sigma)))
            harm_vals.append(harm)
        axt.semilogy(h_crystal_vals/1e3, tau_vals,
                     label=phi_str, color=col)
        axl.semilogy(h_crystal_vals/1e3, harm_vals,
                     label=phi_str, color=col)
    axt.semilogy(crystallized / 1e3, time, color='k')
    axt.semilogy(crystallized[2:] / 1e3, crystallized[2:]**2 / pnt.kappa, color='k',
                 linestyle='--')
    axt.set_ylim([None, 1e19])

    i_annot = 2 * len(crystallized) // 3
    axt.annotate('Crystallization time', fontsize=8, va='top',
                 xy=(crystallized[i_annot] / 1e3, time[i_annot]), rotation=3)
    axt.annotate('Diffusive time', fontsize=8, va='top',
                 xy=(crystallized[i_annot]/1e3,
                     crystallized[i_annot]**2/pnt.kappa), rotation=3)

    # more human friendly time units
    axt_right = axt.twinx()
    axt_right.set_yscale('log')
    axt_right.grid(axis='y', ls=':')
    axt_right.set_ylim(axt.get_ylim())
    axt_right.set_yticks([86400, 3.15e7, 3.15e10, 3.15e13, 3.15e16])
    axt_right.set_yticklabels(['1 day', '1 year', '1 kyr', '1 Myr', '1 Gyr'])

    axt.set_ylabel(r'Destabilization time scale $\tau$ (s)', fontsize=FTSZ)
    axl.set_ylabel(r'Harmonic degree $l$', fontsize=FTSZ)
    for axis in (axt, axl):
        axis.set_xlabel(r'Crystallized mantle thickness $h$ (km)', fontsize=FTSZ)
        axis.legend()
        axis.tick_params(labelsize=FTSZ)
    savefig(figt, 'DestabilizationTimeCryst.pdf')
    savefig(figl, 'DestabilizationTimeCryst_l.pdf')


def _sigma(h_cryst, pnt, ana):
    """Return dimensionless growth rate"""
    update_thickness(ana, pnt, h_cryst)
    sigma, _ = ana.fastest_mode(pnt.rayleigh, pnt.ra_comp)
    return sigma


def _time_diff(h_cryst, pnt, ana, crystallized, time):
    """Compute time difference between destab and cooling time"""
    sigma = _sigma(h_cryst, pnt, ana)
    destab = np.real(pnt.tau(sigma))
    cooling = np.interp(h_cryst, crystallized, time)
    return cooling - destab


def time_intersection(ana, pnt, crystallized, time, eps=1e-3):
    """Research time at which destab = cooling time"""
    h_min = crystallized[1]
    h_max = crystallized[-1]
    if _sigma(h_min, pnt, ana) < 0 and _sigma(h_max, pnt, ana) < 0:
        # never unstable
        return np.nan
    elif _time_diff(h_max, pnt, ana, crystallized, time) < 0:
        # not unstable enough to be faster than crystallization
        return np.nan

    if _sigma(h_min, pnt, ana) < 0:
        # search thickness beyond which sigma > 0
        while (h_max - h_min) / h_min > eps:
            h_zero = np.sqrt(h_min * h_max)
            if _sigma(h_zero, pnt, ana) < 0:
                h_min = h_zero
            else:
                h_max = h_zero
        h_min = h_max
        h_max = crystallized[-1]
    if _time_diff(h_min, pnt, ana, crystallized, time) > 0:
        # crystallization too slow to determine intersection
        # could try to determine h_zero with better accuracy
        return np.nan

    while (h_max - h_min) / h_min > eps:
        h_cryst = np.sqrt(h_min * h_max)
        if _time_diff(h_cryst, pnt, ana, crystallized, time) < 0:
            h_min = h_cryst
        else:
            h_max = h_cryst
    return np.interp(h_cryst, crystallized, time)


def plot_min_time(pnt, ana, crystallized, time):
    """Plot time at which destab = cooling time"""
    fig, axt = plt.subplots(1, 1)
    eta_logs = np.linspace(15, 18, 10)
    for phi_bot, phi_top in phi_vals:
        ana.phys.phi_top = phi_top
        ana.phys.phi_bot = phi_bot
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)
        tau_vals = []
        for eta in eta_logs:
            pnt.eta = 10**eta
            tau_val = time_intersection(ana, pnt, crystallized, time)
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
    pnt.compo_effects = COMPO_EFFECTS
    h_crystal_max = pnt.r_eut - pnt.r_int - 150e3
    h_crystal_vals = np.logspace(np.log10(5) + 3, np.log10(h_crystal_max), 200)
    phi_vals = [(None, None), (None, 1e-2), (1e-2, 1e-2)]

    crystallized, time = pnt.cooling_time(h_crystal_max, True)
    plot_cooling_time(crystallized, time)

    gamma_from_h = lambda h: pnt.r_int / (pnt.r_int + h)
    gam_smo = gamma_from_h(crystallized[1:])
    gamt_smo = (crystallized[1:] / pnt.d_crystal)
    w_smo = ((crystallized[1:] - crystallized[:-1]) / (time[1:] - time[:-1]) *
             crystallized[1:] / pnt.kappa)
    gamt_f = lambda g: np.interp(g, gam_smo, gamt_smo)
    w_f = lambda g: np.interp(g, gam_smo, w_smo)
    ana = LinearAnalyzer(
        PhysicalProblem(cooling_smo=(gamt_f, w_f)),
        ncheb=24)

    plot_min_time(pnt, ana, crystallized, time)
    plot_destab(pnt, ana, crystallized, time)
