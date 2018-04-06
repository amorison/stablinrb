#!/usr/bin/env python3
"""
Crystallization and cumulate destabilization time scales of a SMO
"""
import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import misc
from analyzer import LinearAnalyzer
from physics import PhysicalProblem
from planets import EARTH as pnt
import plotting

# Font and markers size
FTSZ = 14
MSIZE = 5

TEMP_EFFECTS = True
COMPO_EFFECTS = True
FROZEN_TIME = False


def savefig(fig, stem):
    stem += '.pdf'
    misc.savefig(fig, str(out_dir / stem))


# compute time to crystallize given thickness of solid mantle

def update_thickness(ana, pnt, h_crystal):
    """Update analyzer and planet with new thickness"""
    pnt.h_crystal = h_crystal
    ana.phys.gamma = pnt.gamma
    if TEMP_EFFECTS:
        ana.phys.grad_ref_temperature = np.vectorize(
            lambda r: pnt.grad_temp(pnt.radim(r)) *
                pnt.h_crystal / pnt.delta_temp())
    if COMPO_EFFECTS:
        ana.phys.composition = np.vectorize(
            lambda r: pnt.composition(pnt.radim(r)))


def _phi_col_lbl(phi_top, phi_bot):
    if phi_bot is None:
        phi_str = r'$\Phi^+=10^{-2}$' if phi_top else 'closed'
        col = 'g' if phi_top else 'b'
    else:
        phi_str = r'$\Phi^+=\Phi^-=10^{-2}$'
        col = 'r'
    return col, phi_str


_PHI_LBL = {
    'closed': 'closed',
    r'$\Phi^+=10^{-2}$': 'open at top',
    r'$\Phi^+=\Phi^-=10^{-2}$': 'open at top and bottom',
}


def plot_destab(pnt, ana, crystallized, time, outfile):
    """Plot destabilization time scale as function of solid thickness"""
    figt, axt = plt.subplots(1, 1)
    figl, axl = plt.subplots(1, 1)
    pnt.eta = 1e18

    for phi_bot, phi_top in phi_vals:
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)
        with h5py.File(outfile, 'a') as h5f:
            if phi_str in h5f:
                print('Reading destab data from {}/{}'.
                      format(outfile, phi_str))
                tau_vals = h5f[phi_str]['tau'].value
                harm_vals = h5f[phi_str]['harmonics'].value
            else:
                grp = h5f.create_group(phi_str)
                ana.phys.phi_top = phi_top
                ana.phys.phi_bot = phi_bot
                tau_vals = []
                harm_vals = []
                harm = 1
                for h_crystal in h_crystal_vals:
                    update_thickness(ana, pnt, h_crystal)
                    sigma, harm = ana.fastest_mode(pnt.rayleigh, pnt.ra_comp, harm)
                    print(phi_top, phi_bot, sigma, harm, pnt.rayleigh)
                    tau_vals.append(np.real(pnt.tau(sigma)))
                    harm_vals.append(harm)
                tau_vals = np.array(tau_vals)
                harm_vals = np.array(harm_vals)
                grp['tau'] = tau_vals
                grp['harmonics'] = harm_vals
                grp['thickness'] = h_crystal_vals
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
    savefig(figt, 'DestabilizationTimeCryst')
    savefig(figl, 'DestabilizationTimeCryst_l')


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


def plot_min_time(pnt, ana, crystallized, time, outfile):
    """Plot time at which destab = cooling time"""
    fig, axt = plt.subplots(1, 1)
    eta_logs = np.linspace(15, 18, 10)
    for phi_bot, phi_top in phi_vals:
        col, phi_str = _phi_col_lbl(phi_top, phi_bot)
        with h5py.File(outfile, 'a') as h5f:
            if phi_str in h5f:
                print('Reading destab X cooling from {}/{}'.
                      format(outfile, phi_str))
                tau_vals = h5f[phi_str]['tau'].value
            else:
                grp = h5f.create_group(phi_str)
                ana.phys.phi_top = phi_top
                ana.phys.phi_bot = phi_bot
                tau_vals = []
                for eta in eta_logs:
                    pnt.eta = 10**eta
                    tau_val = time_intersection(ana, pnt, crystallized, time)
                    tau_vals.append(tau_val)
                    print(eta, tau_val)
                tau_vals = np.array(tau_vals)
                grp['tau'] = tau_vals
                grp['eta_logs'] = eta_logs
        axt.loglog(10**eta_logs, tau_vals / 3.15e10,
                   color=col, label=_PHI_LBL[phi_str])
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
    savefig(fig, 'CoolingDestab')


def plot_modes(ana, pnt, thickness):
    """Plot fastest growing mode at a give thickness."""
    update_thickness(ana, pnt, thickness)
    sigma, harm = ana.fastest_mode(pnt.rayleigh, pnt.ra_comp)
    plotting.plot_fastest_mode(ana, harm, pnt.rayleigh, pnt.ra_comp)


def plot_cooling_time(pnt, crystallized, time):
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
    savefig(fig, 'CoolingTime')


def plot_composition(pnt):
    """Plot cooling time"""
    compo = np.vectorize(pnt.composition)
    #tim = time / 3.15e10
    compo_liq = np.vectorize(
        lambda r: pnt.composition(r) / pnt.part if r < pnt.r_eut else 1)
    rad = np.linspace(pnt.r_int, pnt.r_tot, 100)
    fig, axis = plt.subplots(1, 1)
    axis.plot(compo(rad), rad / 1e3, label='FeO content of solid')
    axis.plot(compo_liq(rad), rad / 1e3, label='FeO content of liquid')
    axis.set_xlabel('FeO content at SMO/solid boundary')
    axis.set_ylabel(r'$R^+$ (km)')
    axis.set_xlim([0, 1])
    axis.set_ylim([pnt.r_int / 1e3, pnt.r_tot / 1e3])
    axis.legend(loc='best')
    savefig(fig, 'CompoTime')


if __name__ == '__main__':

    suffix = '_frozen' if FROZEN_TIME else ''
    if not TEMP_EFFECTS or COMPO_EFFECTS:
        ValueError('TEMP_EFFECTS or COMPO_EFFECTS should be switched on')
    pnt_dir = pathlib.Path(pnt.name)
    if not TEMP_EFFECTS:
        out_dir = pnt_dir / ('onlyCompo' + suffix)
    elif not COMPO_EFFECTS:
        out_dir = pnt_dir / ('onlyTemp' + suffix)
    else:
        out_dir = pnt_dir / ('both' + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    pnt.compo_effects = COMPO_EFFECTS
    h_crystal_max = pnt.r_eut - pnt.r_int - (10e3 if pnt.name != 'Moon' else 1e2)
    h_crystal_vals = np.logspace(np.log10(5) + 3, np.log10(h_crystal_max), 2000)
    phi_vals = [(None, None), (None, 1e-2), (1e-2, 1e-2)]
    if pnt.name != 'Earth':
        phi_vals = phi_vals[:-1]

    crystallized, time = pnt.cooling_time(h_crystal_max, pnt_dir / 'CoolingSMO.h5',
                                          verbose=True)
    plot_cooling_time(pnt, crystallized, time)

    gamma_from_h = lambda h: pnt.r_int / (pnt.r_int + h)
    gam_smo = gamma_from_h(crystallized[1:])
    gamt_smo = (crystallized[1:] / pnt.d_crystal)
    w_smo = ((crystallized[1:] - crystallized[:-1]) / (time[1:] - time[:-1]) *
             crystallized[1:] / pnt.kappa)
    gamt_f = lambda g: np.interp(g, gam_smo, gamt_smo)
    w_f = lambda g: np.interp(g, gam_smo, w_smo)
    ana = LinearAnalyzer(
        PhysicalProblem(cooling_smo=(gamt_f, w_f),
                        grad_ref_temperature=None,
                        frozen_time=FROZEN_TIME),
        ncheb=24)

    plot_min_time(pnt, ana, crystallized, time, out_dir / 'interTime.h5')
    plot_destab(pnt, ana, crystallized, time, out_dir / 'DestabTime.h5')

    #for phi_bot, phi_top in phi_vals:
    #    ana.phys.phi_bot = phi_bot
    #    ana.phys.phi_top = phi_top
    #    plot_modes(ana, pnt, 2 * pnt.d_crystal / 3)
    #plot_composition(pnt)#, crystallized, time)
