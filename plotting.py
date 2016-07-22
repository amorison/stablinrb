import dmsuite.dmsuite as dm
from misc import normalize_modes
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, \
    MaxNLocator, DictFormatter
import seaborn as sns

# Font and markers size
FTSZ = 20
MSIZE = 6


def plot_fastest_mode(analyzer, harm, ra_num, ra_comp=None,
                      name=None, plot_theory=False):
    """Plot fastest growing mode for a given harmonic and Ra

    plot_theory: theory in case of transition, cartesian geometry
    """
    spherical = analyzer.phys.spherical
    gamma = analyzer.phys.gamma
    if name is None:
        name = analyzer.phys.name()

    sigma, modes, _ = analyzer.eigval(harm, ra_num, ra_comp)
    # p is pressure in cartesian geometry and
    # poloidal potential in spherical geometry
    p_mode, u_mode, w_mode, t_mode = modes
    rad_cheb = analyzer.rad
    if spherical:
        # stream function
        psi_mode = 1.j * harm * p_mode

    # interpolation
    n_rad = 100
    n_phi = 400  # could depends on wavelength
    cheb_space = np.linspace(-1, 1, n_rad)
    if spherical:
        rad = np.linspace(analyzer.phys.gamma, 1, n_rad)
        psi_interp = dm.chebint(psi_mode, cheb_space)
    else:
        rad = np.linspace(-1/2, 1/2, n_rad)
    p_interp = dm.chebint(p_mode, cheb_space)
    u_interp = dm.chebint(u_mode, cheb_space)
    w_interp = dm.chebint(w_mode, cheb_space)
    t_interp = dm.chebint(t_mode, cheb_space)

    # normalization with max of T and then
    # element of max modulus of each vector
    norms, maxs = normalize_modes(modes)
    p_norm, u_norm, w_norm, t_norm = norms
    p_max, u_max, w_max, t_max = maxs

    # profiles
    fig, axis = plt.subplots(1, 4, sharey=True)
    if spherical:
        plt.setp(axis, xlim=[-1.1, 1.1], ylim=[gamma, 1],
                 xticks=[-1, -0.5, 0., 0.5, 1])
    else:
        plt.setp(axis, xlim=[-1.1, 1.1], ylim=[-1/2, 1/2],
                 xticks=[-1, -0.5, 0., 0.5, 1])
    if plot_theory:
        axis[0].plot(rad*4/(13*np.sqrt(13))*(39-64*rad**2), rad)
    else:
        axis[0].plot(p_interp / t_max / p_max, rad)
    axis[0].plot(p_norm, rad_cheb, 'o')
    axis[0].set_xlabel(r'$P/(%.3f)$' %(np.real(p_max)), fontsize=FTSZ)
    if plot_theory:
        axis[1].plot(-2 * rad, rad)
    else:
        axis[1].plot(u_interp / t_max / u_max, rad)
    axis[1].plot(u_norm, rad_cheb, 'o')
    axis[1].set_xlabel(r'$U/(%.3fi)$' %(np.imag(u_max)), fontsize=FTSZ)
    if plot_theory:
        axis[2].plot(np.ones(rad.shape), rad)
    else:
        axis[2].plot(w_interp / t_max / w_max, rad)
    axis[2].plot(w_norm, rad_cheb, 'o')
    axis[2].set_xlabel(r'$W/(%.3f)$' %(np.real(w_max)), fontsize=FTSZ)
    if plot_theory:
        axis[3].plot(1-4*rad**2, rad)
    else:
        axis[3].plot(t_interp / t_max, rad)
    axis[3].plot(t_norm, rad_cheb, 'o')
    axis[3].set_xlabel(r'$T$', fontsize=FTSZ)
    filename = '_'.join((name, 'mode_prof.pdf'))
    plt.savefig(filename, format='PDF')
    plt.close(fig)

    if spherical:
        # 2D plot on annulus
        # mesh construction
        theta = np.pi/2
        phi = np.linspace(0, 2*np.pi, n_phi)
        rad_mesh, phi_mesh = np.meshgrid(rad, phi)

        # spherical harmonic
        s_harm = sph_harm(harm, harm, phi_mesh, theta)
        t_field = (t_interp * s_harm).real
        ur_field = (w_interp * s_harm).real
        up_field = (u_interp * s_harm).real
        psi_field = (psi_interp * s_harm).real

        # normalization
        t_min, t_max = t_field.min(), t_field.max()
        t_field = 2 * (t_field - t_min) / (t_max - t_min) - 1

        # create annulus frame
        fig = plt.figure()
        tr = PolarAxes.PolarTransform()

        angle_ticks = []
        grid_locator1 = FixedLocator([v for v, _ in angle_ticks])
        tick_formatter1 = DictFormatter(dict(angle_ticks))

        radius_ticks = []
        grid_locator2 = FixedLocator([v for v, _ in radius_ticks])
        tick_formatter2 = DictFormatter(dict(radius_ticks))

        grid_helper = floating_axes.GridHelperCurveLinear(
            tr, extremes=(2.*np.pi, 0, 1, gamma),
            grid_locator1=grid_locator1, tick_formatter1=tick_formatter1,
            grid_locator2=grid_locator2, tick_formatter2=tick_formatter2)

        ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
        fig.add_subplot(ax1)
        axis = ax1.get_aux_axes(tr)

        # plot Temperature
        surf = axis.pcolormesh(phi_mesh, rad_mesh, t_field,
                               cmap='RdBu_r', shading='gouraud')
        # plot stream lines
        axis.contour(phi_mesh, rad_mesh, psi_field)
        axis.set_axis_off()
    else:
        # 2D cartesian box
        # make a version with the total temperature
        xvar = np.linspace(0, 2*np.pi/harm, n_phi)
        xgr, zgr = np.meshgrid(xvar, rad)
        # temperature
        modx = np.exp(1j * harm * xvar)
        t2d1, t2d2 = np.meshgrid(modx, t_interp / t_max)
        t2d = np.real(t2d1 * t2d2)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        if harm > 0.2:
            fig = plt.figure(figsize=(6*np.pi/harm, 3), dpi=300)
        else:
            fig = plt.figure()
        axis = fig.add_subplot(111)
        surf = plt.pcolormesh(xgr, zgr, t2d, cmap='RdBu_r', linewidth=0,)
        plt.axis([xgr.min(), xgr.max(), zgr.min(), zgr.max()])
        # stream function
        u2d1, u2d2 = np.meshgrid(modx, u_interp / t_max)
        u2d = np.real(u2d1*u2d2)
        w2d1, w2d2 = np.meshgrid(modx, w_interp / t_max)
        w2d = np.real(w2d1*w2d2)
        speed = np.sqrt(u2d**2+w2d**2)
        lw = 2 * speed / speed.max()
        plt.streamplot(xgr, zgr, u2d, w2d, linewidth=lw, density=0.7)
        plt.xlabel(r'$x$', fontsize=FTSZ)
        plt.ylabel(r'$z$', fontsize=FTSZ)

    cbar = plt.colorbar(surf, shrink=0.8)
    cbar.set_label(r'Temperature $\Theta$')
    plt.tight_layout()
    filename = '_'.join((name, 'mode.pdf'))
    plt.savefig(filename, format='PDF')
    plt.close(fig)


def plot_ran_harm(analyzer, harm, ra_comp=None, name=None):
    """Plot neutral Ra vs harmonic around given harm"""
    if name is None:
        name = analyzer.phys.name()
    fig, axis = plt.subplots(1, 1)
    if analyzer.phys.spherical:
        rac_l = []
        lmin = max(1, harm - 5)
        lmax = lmin + 9
        harms = range(lmin, lmax + 1)
        for idx, l_harm in enumerate(harms):
            rac_l.append(analyzer.neutral_ra(
                l_harm, ra_guess=(rac_l[idx-1] if idx else 600)))

        l_c, ra_c = min(enumerate(rac_l), key=lambda tpl: tpl[1])
        l_c += lmin

        plt.setp(axis, xlim=[lmin - 0.3, lmax + 0.3])
        plt.plot(harms, rac_l, 'o', markersize=MSIZE)
        plt.plot(l_c, ra_c, 'o',
                 label=r'$Ra_{min}=%.2f ; l=%d$' %(ra_c, l_c),
                 markersize=MSIZE*1.5)
        plt.xlabel(r'Spherical harmonic $l$', fontsize=FTSZ)
        plt.ylabel(r'Critical Rayleigh number $Ra_c$', fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        filename = '_'.join((name, 'Ra_l.pdf'))
    else:
        kxmin = harm
        ramin = analyzer.neutral_ra(kxmin, ra_comp=ra_comp)
        wnum = np.linspace(kxmin/2, kxmin*1.5, 50)
        rayl = [analyzer.neutral_ra(wnum[0], ramin, ra_comp)]
        for i, kk in enumerate(wnum[1:]):
            ra2 = analyzer.neutral_ra(kk, rayl[i], ra_comp)
            rayl = np.append(rayl, ra2)

        plt.plot(wnum, rayl, linewidth=2)
        if ramin < 1:
            plt.plot(kxmin, ramin, 'o',
                     label=r'$Ra_{min}=%.2e ; k=%.2e$' %(ramin, kxmin))
        else:
            plt.plot(kxmin, ramin, 'o',
                     label=r'$Ra_{min}=%.2f ; k=%.2f$' %(ramin, kxmin))
        plt.xlabel('Wavenumber', fontsize=FTSZ)
        plt.ylabel('Rayleigh number', fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        filename = '_'.join((name, 'Ra_kx.pdf'))
    plt.xticks(fontsize=FTSZ)
    plt.yticks(fontsize=FTSZ)
    plt.tight_layout()
    plt.savefig(filename, format='PDF')
    plt.close(fig)
