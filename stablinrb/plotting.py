from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from dmsuite.interp import ChebyshevSampling
from lazympl.plot import Plot
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import sph_harm

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .cartesian import CartStability
    from .nonlin import NonLinearAnalyzer
    from .spherical import SphStability


def normalize(arr: NDArray) -> tuple[NDArray, np.complexfloating]:
    """Normalize complex array with element of higher modulus."""
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax


# scientific format for text
def fmt(x: float | None) -> str:
    if x is None:
        return r"\infty"
    a, bs = "{:.2e}".format(x).split("e")
    b = int(bs)
    if b:
        if float(a) != 1:
            a = r"{} \times ".format(a)
        else:
            a = ""
        return r"{}10^{{{}}}".format(a, b)
    else:
        return a


def image_mode(
    xgr: NDArray,
    zgr: NDArray,
    t2d: NDArray,
    u2d: NDArray,
    w2d: NDArray,
    harm: float,
    filename: str,
    notbare: bool = True,
) -> None:
    """Create an image for one mode and save it in a file.

    Takes 2D fields in input as grids
    Only 2D cartesian for now
    """
    dpi = 300
    # prepare plot. 1 temperature anomaly
    fig = plt.figure(dpi=dpi)
    axis = fig.add_subplot(111)
    ax = plt.gca()
    # plot temperature
    surf = plt.pcolormesh(
        xgr, zgr, t2d, cmap="inferno", rasterized=True, linewidth=0, shading="gouraud"
    )
    plt.axis((xgr.min(), xgr.max(), zgr.min(), zgr.max()))
    # stream function
    speed = np.sqrt(u2d**2 + w2d**2)
    if speed.max() > 0:
        lw = 2 * speed / speed.max()
        plt.streamplot(xgr, zgr, u2d, w2d, linewidth=lw, density=0.7)
    # labels etc.
    if notbare:
        axis.tick_params(axis="both", which="major")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$z$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(surf, cax=cax)
        cbar.set_label(r"$T$")
    if harm > 0.6:
        fig.set_figwidth(9)
        axis.set_aspect("equal")
    else:
        fig.set_size_inches(9, 2)
    axis.set_adjustable("box")
    plt.savefig(filename, bbox_inches="tight", format="PDF")
    plt.close(fig)


def plot_mode_image(
    analyzer: NonLinearAnalyzer,
    eps: float = 1,
    name: str | None = None,
    plot_ords: bool = False,
) -> None:
    """Plot velocity and temperature image of a mode given on input

    Generic function for any mode, not necessarily the fastest growing one.
    Used also for non-linear solutions, only in cartesian for now
    inputs:
    analyser: holding all numerical and physical parameters
    eps: epsilon value for the non-linear mode
    plot_ords: if True, plots individual modes on separate figures, in addition to the full solution
    """
    if name is None:
        name = analyzer.linear_pblm.name()

    # Define plot space
    n_rad = 100
    n_phi = 400  # could depends on wavelength
    # initialize fields
    t2d = np.zeros((n_rad, n_phi))
    u2d = np.zeros((n_rad, n_phi))
    w2d = np.zeros((n_rad, n_phi))
    # local version for plotting individual modes
    t2dl = np.zeros((n_rad, n_phi))
    u2dl = np.zeros((n_rad, n_phi))
    w2dl = np.zeros((n_rad, n_phi))

    rad = np.linspace(-1 / 2, 1 / 2, n_rad)
    harm = analyzer.harm_c

    # 2D cartesian box
    # make a version with the total temperature
    xvar = np.linspace(0, 2 * np.pi / harm, n_phi)
    xgr, zgr = np.meshgrid(xvar, rad)

    # conduction solution
    tcond = 0.5 - zgr

    cheb_sampling = ChebyshevSampling(
        degree=analyzer.linear_pblm.chebyshev_degree,
        positions=np.linspace(-1, 1, n_rad),
    )
    for i, mode in enumerate(analyzer.all_modes):
        nord, nharm = analyzer.mode_index.ord_harm(i)

        (p_mode, u_mode, w_mode, t_mode) = analyzer.linear_pblm.split_mode(mode)

        # interpolate and normalize according to vertical velocity
        u_interp = cheb_sampling.apply_on(u_mode)
        w_interp = cheb_sampling.apply_on(w_mode)
        t_interp = cheb_sampling.apply_on(t_mode)

        # horizontal dependence
        modx = np.exp(1j * harm * nharm * xvar)
        # temperature
        t2d1, t2d2 = np.meshgrid(modx, t_interp)
        t2dl = eps**nord * 2 * np.real(t2d1 * t2d2)
        t2d += t2dl
        # velocity
        u2d1, u2d2 = np.meshgrid(modx, u_interp)
        u2dl = eps**nord * 2 * np.real(u2d1 * u2d2)
        u2d += u2dl
        w2d1, w2d2 = np.meshgrid(modx, w_interp)
        w2dl = eps**nord * 2 * np.real(w2d1 * w2d2)
        w2d += w2dl
        if plot_ords:
            # individual contributions
            filename = f"{name}_mode_theta_ord-{nord}_nharm-{nharm}.pdf"
            image_mode(xgr, zgr, t2dl, u2dl, w2dl, harm, filename)
            # parial sum
            filename = f"{name}_mode_theta_ordd-{nord}_nharm-{nharm}.pdf"
            image_mode(xgr, zgr, t2d, u2d, w2d, harm, filename)

    # Plot 1: temperature anomaly
    filename = f"{name}_mode_theta_eps-{eps}_ord-{nord}.pdf"
    image_mode(xgr, zgr, t2d, u2d, w2d, harm, filename)

    # Plot 2: total temperature
    t2d += tcond
    filename = f"{name}_mode_T_eps-{eps}_ord-{nord}.pdf"
    image_mode(xgr, zgr, t2d, u2d, w2d, harm, filename)


def plot_mode_profiles(
    analyzer: NonLinearAnalyzer,
    name: str | None = None,
) -> None:
    """Plot all mode profiles"""
    if name is None:
        name = analyzer.linear_pblm.name()

    rad_cheb = analyzer.nodes

    for i, mode in enumerate(analyzer.all_modes):
        nord, nharm = analyzer.mode_index.ord_harm(i)
        (p_mode, u_mode, w_mode, t_mode) = analyzer.linear_pblm.split_mode(mode)

        fig, axe = plt.subplots(1, 4, sharey=True)
        axe[0].plot(np.real(w_mode), rad_cheb, "o")
        axe[0].set_ylabel(r"$z$")
        axe[0].set_xlabel(r"$W$")
        axe[1].plot(np.imag(u_mode), rad_cheb, "o")
        axe[1].set_xlabel(r"$U$")
        axe[2].plot(np.real(t_mode), rad_cheb, "o")
        axe[2].set_xlabel(r"$\Theta$")
        axe[3].plot(np.real(p_mode), rad_cheb, "o")
        axe[3].set_xlabel(r"$P$")

        plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
        plt.close(fig)


def plot_fastest_mode_cart(
    pblm: CartStability,
    harm: float,
    ra_num: float,
    ra_comp: float | None = None,
    name: str | None = None,
    plot_theory: bool = False,
    notbare: bool = True,
) -> None:
    """Plot fastest growing mode for a given harmonic and Ra

    plot_theory: theory in case of transition, cartesian geometry
    """
    if name is None:
        name = pblm.name()

    _, modes = pblm.eigen_problem(harm, ra_num, ra_comp).max_eigvec()
    modes = modes.normalize_by_max_of("T")
    (p_mode, u_mode, w_mode, t_mode) = pblm.split_mode(modes)

    rad_cheb = pblm.nodes

    # interpolation
    n_rad = 100
    n_phi = 400  # could depends on wavelength
    cheb_sampling = ChebyshevSampling(
        degree=rad_cheb.size - 1,
        positions=np.linspace(-1, 1, n_rad),
    )
    rad = np.linspace(-1 / 2, 1 / 2, n_rad)
    p_interp = cheb_sampling.apply_on(p_mode)
    u_interp = cheb_sampling.apply_on(u_mode)
    w_interp = cheb_sampling.apply_on(w_mode)
    t_interp = cheb_sampling.apply_on(t_mode)

    p_mode, p_max = normalize(p_mode)
    u_mode, u_max = normalize(u_mode)
    w_mode, w_max = normalize(w_mode)

    # profiles
    fig, axis = plt.subplots(1, 4, sharey=True)
    plt.setp(
        axis, xlim=[-1.1, 1.1], ylim=[-1 / 2, 1 / 2], xticks=[-1, -0.5, 0.0, 0.5, 1]
    )
    # pressure
    if plot_theory:
        axis[0].plot(-rad * 4 / (13 * np.sqrt(13)) * (39 - 64 * rad**2), rad)
    else:
        axis[0].plot(np.real(p_interp / p_max), rad)
    axis[0].plot(np.real(p_mode), rad_cheb, "o")
    axis[0].set_xlabel(r"$P/(%.3f)$" % (np.real(p_max)))
    # horizontal velocity
    if plot_theory:
        axis[1].plot(-2 * rad, rad)
    else:
        axis[1].plot(np.real(u_interp / u_max), rad)
    axis[1].plot(np.real(u_mode), rad_cheb, "o")
    axis[1].set_xlabel(r"$U/(%.3fi)$" % (np.imag(u_max)))
    # vertical velocity
    if plot_theory:
        axis[2].plot(np.ones(rad.shape), rad)
    else:
        axis[2].plot(np.real(w_interp / w_max), rad)
    axis[2].plot(np.real(w_mode), rad_cheb, "o")
    axis[2].set_xlabel(r"$W/(%.3f)$" % (np.real(w_max)))
    # temperature
    if plot_theory:
        axis[3].plot(1 - 4 * rad**2, rad)
    else:
        axis[3].plot(np.real(t_interp), rad)
    axis[3].plot(np.real(t_mode), rad_cheb, "o")
    axis[3].set_xlabel(r"$T$")
    filename = "_".join((name, "mode_prof.pdf"))
    plt.savefig(filename, format="PDF")
    plt.close(fig)

    # make a version with the total temperature
    xvar = np.linspace(0, 2 * np.pi / harm, n_phi)
    xgr, zgr = np.meshgrid(xvar, rad)
    # temperature
    modx = np.exp(1j * harm * xvar)
    t2d1, t2d2 = np.meshgrid(modx, t_interp)
    t2d = np.real(t2d1 * t2d2)
    # stream function
    u2d1, u2d2 = np.meshgrid(modx, u_interp)
    u2d = np.real(u2d1 * u2d2)
    w2d1, w2d2 = np.meshgrid(modx, w_interp)
    w2d = np.real(w2d1 * w2d2)
    filename = "_".join((name, "mode.pdf"))
    image_mode(xgr, zgr, t2d, u2d, w2d, harm, filename, notbare)


def plot_fastest_mode_sph(
    pblm: SphStability,
    harm: int,
    ra_num: float,
    ra_comp: float | None = None,
    name: str | None = None,
    plot_text: bool = True,
) -> None:
    """Plot fastest growing mode for a given harmonic and Ra

    plot_theory: theory in case of transition, cartesian geometry
    """
    if name is None:
        name = pblm.name()

    _, modes = pblm.eigen_problem(harm, ra_num, ra_comp).max_eigvec()
    modes = modes.normalize_by_max_of("T")
    (p_mode, u_mode, w_mode, t_mode) = pblm.split_mode(modes, harm)

    rad_cheb = pblm.nodes
    # stream function
    psi_mode = 1.0j * harm * p_mode

    # interpolation
    n_rad = 100
    n_phi = 400  # could depends on wavelength
    cheb_sampling = ChebyshevSampling(
        degree=rad_cheb.size - 1,
        positions=np.linspace(-1, 1, n_rad),
    )
    rad = np.linspace(1, 2, n_rad)
    psi_interp = cheb_sampling.apply_on(psi_mode)
    p_interp = cheb_sampling.apply_on(p_mode)
    u_interp = cheb_sampling.apply_on(u_mode)
    w_interp = cheb_sampling.apply_on(w_mode)
    t_interp = cheb_sampling.apply_on(t_mode)

    p_mode, p_max = normalize(p_mode)
    u_mode, u_max = normalize(u_mode)
    w_mode, w_max = normalize(w_mode)

    # profiles
    fig, axis = plt.subplots(1, 4, sharey=True)
    plt.setp(axis, xlim=[-1.1, 1.1], ylim=[1, 2], xticks=[-1, -0.5, 0.0, 0.5, 1])
    # pressure
    axis[0].plot(np.real(p_interp / p_max), rad)
    axis[0].plot(np.real(p_mode), rad_cheb, "o")
    axis[0].set_xlabel(r"$P/(%.3f)$" % (np.real(p_max)))
    # horizontal velocity
    axis[1].plot(np.real(u_interp / u_max), rad)
    axis[1].plot(np.real(u_mode), rad_cheb, "o")
    axis[1].set_xlabel(r"$U/(%.3fi)$" % (np.imag(u_max)))
    # vertical velocity
    axis[2].plot(np.real(w_interp / w_max), rad)
    axis[2].plot(np.real(w_mode), rad_cheb, "o")
    axis[2].set_xlabel(r"$W/(%.3f)$" % (np.real(w_max)))
    # temperature
    axis[3].plot(np.real(t_interp), rad)
    axis[3].plot(np.real(t_mode), rad_cheb, "o")
    axis[3].set_xlabel(r"$T$")
    filename = "_".join((name, "mode_prof.pdf"))
    plt.savefig(filename, format="PDF")
    plt.close(fig)

    # 2D plot on annulus
    # mesh construction
    gamma = pblm.gamma
    theta = np.pi / 2
    phi = np.linspace(0, 2 * np.pi, n_phi)
    rad = np.linspace(gamma, 1, n_rad)
    rad_mesh, phi_mesh = np.meshgrid(rad, phi)

    # spherical harmonic
    s_harm = sph_harm(harm, harm, phi_mesh, theta)
    t_field = (t_interp * s_harm).real
    psi_field = (psi_interp * s_harm).real

    # normalization
    t_min, t_max = t_field.min(), t_field.max()
    t_field = 2 * (t_field - t_min) / (t_max - t_min) - 1

    # create annulus frame
    dpi = 300
    fig = plt.figure(dpi=dpi)
    tr = PolarAxes.PolarTransform()

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(2.0 * np.pi, 0, gamma, 1),
    )

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    axis = ax1.get_aux_axes(tr)

    # plot Temperature
    axis.pcolormesh(
        phi_mesh, rad_mesh, t_field, cmap="inferno", rasterized=True, shading="gouraud"
    )
    # plot stream lines
    axis.contour(phi_mesh, rad_mesh, psi_field, negative_linestyles="solid")
    axis.set_axis_off()
    fig.set_size_inches(9, 9)
    if plot_text:
        phitop = None
        phibot = None
        if pblm.bc_mom_top.flow_through:
            phitop = pblm.bc_mom_top.phase_number  # type: ignore
        if pblm.bc_mom_bot.flow_through:
            phibot = pblm.bc_mom_bot.phase_number  # type: ignore
        axis.text(
            0.5,
            0.58,
            r"$Ra={}$".format(fmt(ra_num)),
            fontsize=25,
            transform=axis.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
        )
        axis.text(
            0.5,
            0.5,
            r"$\Phi^+={}$".format(fmt(phitop)),
            fontsize=25,
            transform=axis.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
        )
        axis.text(
            0.5,
            0.42,
            r"$\Phi^-={}$".format(fmt(phibot)),
            fontsize=25,
            transform=axis.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
        )
    filename = "_".join((name, "mode.pdf"))
    plt.savefig(filename, bbox_inches="tight", format="PDF", transparent=True)
    plt.close(fig)


@dataclass(frozen=True)
class RanVsKxCartesian(Plot):
    problem: CartStability
    ra_comp: float | None
    kx_min: float
    kx_max: float
    n_kx: int

    @cached_property
    def _kxs(self) -> NDArray[np.float64]:
        return np.linspace(self.kx_min, self.kx_max, self.n_kx, dtype=np.float64)

    @cached_property
    def _ran(self) -> NDArray[np.float64]:
        rac_k: list[float] = []
        for kx in self._kxs:
            rac_k.append(
                self.problem.neutral_ra(
                    harm=kx.item(),
                    ra_guess=(rac_k[-1] if rac_k else 600),
                    ra_comp=self.ra_comp,
                )
            )
        return np.array(rac_k)

    def draw_on(self, ax: Axes) -> None:
        ax.plot(self._kxs, self._ran)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Rayleigh number")


def plot_ran_harm_cart(
    pblm: CartStability,
    harm: float,
    ra_comp: float | None = None,
    name: str | None = None,
    hmin: float | None = None,
    hmax: float | None = None,
) -> None:
    """Plot neutral Ra vs harmonic around given harm"""
    kxmin = harm
    ramin = pblm.neutral_ra(kxmin, ra_comp=ra_comp)
    hhmin = kxmin / 2 if hmin is None else hmin
    hhmax = 1.5 * kxmin if hmax is None else hmax

    plot = RanVsKxCartesian(
        problem=pblm,
        ra_comp=ra_comp,
        kx_min=hhmin,
        kx_max=hhmax,
        n_kx=50,
    )

    fig, axis = plt.subplots(1, 1)
    plot.draw_on(axis)

    # FIXME: this relies on `harm` being the critical wavenumber
    if ramin < 1:
        lbl = rf"$Ra_{{min}}={ramin:.2e} ; k={kxmin:.2e}$"
    else:
        lbl = rf"$Ra_{{min}}={ramin:.2f} ; k={kxmin:.2f}$"
    axis.plot(kxmin, ramin, "o", label=lbl)
    axis.legend(loc="upper right")

    if name is None:
        name = pblm.name()
    fig.tight_layout()
    fig.savefig(f"{name}_Ra_kx.pdf")
    plt.close(fig)


@dataclass(frozen=True)
class RanVsEllSpherical(Plot):
    problem: SphStability
    ra_comp: float | None
    ell_min: int
    ell_max: int

    @property
    def _ells(self) -> range:
        return range(self.ell_min, self.ell_max + 1)

    @cached_property
    def _ran(self) -> NDArray[np.float64]:
        rac_l: list[float] = []
        for l_harm in self._ells:
            rac_l.append(
                self.problem.neutral_ra(
                    l_harm,
                    ra_guess=(rac_l[-1] if rac_l else 600),
                    ra_comp=self.ra_comp,
                )
            )
        return np.array(rac_l)

    def draw_on(self, ax: Axes) -> None:
        ax.plot(self._ells, self._ran, "o")
        ax.plot()

        i_rac = np.argmin(self._ran)
        l_c = self._ells[i_rac]
        ra_c = self._ran[i_rac]
        ax.plot(
            l_c,
            ra_c,
            "o",
            label=rf"$Ra_{{min}}={ra_c:.2f} ; l={l_c}$",
            markersize=mpl.rcParams["lines.markersize"] * 1.5,
        )

        ax.set_xlabel(r"Spherical harmonic $l$")
        ax.set_ylabel(r"Critical Rayleigh number $Ra_c$")
        ax.set_xlim(self.ell_min - 0.3, self.ell_max + 0.3)
        ax.legend()


def plot_ran_harm_sph(
    pblm: SphStability,
    harm: float,
    ra_comp: float | None = None,
    name: str | None = None,
    hmin: float | None = None,
    hmax: float | None = None,
) -> None:
    """Plot neutral Ra vs harmonic around given harm"""
    harm = int(harm)
    if hmin is not None:
        lmin = int(hmin)
    else:
        if harm < 25:
            lmin = 1
        else:
            lmin = harm - 7
    if hmax is not None:
        lmax = int(hmax)
    else:
        if harm < 25:
            lmax = max(15, harm + 5)
        else:
            lmax = lmin + 14

    plot = RanVsEllSpherical(
        problem=pblm,
        ra_comp=ra_comp,
        ell_min=lmin,
        ell_max=lmax,
    )

    fig, axis = plt.subplots(1, 1)
    plot.draw_on(axis)

    if name is None:
        name = pblm.name()
    fig.tight_layout()
    fig.savefig(f"{name}_Ra_l.pdf")
    plt.close(fig)
