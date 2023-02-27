from __future__ import annotations

import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from dmsuite.interp import ChebyshevSampling
from matplotlib.projections import PolarAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import sph_harm

from .matrix import Vector
from .misc import normalize_modes

if typing.TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from .analyzer import Analyser, LinearAnalyzer
    from .nonlin import NonLinearAnalyzer
    from .physics import PhysicalProblem

mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("text", usetex=True)
plt.rcParams["contour.negative_linestyle"] = "solid"

# mpl.rcParams['font.size'] = 16
mpl.rcParams["pdf.fonttype"] = 42

mypal = "inferno"

# Font and markers size
FTSZ = 13
MSIZE = 5


# scientific format for text
def fmt(x: Optional[float]) -> str:
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
        xgr, zgr, t2d, cmap=mypal, rasterized=True, linewidth=0, shading="gouraud"
    )
    plt.axis([xgr.min(), xgr.max(), zgr.min(), zgr.max()])
    # stream function
    speed = np.sqrt(u2d**2 + w2d**2)
    if speed.max() > 0:
        lw = 2 * speed / speed.max()
        plt.streamplot(xgr, zgr, u2d, w2d, linewidth=lw, density=0.7)
    # labels etc.
    if notbare:
        axis.tick_params(axis="both", which="major", labelsize=FTSZ)
        axis.set_xlabel(r"$x$", fontsize=FTSZ + 2)
        axis.set_ylabel(r"$z$", fontsize=FTSZ + 2)
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
    mode: NDArray,
    harm: float,
    eps: float = 1,
    name: Optional[str] = None,
    plot_ords: bool = False,
) -> None:
    """Plot velocity and temperature image of a mode given on input

    Generic function for any mode, not necessarily the fastest growing one.
    Used also for non-linear solutions, only in cartesian for now
    inputs:
    analyser: holding all numerical and physical parameters
    harm: wavenumber of the mode
    eps: epsilon value for the non-linear mode
    plot_ords: if True, plots individual modes on separate figures, in addition to the full solution
    """
    if analyzer.phys.spherical:
        raise ValueError("plot_mode_image not yet implemented in spherical")
    if name is None:
        name = analyzer.phys.name()

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

    # 2D cartesian box
    # make a version with the total temperature
    xvar = np.linspace(0, 2 * np.pi / harm, n_phi)
    xgr, zgr = np.meshgrid(xvar, rad)

    # conduction solution
    tcond = 0.5 - zgr

    nmodes = mode.shape[0]

    cheb_sampling = ChebyshevSampling(
        degree=analyzer.rad.size - 1,
        positions=np.linspace(-1, 1, n_rad),
    )
    for i in range(nmodes):
        nord, nharm = analyzer.indexmat(nmodes, ind=i)[1:3]

        mode_vec = Vector(slices=analyzer.slices, arr=mode[i])
        (p_mode, u_mode, w_mode, t_mode) = analyzer.split_mode(mode_vec, harm)

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


def w11(rad: NDArray, phi: float) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """11 mode for the vertical velocity at low phi for both bc"""
    ww0 = 0.5 * np.ones(rad.shape)
    ww1 = -9 / 128 * rad**2 * phi
    # ww1 = - 3 / 40 * rad ** 2 * phi
    ww2 = 9 * rad**2 * (2.3 * rad**2 - 2.2) / 1.6384e3 * phi**2
    # ww2 = rad ** 2 * (-1.365 + 1.47 * rad ** 2 - 0.56 * rad ** 4) / 112 * phi ** 2
    ww3 = -0.5 * 2.7 * rad**2 * (2.20 * rad**2 - 3.09) / 2.097152e3 * phi**3
    ww4 = 135 * rad**2 * (-4.293 + 2.084 * rad**2) / 4.294967296e6 * phi**4
    return ww0, ww1, ww2, ww3, ww4


def u11(rad: NDArray, phi: float) -> tuple[NDArray, NDArray, NDArray]:
    """11 mode for the horizontal velocity at low phi for both bc"""
    # uu = - np.sqrt(3 * phi * 10) * rad / 2
    # uu2 = - rad * (16 * rad ** 4 - 28 * rad ** 2 + 13) / 160 * np.sqrt(3 / 10) * phi ** 1.5
    uu1 = -3 / 8 * rad * np.sqrt(phi / 2)
    uu2 = 3 * rad * (23 * rad**2 - 11) / (512 * np.sqrt(2)) * phi**1.5
    uu3 = 3 * (4.4 * rad**2 - 3.09) * rad / (2.62144e3 * np.sqrt(2)) * phi**2.5
    return uu1, uu2, uu3


def p11(rad: NDArray, phi: float) -> tuple[NDArray, NDArray, NDArray]:
    """11 mode for the pressure at low phi for both bc"""
    pp1 = (39 / 32 - 2 * rad**2) * rad * phi
    pp2 = 9 * rad * (35 * rad**2 - 22) / 2048 * phi**2
    pp3 = -9 * rad * (1.772 * rad**2 - 0.751) / 4.194304e3 * phi**3
    return pp1, pp2, pp3


def t11(rad: NDArray, phi: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """11 mode for the temperature at low phi for both bc"""
    tt0 = (1 - 4 * rad**2) / 16
    # tt2 = - 3 / 1280 * phi * (1 - 4 * rad ** 2)
    # tt3 = (0.233 - 4.108 * rad ** 2 + 1.488 * rad ** 4 - 0.320 * rad ** 6) / 14336 * phi ** 2 * (1 - 4 * rad ** 2)
    tt1 = -9 / 4096 * phi * (1 - 4 * rad**2)
    tt2 = 3 * (1 - 4 * rad**2) * (212 * rad**2 - 1) / 2.097152e6 * phi**2
    tt3 = 27 * (1 - 4 * rad**2) * (4.3 * rad**2 + 1.07) / 2.68435456e6 * phi**3
    return tt0, tt1, tt2, tt3


def t20(rad: NDArray, phi: float) -> tuple[NDArray, NDArray]:
    """20 mode for the temperature at low phi for both bc"""
    # zeroth order in phi
    tt = (1 - 4 * rad**2) * rad / 96
    # order 1 in phi
    tt1 = (-57 * rad / 163840 + 3 * rad**3 / 4096 + 27 * rad**5 / 10240) * phi
    return tt, tt1


def p20(rad: NDArray, phi: float) -> NDArray:
    """20 mode for the pressure at low phi for both bc"""
    # order 1 in phi
    pp1 = -0.25 * (1 / 16 - 0.5 * rad**2 + rad**4) * phi
    # order 1 in phi
    # tt1 = (-57 * rad /163840 +3 * rad ** 3 / 4096 + 27 * rad ** 5 / 10240 ) * phi
    return pp1


def t22(rad: NDArray, phi: float) -> tuple[NDArray, NDArray]:
    """22 mode for the temperature at low phi for both bc"""
    # zeroth order in phi
    tt = (1 - 4 * rad**2) * rad / 96
    # order 1 in phi
    tt1 = -(1 - 4 * rad**2) * rad * (35 / 49152 * phi + 565 / 6291456 * phi**2)
    return tt, tt1


def w22(rad: NDArray, phi: float) -> tuple[NDArray, NDArray]:
    """22 mode for the vertical velocity at low phi for both bc"""
    # ww = phi * rad * (1 - 4 * rad ** 2) * 3 / 256
    # ww = (np.sqrt(2 * phi) * harm + phi) * rad * 3 /256
    # order 1
    ww1 = phi * rad / 128
    # order 2
    ww2 = phi**2 * (-1631 * rad / 524288 - 17 * rad**3 / 131072)
    return ww1, ww2


def u22(rad: NDArray, phi: float) -> tuple[NDArray, NDArray]:
    """22 mode for the horizontal velocity at low phi for both bc"""
    # order 1/2
    uu1 = np.sqrt(phi / 2) / 96 * np.ones(rad.shape)  # - harm * phi / 4096
    # uu = np.sqrt(phi * 2) / 128 * (1 - 12 * rad ** 2)
    # order 3/2
    uu2 = (
        -(phi ** (3 / 2))
        / np.sqrt(2)
        * (1631 / 393216 + 17 * rad**2 / 32768 + rad**4 / 64)
    )
    return uu1, uu2


def plot_mode_profiles(
    analyzer: NonLinearAnalyzer,
    mode: NDArray,
    harm: float,
    name: Optional[str] = None,
    plot_theory: bool = False,
) -> None:
    """Plot all mode profiles"""
    if name is None:
        name = analyzer.phys.name()
    phitop = analyzer.phys.phi_top
    phibot = analyzer.phys.phi_bot

    rad_cheb = analyzer.rad
    nmodes = mode.shape[0]

    for i in range(nmodes):
        nord, nharm = analyzer.indexmat(nmodes, ind=i)[1:3]
        mode_vec = Vector(slices=analyzer.slices, arr=mode[i])
        (p_mode, u_mode, w_mode, t_mode) = analyzer.split_mode(mode_vec, harm)

        fig, axe = plt.subplots(1, 4, sharey=True)

        pik2 = np.pi**2 + harm**2

        axe[0].plot(np.real(w_mode), rad_cheb, "o")
        axe[0].set_ylabel(r"$z$", fontsize=FTSZ)
        axe[0].set_xlabel(r"$W$", fontsize=FTSZ)
        axe[1].plot(np.imag(u_mode), rad_cheb, "o")
        axe[1].set_xlabel(r"$U$", fontsize=FTSZ)
        axe[2].plot(np.real(t_mode), rad_cheb, "o")
        axe[2].set_xlabel(r"$\Theta$", fontsize=FTSZ)
        axe[3].plot(np.real(p_mode), rad_cheb, "o")
        # if nord == 2 and nharm == 0:
        # print('p_mode = ', np.real(p_mode))
        # print('p20 theorique =', p20(rad_cheb, phitop))
        axe[3].set_xlabel(r"$P$", fontsize=FTSZ)

        if plot_theory:
            if (phitop is None or phitop >= 1e2) and (phibot is None or phibot >= 1e2):
                if nord == 1 and nharm == 1:
                    axe[0].plot(0.5 * np.cos(np.pi * rad_cheb), rad_cheb)
                    axe[1].plot(
                        -0.5 * np.sin(np.pi * rad_cheb) * np.pi / harm, rad_cheb
                    )
                    axe[2].plot(0.5 * np.cos(np.pi * rad_cheb) / pik2, rad_cheb)
                if nord == 2 and nharm == 0:
                    axe[2].plot(
                        np.sin(2 * np.pi * rad_cheb) / pik2 / (16 * np.pi), rad_cheb
                    )
                if nord == 3 and nharm == 1:
                    axe[0].plot(
                        pik2**2
                        * np.cos(3 * np.pi * rad_cheb)
                        / 16
                        / (pik2**3 - (9 * np.pi**2 + harm**2) ** 3),
                        rad_cheb,
                    )
                    axe[1].plot(
                        -(pik2**2)
                        * 3
                        / harm
                        * np.pi
                        * np.sin(3 * np.pi * rad_cheb)
                        / 16
                        / (pik2**3 - (9 * np.pi**2 + harm**2) ** 3),
                        rad_cheb,
                    )
                plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
                plt.close(fig)
            if (phitop is not None and phitop <= 10) and (phibot == phitop):
                if nord == 1 and nharm == 1:
                    # leading order
                    axe[0].plot(
                        w11(rad_cheb, phitop)[0] + w11(rad_cheb, phitop)[1], rad_cheb
                    )
                    axe[1].plot(
                        u11(rad_cheb, phitop)[0] + u11(rad_cheb, phitop)[1], rad_cheb
                    )
                    axe[2].plot(
                        t11(rad_cheb, phitop)[0] + t11(rad_cheb, phitop)[1], rad_cheb
                    )
                    axe[3].plot(
                        p11(rad_cheb, phitop)[0] + p11(rad_cheb, phitop)[1], rad_cheb
                    )
                    plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
                    plt.close(fig)
                    # next order : new figure
                    fig, axe = plt.subplots(1, 4, sharey=True)
                    # W
                    axe[0].plot(
                        np.real(w_mode) - w11(rad_cheb, phitop)[0], rad_cheb, "o"
                    )
                    axe[0].plot(w11(rad_cheb, phitop)[1], rad_cheb)
                    axe[0].set_ylabel(r"$z$", fontsize=FTSZ)
                    axe[0].set_xlabel(r"$W$", fontsize=FTSZ)
                    # U
                    axe[1].plot(
                        np.imag(u_mode) - u11(rad_cheb, phitop)[0], rad_cheb, "o"
                    )
                    axe[1].plot(u11(rad_cheb, phitop)[1], rad_cheb)
                    axe[1].set_xlabel(r"$U$", fontsize=FTSZ)
                    # T
                    axe[2].plot(
                        np.real(t_mode) - t11(rad_cheb, phitop)[0], rad_cheb, "o"
                    )
                    axe[2].plot(t11(rad_cheb, phitop)[1], rad_cheb)
                    axe[2].set_xlabel(r"$\Theta$", fontsize=FTSZ)
                    # P
                    axe[3].plot(
                        np.real(p_mode) - p11(rad_cheb, phitop)[0], rad_cheb, "o"
                    )
                    axe[3].plot(p11(rad_cheb, phitop)[1], rad_cheb)
                    # save
                    plt.savefig(f"name_ord1_n-{nord}_l-{nharm}.pdf", transparent=True)
                    plt.close(fig)
                    # next order : new figure
                    fig, axe = plt.subplots(1, 4, sharey=True)
                    # W
                    axe[0].plot(
                        np.real(w_mode)
                        - w11(rad_cheb, phitop)[0]
                        - w11(rad_cheb, phitop)[1],
                        rad_cheb,
                        "o",
                    )
                    axe[0].plot(w11(rad_cheb, phitop)[2], rad_cheb)
                    axe[0].set_ylabel(r"$z$", fontsize=FTSZ)
                    axe[0].set_xlabel(r"$W$", fontsize=FTSZ)
                    # U
                    axe[1].plot(
                        np.imag(u_mode)
                        - u11(rad_cheb, phitop)[0]
                        - u11(rad_cheb, phitop)[1],
                        rad_cheb,
                        "o",
                    )
                    axe[1].plot(u11(rad_cheb, phitop)[2], rad_cheb)
                    axe[1].set_xlabel(r"$U$", fontsize=FTSZ)
                    # T
                    axe[2].plot(
                        np.real(t_mode)
                        - t11(rad_cheb, phitop)[0]
                        - t11(rad_cheb, phitop)[1],
                        rad_cheb,
                        "o",
                    )
                    axe[2].plot(t11(rad_cheb, phitop)[2], rad_cheb)
                    axe[2].set_xlabel(r"$\Theta$", fontsize=FTSZ)
                    # P
                    axe[3].plot(
                        np.real(p_mode)
                        - p11(rad_cheb, phitop)[0]
                        - p11(rad_cheb, phitop)[1],
                        rad_cheb,
                        "o",
                    )
                    axe[3].plot(p11(rad_cheb, phitop)[2], rad_cheb)
                    axe[3].set_xlabel(r"$P$", fontsize=FTSZ)
                    # save
                    plt.savefig(f"name_ord2_n-{nord}_l-{nharm}.pdf", transparent=True)
                    plt.close(fig)
                if nord == 2 and nharm == 0:
                    axe[0].plot(np.zeros(rad_cheb.shape), rad_cheb)
                    axe[1].plot(np.zeros(rad_cheb.shape), rad_cheb)
                    axe[2].plot(t20(rad_cheb, phitop)[0], rad_cheb)
                    axe[3].plot(p20(rad_cheb, phitop), rad_cheb)
                    plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
                    plt.close(fig)
                if nord == 2 and nharm == 2:
                    axe[0].plot(w22(rad_cheb, phitop)[0], rad_cheb)
                    axe[1].plot(u22(rad_cheb, phitop)[0], rad_cheb)
                    axe[2].plot(t22(rad_cheb, phitop)[0], rad_cheb)
                    plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
                    plt.close(fig)
                if nord >= 3:
                    plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
                    plt.close(fig)
        else:
            plt.savefig(f"name_n-{nord}_l-{nharm}.pdf")
            plt.close(fig)


def plot_fastest_mode(
    analyzer: Analyser,
    harm: float,
    ra_num: float,
    ra_comp: Optional[float] = None,
    name: Optional[str] = None,
    plot_theory: bool = False,
    notbare: bool = True,
    plot_text: bool = True,
) -> None:
    """Plot fastest growing mode for a given harmonic and Ra

    plot_theory: theory in case of transition, cartesian geometry
    """
    spherical = analyzer.phys.spherical

    gamma = analyzer.phys.gamma
    if name is None:
        name = analyzer.phys.name()

    sigma, modes = analyzer.eigvec(harm, ra_num, ra_comp)
    # p is pressure in cartesian geometry and
    # poloidal potential in spherical geometry
    (p_mode, u_mode, w_mode, t_mode) = analyzer.split_mode(modes, harm)

    rad_cheb = analyzer.rad
    if spherical:
        # stream function
        psi_mode = 1.0j * harm * p_mode

    # interpolation
    n_rad = 100
    n_phi = 400  # could depends on wavelength
    cheb_sampling = ChebyshevSampling(
        degree=rad_cheb.size - 1,
        positions=np.linspace(-1, 1, n_rad),
    )
    if spherical:
        rad = np.linspace(1, 2, n_rad)
        psi_interp = cheb_sampling.apply_on(psi_mode)
    else:
        rad = np.linspace(-1 / 2, 1 / 2, n_rad)
    p_interp = cheb_sampling.apply_on(p_mode)
    u_interp = cheb_sampling.apply_on(u_mode)
    w_interp = cheb_sampling.apply_on(w_mode)
    t_interp = cheb_sampling.apply_on(t_mode)

    # normalization with max of T and then
    # element of max modulus of each vector
    norms, maxs = normalize_modes((p_mode, u_mode, w_mode, t_mode))
    p_norm, u_norm, w_norm, t_norm = norms
    p_max, u_max, w_max, t_max = maxs
    # profiles
    fig, axis = plt.subplots(1, 4, sharey=True)
    if spherical:
        plt.setp(axis, xlim=[-1.1, 1.1], ylim=[1, 2], xticks=[-1, -0.5, 0.0, 0.5, 1])
    else:
        plt.setp(
            axis, xlim=[-1.1, 1.1], ylim=[-1 / 2, 1 / 2], xticks=[-1, -0.5, 0.0, 0.5, 1]
        )
    # pressure
    if plot_theory:
        axis[0].plot(-rad * 4 / (13 * np.sqrt(13)) * (39 - 64 * rad**2), rad)
    else:
        axis[0].plot(np.real(p_interp / t_max / p_max), rad)
    axis[0].plot(np.real(p_norm), rad_cheb, "o")
    axis[0].set_xlabel(r"$P/(%.3f)$" % (np.real(p_max)), fontsize=FTSZ)
    # horizontal velocity
    if plot_theory:
        axis[1].plot(-2 * rad, rad)
    else:
        axis[1].plot(np.real(u_interp / t_max / u_max), rad)
    axis[1].plot(np.real(u_norm), rad_cheb, "o")
    axis[1].set_xlabel(r"$U/(%.3fi)$" % (np.imag(u_max)), fontsize=FTSZ)
    # vertical velocity
    if plot_theory:
        axis[2].plot(np.ones(rad.shape), rad)
    else:
        axis[2].plot(np.real(w_interp / t_max / w_max), rad)
    axis[2].plot(np.real(w_norm), rad_cheb, "o")
    axis[2].set_xlabel(r"$W/(%.3f)$" % (np.real(w_max)), fontsize=FTSZ)
    # temperature
    if plot_theory:
        axis[3].plot(1 - 4 * rad**2, rad)
    else:
        axis[3].plot(np.real(t_interp / t_max), rad)
    axis[3].plot(np.real(t_norm), rad_cheb, "o")
    axis[3].set_xlabel(r"$T$", fontsize=FTSZ)
    filename = "_".join((name, "mode_prof.pdf"))
    plt.savefig(filename, format="PDF")
    plt.close(fig)

    if spherical:
        # 2D plot on annulus
        # mesh construction
        assert gamma is not None
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
            phi_mesh, rad_mesh, t_field, cmap=mypal, rasterized=True, shading="gouraud"
        )
        # plot stream lines
        axis.contour(phi_mesh, rad_mesh, psi_field)
        axis.set_axis_off()
        fig.set_size_inches(9, 9)
        if plot_text:
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
                r"$\Phi^+={}$".format(fmt(analyzer.phys.phi_top)),
                fontsize=25,
                transform=axis.transAxes,
                verticalalignment="center",
                horizontalalignment="center",
            )
            axis.text(
                0.5,
                0.42,
                r"$\Phi^-={}$".format(fmt(analyzer.phys.phi_bot)),
                fontsize=25,
                transform=axis.transAxes,
                verticalalignment="center",
                horizontalalignment="center",
            )
        filename = "_".join((name, "mode.pdf"))
        plt.savefig(filename, bbox_inches="tight", format="PDF", transparent=True)
        plt.close(fig)

    else:
        # 2D cartesian box
        # make a version with the total temperature
        xvar = np.linspace(0, 2 * np.pi / harm, n_phi)
        xgr, zgr = np.meshgrid(xvar, rad)
        # temperature
        modx = np.exp(1j * harm * xvar)
        t2d1, t2d2 = np.meshgrid(modx, t_interp / t_max)
        t2d = np.real(t2d1 * t2d2)
        # stream function
        u2d1, u2d2 = np.meshgrid(modx, u_interp / t_max)
        u2d = np.real(u2d1 * u2d2)
        w2d1, w2d2 = np.meshgrid(modx, w_interp / t_max)
        w2d = np.real(w2d1 * w2d2)
        filename = "_".join((name, "mode.pdf"))
        image_mode(xgr, zgr, t2d, u2d, w2d, harm, filename, notbare)


def plot_ran_harm(
    analyzer: LinearAnalyzer,
    harm: float,
    ra_comp: Optional[float] = None,
    name: Optional[str] = None,
    hmin: Optional[float] = None,
    hmax: Optional[float] = None,
) -> None:
    """Plot neutral Ra vs harmonic around given harm"""
    if name is None:
        name = analyzer.phys.name()
    fig, axis = plt.subplots(1, 1)
    if analyzer.phys.spherical:
        rac_l: list[float] = []
        harm = int(harm)
        if harm < 25:
            lmin = 1
            lmax = max(15, harm + 5)
        else:
            lmin = harm - 7
            lmax = lmin + 14
        harms = range(lmin, lmax + 1)
        for idx, l_harm in enumerate(harms):
            rac_l.append(
                analyzer.neutral_ra(
                    l_harm, ra_guess=(rac_l[idx - 1] if idx else 600), ra_comp=ra_comp
                )
            )

        l_c, ra_c = min(enumerate(rac_l), key=lambda tpl: tpl[1])
        l_c += lmin

        plt.setp(axis, xlim=[lmin - 0.3, lmax + 0.3])
        plt.plot(harms, rac_l, "o", markersize=MSIZE)
        plt.plot(
            l_c,
            ra_c,
            "o",
            label=rf"$Ra_{{min}}={ra_c:.2f} ; l={l_c}$",
            markersize=MSIZE * 1.5,
        )
        plt.xlabel(r"Spherical harmonic $l$", fontsize=FTSZ)
        plt.ylabel(r"Critical Rayleigh number $Ra_c$", fontsize=FTSZ)
        plt.legend(loc="upper right", fontsize=FTSZ)
        filename = "_".join((name, "Ra_l.pdf"))
    else:
        kxmin = harm
        ramin = analyzer.neutral_ra(kxmin, ra_comp=ra_comp)
        hhmin = kxmin / 2 if hmin is None else hmin
        hhmax = 1.5 * kxmin if hmax is None else hmax
        wnum = np.linspace(hhmin, hhmax, 50)
        rayl = [analyzer.neutral_ra(wnum[0], ramin, ra_comp)]
        for i, kk in enumerate(wnum[1:]):
            ra2 = analyzer.neutral_ra(kk, rayl[i], ra_comp)
            rayl.append(ra2)

        plt.plot(wnum, rayl, linewidth=2)
        if ramin < 1:
            plt.plot(
                kxmin, ramin, "o", label=r"$Ra_{min}=%.2e ; k=%.2e$" % (ramin, kxmin)
            )
        else:
            plt.plot(
                kxmin, ramin, "o", label=r"$Ra_{min}=%.2f ; k=%.2f$" % (ramin, kxmin)
            )
        plt.xlabel("Wavenumber", fontsize=FTSZ)
        plt.ylabel("Rayleigh number", fontsize=FTSZ)
        plt.legend(loc="upper right", fontsize=FTSZ)
        filename = "_".join((name, "Ra_kx.pdf"))
    plt.xticks(fontsize=FTSZ)
    plt.yticks(fontsize=FTSZ)
    plt.tight_layout()
    plt.savefig(filename, format="PDF")
    plt.close(fig)


def plot_viscosity(pblm: PhysicalProblem) -> None:
    """Plot viscosity profile of a give physical problem"""
    if pblm.eta_r is None:
        return
    nrad = 100
    gamma = pblm.gamma
    if pblm.spherical:
        assert gamma is not None
        rad = np.linspace(gamma / (1 - gamma), 1 / (1 - gamma), nrad)
    else:
        rad = np.linspace(-1 / 2, 1 / 2, nrad)
    eta_r = np.vectorize(pblm.eta_r)(rad)
    fig, axis = plt.subplots(1, 2, sharey=True)
    axis[0].plot(eta_r, rad)
    axis[1].semilogx(eta_r, rad)
    if pblm.spherical:
        assert gamma is not None
        plt.setp(axis, ylim=[gamma / (1 - gamma), 1 / (1 - gamma)])
    else:
        plt.setp(axis, ylim=[-1 / 2, 1 / 2])
    filename = "_".join((pblm.name(), "visco.pdf"))
    axis[0].set_xlabel(r"$\eta$", fontsize=FTSZ)
    axis[1].set_xlabel(r"$\eta$", fontsize=FTSZ)
    axis[0].set_ylabel(r"$r$", fontsize=FTSZ)
    plt.tight_layout()
    plt.savefig(filename, format="PDF")
