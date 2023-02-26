from __future__ import annotations

import typing

import numpy as np
import numpy.ma as ma
from dmsuite.poly_diff import Chebyshev
from scipy import linalg

from .misc import build_slices
from .physics import wtran

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Sequence

    from numpy.typing import NDArray

    from .physics import PhysicalProblem


def cartesian_matrices(
    self: Analyser, wnk: float, ra_num: float, ra_comp: Optional[float] = None
) -> tuple[NDArray, NDArray]:
    """Build left- and right-hand-side matrices in cartesian geometry case"""
    # parameters
    ncheb = self._ncheb
    zphys = self.rad
    h_int = self.phys.h_int
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    lewis = self.phys.lewis
    composition = self.phys.composition
    prandtl = self.phys.prandtl
    comp_terms = lewis is not None or composition is not None
    translation = self.phys.ref_state_translation
    water = self.phys.water
    thetar = self.phys.thetar
    if comp_terms and ra_comp is None:
        raise ValueError("ra_comp must be specified for compositional problem")

    # first and second order z-derivatives
    dz1, dz2 = self.dr1, self.dr2
    # Identity matrix
    one = np.identity(ncheb + 1)
    # horizontal derivative
    dh1 = 1j * wnk * one
    # Laplace operator
    lapl = dz2 - wnk**2 * one

    # global indices and slices
    i0n, igf, slall, slint, slgall, slgint = self._slices()
    i_0s, i_ns = zip(*i0n)
    if comp_terms:
        ip0, iu0, iw0, it0, ic0 = i_0s
        ipn, iun, iwn, itn, icn = i_ns
        ipg, iug, iwg, itg, icg = igf
        pall, uall, wall, tall, call = slall
        pint, uint, wint, tint, cint = slint
        pgall, ugall, wgall, tgall, cgall = slgall
        pgint, ugint, wgint, tgint, cgint = slgint
    else:
        ip0, iu0, iw0, it0 = i_0s
        ipn, iun, iwn, itn = i_ns
        ipg, iug, iwg, itg = igf
        pall, uall, wall, tall = slall
        pint, uint, wint, tint = slint
        pgall, ugall, wgall, tgall = slgall
        pgint, ugint, wgint, tgint = slgint

    # For pressure. No BCs but side values needed or removed
    # depending on the BCs for W. number of lines need to be
    # the same as that of d2w and depends on bcsw.

    if translation:
        assert phi_bot is not None and phi_top is not None
        rtr = 12 * (phi_top + phi_bot)
        wtrans = wtran((ra_num - rtr) / rtr)[0]

    lmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1)) + 0j
    rmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1))

    # Pressure equations
    # mass conservation
    lmat[pgall, ugall] = dh1[pall, uall]
    lmat[pgall, wgall] = dz1[pall, wall]

    # U equations
    # free-slip at top
    if phi_top is not None or freeslip_top:
        lmat[iug(iu0), ugall] = dz1[iu0, uall]
    if phi_top is not None:
        lmat[iug(iu0), wgall] = dh1[iu0, wall]
    # horizontal momentum conservation
    lmat[ugint, pgall] = -dh1[uint, pall]
    lmat[ugint, ugall] = lapl[uint, uall]
    # free-slip at bot
    if phi_bot is not None or freeslip_bot:
        lmat[iug(iun), ugall] = dz1[iun, uall]
    if phi_bot is not None:
        lmat[iug(iun), wgall] = dh1[iun, wall]

    # W equations
    if phi_top is not None:
        # phase change at top
        lmat[iwg(iw0), pgall] = -one[iw0, pall]
        lmat[iwg(iw0), wgall] = phi_top * one[iw0, wall] + 2 * dz1[iw0, wall]
    # vertical momentum conservation
    lmat[wgint, pgall] = -dz1[wint, pall]
    lmat[wgint, wgall] = lapl[wint, wall]
    if water:
        theta0 = thetar - zphys
        lmat[wgint, tgall] = -ra_num * np.diag(theta0)[wint, tall]
    else:
        lmat[wgint, tgall] = ra_num * one[wint, tall]
    if comp_terms:
        assert ra_comp is not None
        lmat[wgint, cgall] = ra_comp * one[wint, call]
    if phi_bot is not None:
        # phase change at bot
        lmat[iwg(iwn), pgall] = -one[iwn, pall]
        lmat[iwg(iwn), wgall] = -phi_bot * one[iwn, wall] + 2 * dz1[iwn, wall]

    # Neumann boundary condition if imposed flux
    if heat_flux_top is not None:
        lmat[itg(it0), tgall] = dz1[it0, tall]
    elif heat_flux_bot is not None:
        lmat[itg(itn), tgall] = dz1[itn, tall]
    if self.phys.biot_top is not None:
        lmat[itg(it0), tgall] = (self.phys.biot_top * one + dz1)[it0, tall]
    if self.phys.biot_bot is not None:
        lmat[itg(itn), tgall] = (self.phys.biot_bot * one + dz1)[itn, tall]

    lmat[tgint, tgall] = lapl[tint, tall]

    # need to take heat flux into account in T conductive
    if translation:
        # only written for Dirichlet BCs on T and without internal heating
        lmat[tgint, tgall] -= wtrans * dz1[tint, tall]
        lmat[tgint, wgall] = np.diag(np.exp(wtrans * self.rad[wall]))[tint, wall]
        if np.abs(wtrans) > 1.0e-3:
            lmat[tgint, wgall] *= wtrans / (2 * np.sinh(wtrans / 2))
        else:
            # use a limited development
            lmat[tgint, wgall] *= 1 - wtrans**2 / 24
    else:
        grad_tcond = -h_int * zphys
        if heat_flux_bot is not None:
            grad_tcond += heat_flux_bot - h_int / 2
        elif heat_flux_top is not None:
            grad_tcond += heat_flux_top + h_int / 2
        else:
            if water:
                # cooled from below
                grad_tcond -= 1
            else:
                grad_tcond += 1
        lmat[tgint, wgall] = np.diag(grad_tcond)[tint, wall]

    rmat[tgint, tgall] = one[tint, tall]
    if prandtl is not None:
        # finite Prandtl number case
        rmat[ugint, ugall] = one[uint, uall] / prandtl
        rmat[wgint, wgall] = one[wint, wall] / prandtl
    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if composition is not None:
        lmat[cgint, wgall] = -np.diag(np.dot(dz1, composition(zphys)))[cint, wall]
    elif lewis is not None:
        lmat[cgint, wgall] = one[cint, wall]
        lmat[cgint, cgall] = lapl[cint, call] / lewis
    if comp_terms:
        rmat[cgint, cgall] = one[cint, call]
    return lmat, rmat


def spherical_matrices(
    self: Analyser,
    l_harm: int,
    ra_num: Optional[float] = None,
    ra_comp: Optional[float] = None,
) -> tuple[NDArray, NDArray]:
    """Build left and right matrices in spherical case"""
    gamma = self.phys.gamma
    assert gamma is not None
    rad = self.rad
    dr1, dr2 = self.dr1, self.dr2

    lam_r = (2 * gamma - 1) / (1 - gamma)
    # r + lambda
    ral = rad + lam_r
    # 1 / (r + lambda)
    orl1 = (1 - gamma) / ((1 - gamma) * rad + 2 * gamma - 1)
    orl2 = orl1**2
    orl3 = orl1**3

    rad = np.diag(rad)
    ral = np.diag(ral)
    orl1 = np.diag(orl1)
    orl2 = np.diag(orl2)
    orl3 = np.diag(orl3)

    ncheb = self._ncheb
    one = np.identity(ncheb + 1)  # identity
    lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
    lapl = dr2 + 2 * np.dot(orl1, dr1) - lh2 * orl2  # laplacian
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    if self.phys.C_top is not None:
        C_top = self.phys.C_top
    else:
        C_top = 0
    if self.phys.C_bot is not None:
        C_bot = self.phys.C_bot
    else:
        C_bot = 0
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot

    h_int = self.phys.h_int
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    grad_ref_temperature = self.phys.grad_ref_temperature
    temp_terms = grad_ref_temperature is not None

    if self.phys.eta_r is not None:
        eta_r = np.diag(np.vectorize(self.phys.eta_r)(np.diag(ral)))
    else:
        eta_r = one

    lewis = self.phys.lewis
    composition = self.phys.composition
    comp_terms = lewis is not None or composition is not None

    if temp_terms and ra_num is None:
        raise ValueError("Temperature effect requires ra_num")
    if comp_terms and ra_comp is None:
        raise ValueError("Composition effect requires ra_comp")
    if not (temp_terms or comp_terms):
        raise ValueError("No buoyancy terms!")

    # global indices and slices
    i0n, igf, slall, slint, slgall, slgint = self._slices()
    i_0s, i_ns = zip(*i0n)
    if temp_terms and comp_terms:
        ip0, iq0, it0, ic0 = i_0s
        ipn, iqn, itn, icn = i_ns
        ipg, iqg, itg, icg = igf
        pall, qall, tall, call = slall
        pint, qint, tint, cint = slint
        pgall, qgall, tgall, cgall = slgall
        pgint, qgint, tgint, cgint = slgint
    elif temp_terms:
        ip0, iq0, it0 = i_0s
        ipn, iqn, itn = i_ns
        ipg, iqg, itg = igf
        pall, qall, tall = slall
        pint, qint, tint = slint
        pgall, qgall, tgall = slgall
        pgint, qgint, tgint = slgint
    else:  # only comp_terms
        ip0, iq0, ic0 = i_0s
        ipn, iqn, icn = i_ns
        ipg, iqg, icg = igf
        pall, qall, call = slall
        pint, qint, cint = slint
        pgall, qgall, cgall = slgall
        pgint, qgint, cgint = slgint

    lmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1))
    rmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1))

    # Poloidal potential equations
    if phi_top is not None:
        # free-slip at top
        lmat[ipg(ip0), pgall] = (
            dr2[ip0, pall] + (lh2 * (1 + C_top) - 2) * orl2[ip0, pall]
        )
    else:
        # no radial velocity, Dirichlet condition but
        # need to keep boundary point to ensure free-slip
        # or rigid boundary
        lmat[ipg(ip0), pgall] = one[ip0, pall]
    # laplacian(P) - Q = 0
    lmat[pgint, pgall] = lapl[pint, pall]
    lmat[pgint, qgall] = -one[qint, qall]
    if phi_bot is not None:
        # free-slip at bot
        lmat[ipg(ipn), pgall] = (
            dr2[ipn, pall] + (lh2 * (1 - C_bot) - 2) * orl2[ipn, pall]
        )
    else:
        lmat[ipg(ipn), pgall] = one[ipn, pall]

    # Q equations
    # normal stress continuity at top
    if phi_top is not None:
        lmat[iqg(iq0), pgall] = lh2 * (
            (phi_top - 2 * C_top * (1 - gamma)) * orl1[iq0, pall]
            - 2 * np.dot(eta_r, orl2)[iq0, pall]
            + 2 * np.dot(eta_r, np.dot(orl1, dr1))[iq0, pall]
        )
        lmat[iqg(iq0), qgall] = (
            -eta_r[iq0, qall] - np.dot(eta_r, np.dot(ral, dr1))[iq0, qall]
        )
    elif freeslip_top:
        lmat[iqg(iq0), pgall] = dr2[iq0, pall]
    else:
        # rigid
        lmat[iqg(iq0), pgall] = dr1[iq0, pall]
    if self.phys.eta_r is not None:
        deta_dr = np.diag(np.dot(dr1, np.diag(eta_r)))
        d2eta_dr2 = np.diag(np.dot(dr2, np.diag(eta_r)))
        lmat[qgint, pgall] = (
            2 * (lh2 - 1) * (np.dot(orl2, d2eta_dr2) - np.dot(orl3, deta_dr))
            - 2 * np.dot(np.dot(orl1, d2eta_dr2) - np.dot(orl2, deta_dr), dr1)
        )[qint, pall]
        lmat[qgint, qgall] = (
            np.dot(eta_r, lapl) + d2eta_dr2 + 2 * np.dot(deta_dr, dr1)
        )[qint, qall]
    else:
        # laplacian(Q) - RaT/r = 0
        lmat[qgint, qgall] = lapl[qint, qall]
    if temp_terms:
        assert ra_num is not None
        lmat[qgint, tgall] = -ra_num * orl1[qint, tall]
    if comp_terms:
        assert ra_comp is not None
        lmat[qgint, cgall] = -ra_comp * orl1[qint, call]
    # normal stress continuity at bot
    if phi_bot is not None:
        lmat[iqg(iqn), pgall] = lh2 * (
            -(phi_bot - 2 * C_bot * (1 - gamma) / gamma) * orl1[iqn, pall]
            - 2 * np.dot(eta_r, orl2)[iqn, pall]
            + 2 * np.dot(eta_r, np.dot(orl1, dr1))[iqn, pall]
        )
        lmat[iqg(iqn), qgall] = (
            -eta_r[iqn, qall] - np.dot(eta_r, np.dot(ral, dr1))[iqn, qall]
        )
    elif freeslip_bot:
        lmat[iqg(iqn), pgall] = dr2[iqn, pall]
    else:
        # rigid
        lmat[iqg(iqn), pgall] = dr1[iqn, pall]

    if self.phys.cooling_smo is not None:
        gamt_f, w_f = self.phys.cooling_smo
        gam2_smo = gamt_f(gamma) ** 2
        w_smo = w_f(gamma)

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T
    if temp_terms:
        # Neumann boundary condition if imposed flux
        if heat_flux_top is not None:
            lmat[itg(it0), tgall] = dr1[it0, tall]
        elif heat_flux_bot is not None:
            lmat[itg(itn), tgall] = dr1[itn, tall]
        if self.phys.biot_top is not None:
            lmat[itg(it0), tgall] = (self.phys.biot_top * one + dr1)[it0, tall]
        if self.phys.biot_bot is not None:
            lmat[itg(itn), tgall] = (self.phys.biot_bot * one + dr1)[itn, tall]

        lmat[tgint, tgall] = lapl[tint, tall]
        if not self.phys.frozen_time and self.phys.cooling_smo is not None:
            # TYPE SAFETY: there is an implicit assumption that grad_ref_temperature is a callable
            # with those options
            grad_ref_temp_top = grad_ref_temperature(np.diag(rad)[0])  # type: ignore
            lmat[tgint, tgall] += (
                w_smo * (np.dot(rad - one, dr1) + grad_ref_temp_top * one)[tint, tall]
            )

        # advection of reference profile
        # using u_r = l(l+1)/r P
        # first compute - 1/r * nabla T
        # then multiply by l(l+1)
        if grad_ref_temperature == "conductive":
            # reference is conductive profile
            grad_tcond = h_int / 3 * np.ones(rad.shape[0])
            if heat_flux_bot is not None:
                grad_tcond -= (
                    -((1 + lam_r) ** 2) * heat_flux_bot + h_int * (1 + lam_r) ** 3 / 3
                ) * np.diag(orl3)
            elif heat_flux_top is not None:
                grad_tcond -= (
                    -((2 + lam_r) ** 2) * heat_flux_top + h_int * (2 + lam_r) ** 3 / 3
                ) * np.diag(orl3)
            else:
                grad_tcond -= (
                    (h_int / 6 * (3 + 2 * lam_r) - 1) * (2 + lam_r) * (1 + lam_r)
                ) * np.diag(orl3)
        else:
            # TYPE SAFETY: there is an implicit assumption that grad_ref_temperature is a callable
            # if not "conductive"
            grad_tcond = np.dot(orl1, -grad_ref_temperature(np.diag(rad)))  # type: ignore
        lmat[tgint, pgall] = np.diag(lh2 * grad_tcond)[tint, pall]

        rmat[tgint, tgall] = one[tint, tall]
        if not self.phys.frozen_time and self.phys.cooling_smo:
            rmat[tgint, tgall] *= gam2_smo

    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if composition is not None:
        grad_comp = np.diag(np.dot(dr1, composition(np.diag(rad))))
        lmat[cgint, pgall] = -lh2 * np.dot(orl1, grad_comp)[cint, pall]
    elif lewis is not None:
        raise ValueError("Finite Lewis not implemented in spherical")
    if comp_terms:
        if not self.phys.frozen_time and self.phys.cooling_smo is not None:
            lmat[cgint, cgall] = w_smo * np.dot(rad - one, dr1)[cint, call]
        rmat[cgint, cgall] = one[cint, call]
        if not self.phys.frozen_time and self.phys.cooling_smo is not None:
            rmat[cgint, cgall] *= gam2_smo

    return lmat, rmat


class Analyser:
    """Define various elements common to both analysers"""

    def __init__(self, phys: PhysicalProblem, ncheb: int = 15, nnonlin: int = 2):
        """Create a generic analyzer

        phys is the PhysicalProblem
        ncheb is the number of Chebyshev nodes
        nnonlin is the maximum order of non-linear analysis
        """
        # get differentiation matrices
        self._ncheb = ncheb
        self._nnonlin = nnonlin
        cheb = Chebyshev(degree=ncheb)
        self._zcheb = cheb.nodes
        # rescaling to thickness 1 (cheb space is of thickness 2)
        self.dr1 = cheb.at_order(1) * 2  # first r-derivative
        self.dr2 = cheb.at_order(2) * 4  # second r-derivative

        # weights
        self._invcp = np.ones(ncheb + 1)
        self._invcp[0] = 1 / 2
        self._invcp[-1] = 1 / 2
        # matrix to get the pseudo-spectrum
        self._tmat = np.zeros((ncheb + 1, ncheb + 1))
        for n in range(ncheb + 1):
            for p in range(ncheb + 1):
                self._tmat[n, p] = (-1) ** n * np.cos(n * p * np.pi / ncheb)

        # Chebyshev polynomials are -1 < z < 1
        if phys.spherical:
            # physical space is 1 < r < 2
            self.rad = (self._zcheb + 3) / 2
        else:
            # physical space is -1/2 < z < 1/2
            self.rad = self._zcheb / 2
        self._phys = phys

    def _insert_boundaries(self, mode: NDArray, im0: int, imn: int) -> NDArray:
        """Insert zero at boundaries of mode if needed

        This need to be done when Dirichlet BCs are applied
        """
        if im0 == 1:
            mode = np.insert(mode, [0], [0])
        if imn == self._ncheb - 1:
            mode = np.append(mode, 0)
        return mode

    @property
    def phys(self) -> PhysicalProblem:
        """Property holding the physical problem"""
        return self._phys

    def _slices(
        self,
    ) -> tuple[
        Sequence[tuple[int, int]],
        Sequence[Callable],
        Sequence[slice],
        Sequence[slice],
        Sequence[slice],
        Sequence[slice],
    ]:
        """slices defining the different parts of the global matrix"""
        ncheb = self._ncheb
        phi_top = self.phys.phi_top
        phi_bot = self.phys.phi_bot
        freeslip_top = self.phys.freeslip_top
        freeslip_bot = self.phys.freeslip_bot
        heat_flux_top = self.phys.heat_flux_top
        heat_flux_bot = self.phys.heat_flux_bot
        # index min and max
        # remove boundary when Dirichlet condition
        i0n = []
        if self.phys.spherical:
            # poloidal
            i0n.append((0, ncheb))
            # laplacian of poloidal
            i0n.append((0, ncheb))
        else:
            # pressure
            i0n.append((0, ncheb))
            # horizontal velocity
            i_0 = 0 if (phi_top is not None) or freeslip_top else 1
            i_n = ncheb if (phi_bot is not None) or freeslip_bot else ncheb - 1
            i0n.append((i_0, i_n))
            # vertical velocity
            i_0 = 0 if (phi_top is not None) else 1
            i_n = ncheb if (phi_bot is not None) else ncheb - 1
            i0n.append((i_0, i_n))
        # temperature
        if not (self.phys.spherical and self.phys.grad_ref_temperature is None):
            # handling of arbitrary grad_ref_temperature is only implemented
            # in spherical geometry
            i_0 = (
                0
                if (heat_flux_top is not None or self.phys.biot_top is not None)
                else 1
            )
            i_n = (
                ncheb
                if (heat_flux_bot is not None or self.phys.biot_bot is not None)
                else ncheb - 1
            )
            i0n.append((i_0, i_n))
        if self.phys.composition is not None or self.phys.lewis is not None:
            i0n.append((1, ncheb - 1))
        return build_slices(i0n, ncheb)

    def matrices(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> tuple[NDArray, NDArray]:
        """Build left and right matrices"""
        if self.phys.spherical:
            return spherical_matrices(self, int(harm), ra_num, ra_comp)
        else:
            return cartesian_matrices(self, harm, ra_num, ra_comp)

    def eigval(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> np.floating:
        """Compute the max eigenvalue

        harm: wave number
        ra_num: thermal Rayleigh number
        ra_comp: compositional Ra
        """
        lmat, rmat = self.matrices(harm, ra_num, ra_comp)
        eigvals = linalg.eigvals(lmat, rmat)
        return np.max(np.real(ma.masked_invalid(eigvals)))

    def eigvec(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> tuple[np.complexfloating, NDArray]:
        """Compute the max eigenvalue and associated eigenvector

        harm: wave number
        ra_num: thermal Rayleigh number
        ra_comp: compositional Ra
        """
        lmat, rmat = self.matrices(harm, ra_num, ra_comp)
        eigvals, eigvecs = linalg.eig(lmat, rmat)
        iegv = np.argmax(np.real(ma.masked_invalid(eigvals)))
        return eigvals[iegv], eigvecs[:, iegv]

    def _split_mode_cartesian(
        self, eigvec: NDArray, apply_bc: bool = False
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Split 1D cartesian mode into (p, u, w, t) tuple

        Optionally apply boundary conditions
        """
        # global indices and slices
        i0n, _, _, _, slgall, _ = self._slices()
        i_0s, i_ns = zip(*i0n)
        if self.phys.composition is None and self.phys.lewis is None:
            ip0, iu0, iw0, it0 = i_0s
            ipn, iun, iwn, itn = i_ns
            pgall, ugall, wgall, tgall = slgall
        else:
            ip0, iu0, iw0, it0, ic0 = i_0s
            ipn, iun, iwn, itn, icn = i_ns
            pgall, ugall, wgall, tgall, cgall = slgall

        p_mode = eigvec[pgall]
        u_mode = eigvec[ugall]
        w_mode = eigvec[wgall]
        t_mode = eigvec[tgall]

        if apply_bc:
            p_mode = self._insert_boundaries(p_mode, ip0, ipn)
            u_mode = self._insert_boundaries(u_mode, iu0, iun)
            w_mode = self._insert_boundaries(w_mode, iw0, iwn)
            t_mode = self._insert_boundaries(t_mode, it0, itn)
            # c_mode should be added in case of composition
        return (p_mode, u_mode, w_mode, t_mode)

    def _split_mode_spherical(
        self, eigvec: NDArray, l_harm: int, apply_bc: bool = False
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Split 1D spherical mode into (p, u, w, t) tuple

        Optionally apply boundary conditions
        """
        lewis = self.phys.lewis
        composition = self.phys.composition
        comp_terms = lewis is not None or composition is not None
        # global indices and slices
        i0n, igf, slall, slint, slgall, slgint = self._slices()
        i_0s, i_ns = zip(*i0n)
        if comp_terms:
            ip0, iq0, it0, ic0 = i_0s
            ipn, iqn, itn, icn = i_ns
            ipg, iqg, itg, icg = igf
            pall, qall, tall, call = slall
            pint, qint, tint, cint = slint
            pgall, qgall, tgall, cgall = slgall
            pgint, qgint, tgint, cgint = slgint
        else:
            ip0, iq0, it0 = i_0s
            ipn, iqn, itn = i_ns
            ipg, iqg, itg = igf
            pall, qall, tall = slall
            pint, qint, tint = slint
            pgall, qgall, tgall = slgall
            pgint, qgint, tgint = slgint

        p_mode = eigvec[pgall]
        t_mode = eigvec[tgall]

        if apply_bc:
            p_mode = self._insert_boundaries(p_mode, ip0, ipn)
            t_mode = self._insert_boundaries(t_mode, it0, itn)

        gamma = self.phys.gamma
        assert gamma is not None
        orl1 = (1 - gamma) / ((1 - gamma) * self.rad + 2 * gamma - 1)
        ur_mode = l_harm * (l_harm + 1) * p_mode * orl1
        up_mode = 1j * l_harm * (np.dot(self.dr1, p_mode) + p_mode * orl1)

        return (p_mode, up_mode, ur_mode, t_mode)

    def split_mode(
        self, eigvec: NDArray, harm: float, apply_bc: bool = False
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Generic splitting function"""
        if self.phys.spherical:
            return self._split_mode_spherical(eigvec, int(harm), apply_bc)
        else:
            return self._split_mode_cartesian(eigvec, apply_bc)

    def _join_mode_cartesian(
        self, mode: tuple[NDArray, NDArray, NDArray, NDArray]
    ) -> NDArray:
        """concatenate (p, u, w, t) mode into 1D cartesian eigvec"""
        # global indices and slices
        i0n, igf, slall, slint, slgall, slgint = self._slices()
        i_0s, i_ns = zip(*i0n)
        if self.phys.composition is None and self.phys.lewis is None:
            pall, uall, wall, tall = slall
            pgall, ugall, wgall, tgall = slgall
        else:
            pall, uall, wall, tall = slall
            pgall, ugall, wgall, tgall, cgall = slgall

        (pmod, umod, wmod, tmod) = mode

        eigvec = np.zeros(igf[-1](i_ns[-1]) + 1) + 0j
        eigvec[pgall] = pmod[pall]
        eigvec[ugall] = umod[uall]
        eigvec[wgall] = wmod[wall]
        eigvec[tgall] = tmod[tall]

        # c_mode should be added in case of composition
        return eigvec


class LinearAnalyzer(Analyser):

    """Perform linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def neutral_ra(
        self,
        harm: float,
        ra_guess: float = 600,
        ra_comp: Optional[float] = None,
        eps: float = 1.0e-8,
    ) -> float:
        """Find Ra which gives neutral stability of a given harmonic

        harm is the wave number k or spherical harmonic degree
        """
        ra_min = ra_guess / 2
        ra_max = ra_guess * 2
        sigma_min = np.real(self.eigval(harm, ra_min, ra_comp))
        sigma_max = np.real(self.eigval(harm, ra_max, ra_comp))

        while sigma_min > 0.0 or sigma_max < 0.0:
            if sigma_min > 0.0:
                ra_max = ra_min
                ra_min /= 2
            if sigma_max < 0.0:
                ra_min = ra_max
                ra_max *= 2
            sigma_min = np.real(self.eigval(harm, ra_min, ra_comp))
            sigma_max = np.real(self.eigval(harm, ra_max, ra_comp))

        while (ra_max - ra_min) / ra_max > eps:
            ra_mean = (ra_min + ra_max) / 2
            sigma_mean = np.real(self.eigval(harm, ra_mean, ra_comp))
            if sigma_mean < 0.0:
                sigma_min = sigma_mean
                ra_min = ra_mean
            else:
                sigma_max = sigma_mean
                ra_max = ra_mean

        return (ra_min * sigma_max - ra_max * sigma_min) / (sigma_max - sigma_min)

    def fastest_mode(
        self, ra_num: float, ra_comp: Optional[float] = None, harm: float = 2
    ) -> tuple[float, float]:
        """Find the fastest growing mode at a given Ra"""

        if self.phys.spherical:
            harm = int(harm)
            harms = np.array(range(max(1, harm - 10), harm + 10))
        else:
            eps = [0.1, 0.01]
            harms = np.linspace(harm * (1 - 2 * eps[0]), harm * (1 + eps[0]), 3)

        sigma = [self.eigval(harm, ra_num, ra_comp) for harm in harms]
        if self.phys.spherical:
            max_found = False
            while not max_found:
                max_found = True
                if harms[0] != 1 and sigma[0] > sigma[1]:
                    hs_smaller = range(max(1, harms[0] - 10), harms[0])
                    s_smaller = [self.eigval(h, ra_num, ra_comp) for h in hs_smaller]
                    harms = np.array(range(hs_smaller[0], harms[-1] + 1))
                    sigma = s_smaller + sigma
                    max_found = False
                if sigma[-1] > sigma[-2]:
                    hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                    s_greater = [self.eigval(h, ra_num, ra_comp) for h in hs_greater]
                    harms = np.array(range(harms[0], hs_greater[-1] + 1))
                    sigma = sigma + s_greater
                    max_found = False
            imax = np.argmax(sigma)
            smax = sigma[imax]
            hmax = harms[imax]
        else:
            pol = np.polyfit(harms, sigma, 2)
            # maximum value
            hmax = -0.5 * pol[1] / pol[0]
            smax = self.eigval(hmax, ra_num, ra_comp)
            for i, err in enumerate([0.03, 1.0e-3]):
                while np.abs(hmax - harms[1]) > err * hmax:
                    harms = np.linspace(hmax * (1 - eps[i]), hmax * (1 + eps[i]), 3)
                    sigma = [self.eigval(h, ra_num, ra_comp) for h in harms]
                    pol = np.polyfit(harms, sigma, 2)
                    hmax = -0.5 * pol[1] / pol[0]
                    smax = sigma[1]

        return float(smax), float(hmax)

    def ran_l_mins(self) -> tuple[tuple[int, float], tuple[int, float]]:
        """Find neutral Rayleigh of mode giving square cells and of mode l=1"""
        if not self.phys.spherical:
            raise ValueError("ran_l_mins expects a spherical problem")
        lmax = 2
        rans = [self.neutral_ra(h) for h in (1, 2)]
        ranp, ran = rans
        assert self.phys.gamma is not None
        while ran <= ranp or lmax <= np.pi / (1 - self.phys.gamma):
            lmax += 1
            ranp = ran
            ran = self.neutral_ra(lmax, ranp)
            rans.append(ran)
        ran_mod1 = rans[0]
        ranlast = rans.pop()
        ranllast = rans.pop()
        loff = 0
        while ranllast < ranlast:
            loff += 1
            ranlast = ranllast
            try:
                ranllast = rans.pop()
            except IndexError:
                ranllast = ranlast + 1
        l_mod2 = lmax - loff
        ran_mod2 = ranlast
        return ((1, ran_mod1), (l_mod2, ran_mod2))

    def critical_harm(
        self, ranum: float, hguess: float, eps: float = 1e-4
    ) -> tuple[float, float, float, float]:
        """Find the wavenumbers giving a zero growth rate for a given Ra

        ranum is the Rayleigh number
        hguess is an optional inital guess for the wavenumber giving the maximum growth rate
        """
        # First find the maximum growth rate
        sigmax, hmax = self.fastest_mode(ranum, harm=hguess)
        if np.real(sigmax) < 0:
            # no need point in looking for zeros
            return sigmax, hmax, hmax, hmax

        # search zero on the plus side
        kmin = hmax
        kmax = 2 * hmax
        smin = self.eigval(kmin, ranum)
        smax = self.eigval(kmax, ranum)
        while np.real(smax) > 0 or np.real(smin) < 0:
            if np.real(smax) > 0:
                kmin = kmax
                kmax *= 2
            if np.real(smin) < 0:
                kmax = kmin
                kmin /= 2
            smin = self.eigval(kmin, ranum)
            smax = self.eigval(kmax, ranum)

        while (kmax - kmin) / kmax > eps:
            kplus = (kmin + kmax) / 2
            splus = self.eigval(kplus, ranum)
            if np.real(splus) < 0:
                kmax = kplus
                smax = splus
            else:
                kmin = kplus
                smin = splus
        kplus = float((kmin * smax - kmax * smin) / (smax - smin))

        # search zero on the minus side
        kmin = hmax / 2
        kmax = hmax
        smin = self.eigval(kmin, ranum)
        smax = self.eigval(kmax, ranum)
        while np.real(smax) < 0 or np.real(smin) > 0:
            if np.real(smax) < 0:
                kmin = kmax
                kmax *= 2
            if np.real(smin) > 0:
                kmax = kmin
                kmin /= 2
            smin = self.eigval(kmin, ranum)
            smax = self.eigval(kmax, ranum)

        while (kmax - kmin) / kmax > eps:
            kminus = (kmin + kmax) / 2
            sminus = self.eigval(kminus, ranum)
            if np.real(sminus) < 0:
                kmin = kminus
                smin = sminus
            else:
                kmax = kminus
                smax = sminus
        kminus = float((kmin * smax - kmax * smin) / (smax - smin))

        return sigmax, hmax, kminus, kplus

    def max_ra_trans_instab(
        self, hguess: float = 2, eps: float = 1e-5
    ) -> tuple[float, float, float, float, float]:
        """find maximum Ra that allows instability of the translation mode

        hguess: initial guess for the wavenumber of fastest growing mode
        eps: precision of the zero finding
        """
        # minimum value: the critcal one for translation
        assert self.phys.phi_top is not None and self.phys.phi_bot is not None
        ramin = 12 * (self.phys.phi_top + self.phys.phi_bot)
        ramax = 2 * ramin
        smin, hmin = self.fastest_mode(ramin, harm=hguess)
        smax, hmax = self.fastest_mode(ramax, harm=hmin)
        # keep the minimum values for further use
        sig0 = smin
        harm0 = hmin
        ra0 = ramin
        # make sure sigma changes sign between ramin and ramax
        while np.real(smin) < 0 or np.real(smax) > 0:
            if np.real(smin) < 0:
                ramax = ramin
                ramin /= 2
            if np.real(smax) > 0:
                ramin = ramax
                ramax *= 2
            smin, hmin = self.fastest_mode(ramin, harm=hguess)
            smax, hmax = self.fastest_mode(ramax, harm=hmin)
        # refine the ra that makes sigma change sign
        while (ramax - ramin) / ramax > eps:
            ramean = (ramin + ramax) / 2
            smean, hmean = self.fastest_mode(ramean, harm=hmin)
            if np.real(smean) < 0:
                ramax = ramean
                smax = smean
            else:
                ramin = ramean
                smin = smean
                hmin = hmean
        rastab = (ramin * smax - ramax * smin) / (smax - smin)
        hstab = self.fastest_mode(rastab, harm=hmin)[1]
        return rastab, hstab, ra0, harm0, sig0

    def critical_ra(
        self, harm: float = 2, ra_guess: float = 600, ra_comp: Optional[float] = None
    ) -> tuple[float, float]:
        """Find the harmonic with the lowest neutral Ra

        harm is an optional initial guess
        ra_guess is a guess for Ra_c
        """
        # find 3 values of Ra for 3 different harmonics
        eps = [0.1, 0.01]
        if self.phys.spherical:
            harm = int(harm)
            harms = np.array(range(max(1, harm - 10), harm + 10))
        else:
            harms = np.linspace(harm * (1 - eps[0]), harm * (1 + 2 * eps[0]), 3)
        ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]

        if self.phys.spherical:
            min_found = False
            while not min_found:
                min_found = True
                if harms[0] != 1 and ray[0] < ray[1]:
                    hs_smaller = range(max(1, harms[0] - 10), harms[0])
                    ray_smaller = [
                        self.neutral_ra(h, ray[0], ra_comp) for h in hs_smaller
                    ]
                    harms = np.array(range(hs_smaller[0], harms[-1] + 1))
                    ray = ray_smaller + ray
                    min_found = False
                if ray[-1] < ray[-2]:
                    hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                    ray_greater = [
                        self.neutral_ra(h, ray[-1], ra_comp) for h in hs_greater
                    ]
                    harms = np.array(range(harms[0], hs_greater[-1] + 1))
                    ray = ray + ray_greater
                    min_found = False
            imin = np.argmin(ray)
            ra_guess = ray[imin]
            hmin = harms[imin]
        else:
            # fit a degree 2 polynomial
            pol = np.polyfit(harms, ray, 2)
            # minimum value
            exitloop = False
            kmin = -0.5 * pol[1] / pol[0]
            for i, err in enumerate([0.03, 1.0e-3]):
                while np.abs(kmin - harms[1]) > err * kmin and not exitloop:
                    harms = np.linspace(kmin * (1 - eps[i]), kmin * (1 + eps[i]), 3)
                    ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]
                    pol = np.polyfit(harms, ray, 2)
                    kmin = -0.5 * pol[1] / pol[0]
                    ra_guess = ray[1]
            hmin = kmin

        return ra_guess, hmin
