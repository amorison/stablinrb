from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.ma as ma
from dmsuite.poly_diff import Chebyshev, DiffMatOnDomain
from scipy import linalg

from .geometry import Spherical
from .matrix import All, Bot, Bulk, Matrix, Slices, Top, Vector
from .physics import wtran

if typing.TYPE_CHECKING:
    from typing import Optional

    from dmsuite.poly_diff import DiffMatrices
    from numpy.typing import NDArray

    from .physics import PhysicalProblem, RadialOperators


def cartesian_matrices(
    self: LinearAnalyzer, wnk: float, ra_num: float, ra_comp: Optional[float] = None
) -> tuple[Matrix, Matrix]:
    """Build left- and right-hand-side matrices in cartesian geometry case"""
    # parameters
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    prandtl = self.phys.prandtl
    translation = self.phys.ref_state_translation
    assert self.phys.temperature is not None

    operators = self.operators
    dz1 = operators.grad_r
    one = operators.identity
    # horizontal derivative
    dh1 = 1j * wnk * one
    # Laplace operator
    lapl = operators.lapl_r - wnk**2 * one

    if translation:
        assert phi_bot is not None and phi_top is not None
        rtr = 12 * (phi_top + phi_bot)
        wtrans = wtran((ra_num - rtr) / rtr)[0]

    lmat = Matrix(self.slices, dtype=np.complex128)
    rmat = Matrix(self.slices)

    # Pressure equations
    # mass conservation
    lmat.add_term(All("p"), dh1, "u")
    lmat.add_term(All("p"), dz1, "w")

    # U equations
    # free-slip at top
    if phi_top is not None or freeslip_top:
        lmat.add_term(Top("u"), dz1, "u")
    if phi_top is not None:
        lmat.add_term(Top("u"), dh1, "w")
    # horizontal momentum conservation
    lmat.add_term(Bulk("u"), -dh1, "p")
    lmat.add_term(Bulk("u"), lapl, "u")
    # free-slip at bot
    if phi_bot is not None or freeslip_bot:
        lmat.add_term(Bot("u"), dz1, "u")
    if phi_bot is not None:
        lmat.add_term(Bot("u"), dh1, "w")

    # W equations
    if phi_top is not None:
        # phase change at top
        lmat.add_term(Top("w"), -one, "p")
        lmat.add_term(Top("w"), phi_top * one + 2 * dz1, "w")
    # vertical momentum conservation
    lmat.add_term(Bulk("w"), -dz1, "p")
    lmat.add_term(Bulk("w"), lapl, "w")
    if self.phys.water:
        # FIXME: abstract this away, this would also deserve its own
        # ScalarField implementation for temperature (cooled from below)
        theta0 = self.phys.thetar - operators.phys_coord
        lmat.add_term(Bulk("w"), -ra_num * np.diag(theta0), "T")
    else:
        lmat.add_term(Bulk("w"), ra_num * one, "T")
    if self.phys.composition is not None:
        assert ra_comp is not None
        lmat.add_term(Bulk("w"), ra_comp * one, "c")
    if phi_bot is not None:
        # phase change at bot
        lmat.add_term(Bot("w"), -one, "p")
        lmat.add_term(Bot("w"), -phi_bot * one + 2 * dz1, "w")

    self.phys.temperature.add_bcs("T", lmat, operators)
    lmat.add_term(Bulk("T"), lapl, "T")

    # need to take heat flux into account in T conductive
    if translation:
        # only written for Dirichlet BCs on T and without internal heating
        lmat.add_term(Bulk("T"), -wtrans * dz1, "T")
        w_temp = np.diag(np.exp(wtrans * self.nodes))
        if np.abs(wtrans) > 1.0e-3:
            w_temp *= wtrans / (2 * np.sinh(wtrans / 2))
        else:
            # use a limited development
            w_temp *= 1 - wtrans**2 / 24
        lmat.add_term(Bulk("T"), w_temp, "w")
    else:
        grad_tcond = dz1 @ self.phys.temperature.ref_profile(operators)
        lmat.add_term(Bulk("T"), -np.diag(grad_tcond), "w")

    rmat.add_term(Bulk("T"), one, "T")
    if prandtl is not None:
        # finite Prandtl number case
        rmat.add_term(Bulk("u"), one / prandtl, "u")
        rmat.add_term(Bulk("w"), one / prandtl, "w")
    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if self.phys.composition is not None:
        grad_c_ref = operators.grad_r @ self.phys.composition.ref_profile(operators)
        lmat.add_term(Bulk("c"), -np.diag(grad_c_ref), "w")
        if self.phys.lewis is not None:
            lmat.add_term(Bulk("c"), lapl / self.phys.lewis, "c")
        rmat.add_term(Bulk("c"), one, "c")
    return lmat, rmat


def spherical_matrices(
    self: LinearAnalyzer,
    l_harm: int,
    ra_num: Optional[float] = None,
    ra_comp: Optional[float] = None,
) -> tuple[Matrix, Matrix]:
    """Build left and right matrices in spherical case"""
    # FIXME: delegate to a geometry-aware object
    assert isinstance(self.phys.geometry, Spherical)
    gamma = self.phys.geometry.gamma
    operators = self.operators
    rad = self.nodes
    dr1, dr2 = self.diff_mat(1), self.diff_mat(2)

    # r + lambda
    ral = operators.phys_coord
    # 1 / (r + lambda)
    orl1 = (1 - gamma) / ((1 - gamma) * rad + 2 * gamma - 1)
    orl2 = orl1**2
    orl3 = orl1**3
    one = operators.identity

    rad = np.diag(rad)
    ral = np.diag(ral)
    orl1 = np.diag(orl1)
    orl2 = np.diag(orl2)
    orl3 = np.diag(orl3)

    lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
    lapl = operators.lapl_r - lh2 * orl2  # laplacian
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

    temp_terms = self.phys.temperature is not None

    if self.phys.eta_r is not None:
        eta_r = np.diag(np.vectorize(self.phys.eta_r)(np.diag(ral)))
    else:
        eta_r = one

    if temp_terms and ra_num is None:
        raise ValueError("Temperature effect requires ra_num")
    if (not temp_terms) and self.phys.composition is None:
        raise ValueError("No buoyancy terms!")

    lmat = Matrix(self.slices)
    rmat = Matrix(self.slices)

    # Poloidal potential equations
    if phi_top is not None:
        # free-slip at top
        lmat.add_term(Top("p"), dr2 + (lh2 * (1 + C_top) - 2) * orl2, "p")
    else:
        # no radial velocity, Dirichlet condition but
        # need to keep boundary point to ensure free-slip
        # or rigid boundary
        lmat.add_term(Top("p"), one, "p")
    # laplacian(P) - Q = 0
    lmat.add_term(Bulk("p"), lapl, "p")
    lmat.add_term(Bulk("p"), -one, "q")
    if phi_bot is not None:
        # free-slip at bot
        lmat.add_term(Bot("p"), dr2 + (lh2 * (1 - C_bot) - 2) * orl2, "p")
    else:
        lmat.add_term(Bot("p"), one, "p")

    # Q equations
    # normal stress continuity at top
    if phi_top is not None:
        lmat.add_term(
            Top("q"),
            lh2
            * (
                (phi_top - 2 * C_top * (1 - gamma)) * orl1
                - 2 * np.dot(eta_r, orl2)
                + 2 * np.dot(eta_r, np.dot(orl1, dr1))
            ),
            "p",
        )
        lmat.add_term(Top("q"), -eta_r - np.dot(eta_r, np.dot(ral, dr1)), "q")
    elif freeslip_top:
        lmat.add_term(Top("q"), dr2, "p")
    else:
        # rigid
        lmat.add_term(Top("q"), dr1, "p")
    if self.phys.eta_r is not None:
        deta_dr = np.diag(np.dot(dr1, np.diag(eta_r)))
        d2eta_dr2 = np.diag(np.dot(dr2, np.diag(eta_r)))
        lmat.add_term(
            Bulk("q"),
            2 * (lh2 - 1) * (np.dot(orl2, d2eta_dr2) - np.dot(orl3, deta_dr))
            - 2 * np.dot(np.dot(orl1, d2eta_dr2) - np.dot(orl2, deta_dr), dr1),
            "p",
        )
        lmat.add_term(
            Bulk("q"), np.dot(eta_r, lapl) + d2eta_dr2 + 2 * np.dot(deta_dr, dr1), "q"
        )
    else:
        # laplacian(Q) - RaT/r = 0
        lmat.add_term(Bulk("q"), lapl, "q")
    if temp_terms:
        assert ra_num is not None
        lmat.add_term(Bulk("q"), -ra_num * orl1, "T")
    if self.phys.composition is not None:
        assert ra_comp is not None
        lmat.add_term(Bulk("q"), -ra_comp * orl1, "c")
    # normal stress continuity at bot
    if phi_bot is not None:
        lmat.add_term(
            Bot("q"),
            lh2
            * (
                -(phi_bot - 2 * C_bot * (1 - gamma) / gamma) * orl1
                - 2 * np.dot(eta_r, orl2)
                + 2 * np.dot(eta_r, np.dot(orl1, dr1))
            ),
            "p",
        )
        lmat.add_term(Bot("q"), -eta_r - np.dot(eta_r, np.dot(ral, dr1)), "q")
    elif freeslip_bot:
        lmat.add_term(Bot("q"), dr2, "p")
    else:
        # rigid
        lmat.add_term(Bot("q"), dr1, "p")

    if self.phys.cooling_smo is not None:
        gamt_f, w_f = self.phys.cooling_smo
        gam2_smo = gamt_f(gamma) ** 2
        w_smo = w_f(gamma)

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T
    if self.phys.temperature is not None:
        self.phys.temperature.add_bcs("T", lmat, operators)
        lmat.add_term(Bulk("T"), lapl, "T")

        # advection of reference profile
        # using u_r = l(l+1)/r P
        # first compute - 1/r * nabla T
        # then multiply by l(l+1)
        grad_tcond = operators.grad_r @ self.phys.temperature.ref_profile(operators)
        if not self.phys.frozen_time and self.phys.cooling_smo is not None:
            grad_ref_temp_top = grad_tcond[0]
            lmat.add_term(
                Bulk("T"),
                w_smo * (np.dot(rad - one, dr1) + grad_ref_temp_top * one),
                "T",
            )
        grad_tcond *= -np.diag(orl1)
        lmat.add_term(Bulk("T"), np.diag(lh2 * grad_tcond), "p")

        if not self.phys.frozen_time and self.phys.cooling_smo:
            rmat.add_term(Bulk("T"), one * gam2_smo, "T")
        else:
            rmat.add_term(Bulk("T"), one, "T")

    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if self.phys.composition is not None:
        grad_comp = operators.grad_r @ self.phys.composition.ref_profile(operators)
        lmat.add_term(Bulk("c"), -lh2 * np.diag(orl1 @ grad_comp), "p")
        if self.phys.lewis is not None:
            lmat.add_term(Bulk("c"), lapl / self.phys.lewis, "c")

        if not self.phys.frozen_time and self.phys.cooling_smo is not None:
            lmat.add_term(Bulk("c"), w_smo * np.dot(rad - one, dr1), "c")
            rmat.add_term(Bulk("c"), one * gam2_smo, "c")
        else:
            rmat.add_term(Bulk("c"), one, "c")

    return lmat, rmat


@dataclass(frozen=True)
class LinearAnalyzer:
    """Linear stability analysis of a Rayleigh-Benard problem."""

    phys: PhysicalProblem
    chebyshev_degree: int

    @cached_property
    def _diff_mat(self) -> DiffMatrices:
        xmin, xmax = self.phys.domain_bounds
        # flip xmin and xmax because Chebyshev nodes are in
        # decreasing order, and interpolation methods from
        # dmsuite depend on that.
        return DiffMatOnDomain(
            xmin=xmax,
            xmax=xmin,
            dmat=Chebyshev(degree=self.chebyshev_degree),
        )

    @property
    def nodes(self) -> NDArray:
        return self._diff_mat.nodes

    @cached_property
    def operators(self) -> RadialOperators:
        return self.phys.geometry.with_dmat(self._diff_mat)

    def diff_mat(self, order: int) -> NDArray:
        return self._diff_mat.at_order(order)

    @cached_property
    def slices(self) -> Slices:
        return Slices(var_specs=self.phys.var_specs(), nnodes=self.nodes.size)

    def matrices(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> tuple[Matrix, Matrix]:
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
        eigvals = linalg.eigvals(lmat.array(), rmat.array())
        return np.max(np.real(ma.masked_invalid(eigvals)))

    def eigvec(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> tuple[np.complexfloating, Vector]:
        """Compute the max eigenvalue and associated eigenvector

        harm: wave number
        ra_num: thermal Rayleigh number
        ra_comp: compositional Ra
        """
        lmat, rmat = self.matrices(harm, ra_num, ra_comp)
        eigvals, eigvecs = linalg.eig(lmat.array(), rmat.array())
        iegv = np.argmax(np.real(ma.masked_invalid(eigvals)))
        return eigvals[iegv], Vector(slices=lmat.slices, arr=eigvecs[:, iegv])

    def _split_mode_cartesian(
        self, eigvec: Vector
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Split 1D cartesian mode into (p, u, w, t) tuple."""
        p_mode = eigvec.extract("p")
        u_mode = eigvec.extract("u")
        w_mode = eigvec.extract("w")
        t_mode = eigvec.extract("T")
        # c_mode should be added in case of composition
        return (p_mode, u_mode, w_mode, t_mode)

    def _split_mode_spherical(
        self, eigvec: Vector, l_harm: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Split 1D spherical mode into (p, u, w, t) tuple."""
        p_mode = eigvec.extract("p")
        t_mode = eigvec.extract("T")

        # FIXME: delegate those computations to a geometry-aware object
        geom = self.phys.geometry
        assert isinstance(geom, Spherical)
        orl1 = (1 - geom.gamma) / ((1 - geom.gamma) * self.nodes + 2 * geom.gamma - 1)
        ur_mode = l_harm * (l_harm + 1) * p_mode * orl1
        up_mode = 1j * l_harm * (self.diff_mat(1) @ p_mode + p_mode * orl1)

        return (p_mode, up_mode, ur_mode, t_mode)

    def split_mode(
        self, eigvec: Vector, harm: float
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Generic splitting function"""
        if self.phys.spherical:
            return self._split_mode_spherical(eigvec, int(harm))
        else:
            return self._split_mode_cartesian(eigvec)

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
        # FIXME: delegate to a geometry-aware object
        geom = self.phys.geometry
        assert isinstance(geom, Spherical)
        while ran <= ranp or lmax <= np.pi / (1 - geom.gamma):
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
