from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from dmsuite.poly_diff import Chebyshev, DiffMatOnDomain, DiffMatrices

from . import physics as phy
from .geometry import CartOps
from .matrix import All, Bulk, EigenvalueProblem, Field, Matrix, Slices, VarSpec
from .ref_prof import DiffusiveProf, Dirichlet

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence

    from numpy.typing import NDArray

    from .geometry import Operators
    from .matrix import Vector


@dataclass(frozen=True)
class CartStability:
    """Linear stability analysis in cartesian geometry."""

    chebyshev_degree: int
    temperature: phy.AdvDiffEq = phy.AdvDiffEq(
        bc_top=phy.Zero(),
        bc_bot=phy.Zero(),
        ref_prof=DiffusiveProf(bcs_top=Dirichlet(0.0), bcs_bot=Dirichlet(1.0)),
    )
    composition: Optional[phy.AdvDiffEq] = None
    bc_mom_top: phy.BCMomentum = phy.FreeSlip()
    bc_mom_bot: phy.BCMomentum = phy.FreeSlip()
    prandtl: Optional[float] = None
    ref_state_translation: bool = False
    water: bool = False
    thetar: float = 0.0

    def name(self) -> str:
        """Construct a name for the current case"""
        name = [
            "cart",
            "TOP",
            self.bc_mom_top.name,
            "BOT",
            self.bc_mom_bot.name,
        ]
        return "_".join(name).replace(".", "-")

    def var_specs(self) -> Sequence[VarSpec]:
        common = [Field(var="p"), Field(var="u"), Field(var="w")]
        if self.temperature is not None:
            common.append(Field(var="T"))
        if self.composition is not None:
            common.append(Field(var="c"))
        return common

    @cached_property
    def _diff_mat(self) -> DiffMatrices:
        # flip xmin and xmax because Chebyshev nodes are in
        # decreasing order, and interpolation methods from
        # dmsuite depend on that.
        return DiffMatOnDomain(
            xmin=0.5,
            xmax=-0.5,
            dmat=Chebyshev(degree=self.chebyshev_degree),
        )

    @property
    def nodes(self) -> NDArray:
        return self._diff_mat.nodes

    def operators(self, harmonic: float) -> Operators:
        return CartOps(
            diff_mat=self._diff_mat,
            wavenumber=harmonic,
            eta_r=np.identity(self.nodes.size),
        )

    @cached_property
    def slices(self) -> Slices:
        return Slices(var_specs=self.var_specs(), nnodes=self.nodes.size)

    def eigen_problem(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> EigenvalueProblem:
        prandtl = self.prandtl
        translation = self.ref_state_translation

        # FIXME: viscosity in cartesian
        ops = self.operators(harm)
        dz1 = ops.grad_r
        one = ops.identity

        lmat = Matrix(self.slices, dtype=np.complex128)
        rmat = Matrix(self.slices)

        # Pressure equations
        # mass conservation
        lmat.add_term(All("p"), ops.diff_h, "u")
        lmat.add_term(All("p"), dz1, "w")

        # Momentum equation
        self.bc_mom_top.add_top(lmat, ops)
        self.bc_mom_bot.add_bot(lmat, ops)
        # horizontal momentum conservation
        lmat.add_term(Bulk("u"), -ops.diff_h, "p")
        lmat.add_term(Bulk("u"), ops.lapl, "u")
        # vertical momentum conservation
        lmat.add_term(Bulk("w"), -dz1, "p")
        lmat.add_term(Bulk("w"), ops.lapl, "w")
        # buoyancy
        if self.water:
            # FIXME: abstract this away, this would also deserve its own
            # ScalarField implementation for temperature (cooled from below)
            theta0 = self.thetar - ops.phys_coord
            lmat.add_term(Bulk("w"), -ra_num * np.diag(theta0), "T")
        else:
            lmat.add_term(Bulk("w"), ra_num * one, "T")
        if self.composition is not None:
            assert ra_comp is not None
            lmat.add_term(Bulk("w"), ra_comp * one, "c")

        if translation:
            # only written for Dirichlet BCs on T and without internal heating
            # FIXME: abstract away translation case with AdvDiffEq formalism
            phi_top = self.phys.bc_mom_top.phase_number  # type: ignore
            phi_bot = self.phys.bc_mom_bot.phase_number  # type: ignore
            rtr = 12 * (phi_top + phi_bot)
            wtrans = phy.wtran((ra_num - rtr) / rtr)[0]
            self.temperature.bc_top.add_top("T", lmat, ops)
            self.temperature.bc_bot.add_bot("T", lmat, ops)
            lmat.add_term(Bulk("T"), ops.lapl, "T")
            lmat.add_term(Bulk("T"), -wtrans * dz1, "T")
            w_temp = np.diag(np.exp(wtrans * self.nodes))
            if np.abs(wtrans) > 1.0e-3:
                w_temp *= wtrans / (2 * np.sinh(wtrans / 2))
            else:
                # use a limited development
                w_temp *= 1 - wtrans**2 / 24
            lmat.add_term(Bulk("T"), w_temp, "w")
        else:
            self.temperature.add_pert_eq("T", lmat, ops)

        rmat.add_term(Bulk("T"), one, "T")
        if prandtl is not None:
            # finite Prandtl number case
            rmat.add_term(Bulk("u"), one / prandtl, "u")
            rmat.add_term(Bulk("w"), one / prandtl, "w")

        # C equations
        if self.composition is not None:
            self.composition.add_pert_eq("c", lmat, ops)
            rmat.add_term(Bulk("c"), one, "c")
        return EigenvalueProblem(lmat, rmat)

    def growth_rate(
        self, harm: float, ra_num: float, ra_comp: Optional[float] = None
    ) -> np.floating:
        return np.real(self.eigen_problem(harm, ra_num, ra_comp).max_eigval())

    def split_mode(self, eigvec: Vector) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        p_mode = eigvec.extract("p")
        u_mode = eigvec.extract("u")
        w_mode = eigvec.extract("w")
        t_mode = eigvec.extract("T")
        # c_mode should be added in case of composition
        return (p_mode, u_mode, w_mode, t_mode)

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
        sigma_min = self.growth_rate(harm, ra_min, ra_comp)
        sigma_max = self.growth_rate(harm, ra_max, ra_comp)

        while sigma_min > 0.0 or sigma_max < 0.0:
            if sigma_min > 0.0:
                ra_max = ra_min
                ra_min /= 2
            if sigma_max < 0.0:
                ra_min = ra_max
                ra_max *= 2
            sigma_min = self.growth_rate(harm, ra_min, ra_comp)
            sigma_max = self.growth_rate(harm, ra_max, ra_comp)

        while (ra_max - ra_min) / ra_max > eps:
            ra_mean = (ra_min + ra_max) / 2
            sigma_mean = self.growth_rate(harm, ra_mean, ra_comp)
            if sigma_mean < 0.0:
                sigma_min = sigma_mean
                ra_min = ra_mean
            else:
                sigma_max = sigma_mean
                ra_max = ra_mean

        return float(
            (ra_min * sigma_max - ra_max * sigma_min) / (sigma_max - sigma_min)
        )

    def fastest_mode(
        self, ra_num: float, ra_comp: Optional[float] = None, harm: float = 2.0
    ) -> tuple[float, float]:
        """Find the fastest growing mode at a given Ra"""

        eps = [0.1, 0.01]
        harms = np.linspace(harm * (1 - 2 * eps[0]), harm * (1 + eps[0]), 3)

        sigma = [self.growth_rate(harm, ra_num, ra_comp) for harm in harms]
        pol = np.polyfit(harms, sigma, 2)
        # maximum value
        hmax = -0.5 * pol[1] / pol[0]
        smax = self.growth_rate(hmax, ra_num, ra_comp)
        for i, err in enumerate([0.03, 1.0e-3]):
            while np.abs(hmax - harms[1]) > err * hmax:
                harms = np.linspace(hmax * (1 - eps[i]), hmax * (1 + eps[i]), 3)
                sigma = [self.growth_rate(h, ra_num, ra_comp) for h in harms]
                pol = np.polyfit(harms, sigma, 2)
                hmax = -0.5 * pol[1] / pol[0]
                smax = sigma[1]

        return float(smax), float(hmax)

    def critical_harm(
        self, ranum: float, hguess: float, eps: float = 1e-4
    ) -> tuple[float, float, float, float]:
        """Find the wavenumbers giving a zero growth rate for a given Ra

        ranum is the Rayleigh number
        hguess is an optional inital guess for the wavenumber giving the maximum growth rate
        """
        # First find the maximum growth rate
        sigmax, hmax = self.fastest_mode(ranum, harm=hguess)
        if sigmax < 0:
            # no need point in looking for zeros
            return sigmax, hmax, hmax, hmax

        # search zero on the plus side
        kmin = hmax
        kmax = 2 * hmax
        smin = self.growth_rate(kmin, ranum)
        smax = self.growth_rate(kmax, ranum)
        while smax > 0 or smin < 0:
            if smax > 0:
                kmin = kmax
                kmax *= 2
            if smin < 0:
                kmax = kmin
                kmin /= 2
            smin = self.growth_rate(kmin, ranum)
            smax = self.growth_rate(kmax, ranum)

        while (kmax - kmin) / kmax > eps:
            kplus = (kmin + kmax) / 2
            splus = self.growth_rate(kplus, ranum)
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
        smin = self.growth_rate(kmin, ranum)
        smax = self.growth_rate(kmax, ranum)
        while smax < 0 or smin > 0:
            if smax < 0:
                kmin = kmax
                kmax *= 2
            if smin > 0:
                kmax = kmin
                kmin /= 2
            smin = self.growth_rate(kmin, ranum)
            smax = self.growth_rate(kmax, ranum)

        while (kmax - kmin) / kmax > eps:
            kminus = (kmin + kmax) / 2
            sminus = self.growth_rate(kminus, ranum)
            if sminus < 0:
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
        phi_top = self.phys.bc_mom_top.phase_number  # type: ignore
        phi_bot = self.phys.bc_mom_bot.phase_number  # type: ignore
        ramin = 12 * (phi_top + phi_bot)
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
        self, harm: float = 2.0, ra_guess: float = 600, ra_comp: Optional[float] = None
    ) -> tuple[float, float]:
        """Find the harmonic with the lowest neutral Ra

        harm is an optional initial guess
        ra_guess is a guess for Ra_c
        """
        # find 3 values of Ra for 3 different harmonics
        eps = [0.1, 0.01]
        harms = np.linspace(harm * (1 - eps[0]), harm * (1 + 2 * eps[0]), 3)
        ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]

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
