from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from dmsuite.poly_diff import Chebyshev, DiffMatOnDomain, DiffMatrices

from . import physics as phy
from .geometry import SphOps
from .matrix import Bulk, EigenvalueProblem, Field, Matrix, Slices, VarSpec
from .ref_prof import DiffusiveProf, Dirichlet

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Sequence

    from numpy.typing import NDArray

    from .geometry import Operators
    from .matrix import Vector


@dataclass(frozen=True)
class SphStability:
    """Linear stability analysis in spherical geometry."""

    chebyshev_degree: int
    gamma: float
    temperature: Optional[phy.AdvDiffEq] = phy.AdvDiffEq(
        bc_top=phy.Zero(),
        bc_bot=phy.Zero(),
        ref_prof=DiffusiveProf(bcs_top=Dirichlet(0.0), bcs_bot=Dirichlet(1.0)),
    )
    composition: Optional[phy.AdvDiffEq] = None
    bc_mom_top: phy.BCMomentum = phy.FreeSlip()
    bc_mom_bot: phy.BCMomentum = phy.FreeSlip()
    eta_r: Optional[Callable[[NDArray], NDArray]] = None
    cooling_smo: Optional[tuple[Callable, Callable]] = None
    frozen_time: bool = False

    def name(self) -> str:
        """Construct a name for the current case"""
        name = [
            "sph",
            str(self.gamma),
            "TOP",
            self.bc_mom_top.name,
            "BOT",
            self.bc_mom_bot.name,
        ]
        return "_".join(name).replace(".", "-")

    def var_specs(self) -> Sequence[VarSpec]:
        common = [Field(var="p"), Field(var="q")]
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
            xmin=2.0,
            xmax=1.0,
            dmat=Chebyshev(degree=self.chebyshev_degree),
        )

    @property
    def nodes(self) -> NDArray:
        return self._diff_mat.nodes

    @cached_property
    def _visco_as_mat(self) -> NDArray:
        if self.eta_r is not None:
            return np.diag(self.eta_r(self.nodes))
        return np.identity(self.nodes.size)

    def operators(self, harmonic: int) -> Operators:
        return SphOps(
            gamma=self.gamma,
            diff_mat=self._diff_mat,
            harm_degree=harmonic,
            eta_r=self._visco_as_mat,
        )

    @cached_property
    def slices(self) -> Slices:
        return Slices(var_specs=self.var_specs(), nnodes=self.nodes.size)

    def eigen_problem(
        self, l_harm: int, ra_num: Optional[float], ra_comp: Optional[float] = None
    ) -> EigenvalueProblem:
        gamma = self.gamma

        ops = self.operators(l_harm)
        rad = self.nodes
        dr1, dr2 = ops.diff_r(1), ops.diff_r(2)
        eta_r = ops.viscosity

        # r + lambda
        ral = ops.phys_coord
        # 1 / (r + lambda)
        orl1 = (1 - gamma) / ((1 - gamma) * rad + 2 * gamma - 1)
        orl2 = orl1**2
        orl3 = orl1**3
        one = ops.identity

        rad = np.diag(rad)
        ral = np.diag(ral)
        orl1 = np.diag(orl1)
        orl2 = np.diag(orl2)
        orl3 = np.diag(orl3)

        lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
        temp_terms = self.temperature is not None

        if temp_terms and ra_num is None:
            raise ValueError("Temperature effect requires ra_num")
        if (not temp_terms) and self.composition is None:
            raise ValueError("No buoyancy terms!")

        lmat = Matrix(self.slices)
        rmat = Matrix(self.slices)

        # Momentum equations
        self.bc_mom_top.add_top(lmat, ops)
        self.bc_mom_bot.add_bot(lmat, ops)
        # Poloidal potential equations
        # laplacian(P) - Q = 0
        lmat.add_term(Bulk("p"), ops.lapl, "p")
        lmat.add_term(Bulk("p"), -one, "q")

        # Q equations
        if self.eta_r is not None:
            deta_dr = np.diag(np.dot(dr1, np.diag(eta_r)))
            d2eta_dr2 = np.diag(np.dot(dr2, np.diag(eta_r)))
            lmat.add_term(
                Bulk("q"),
                2 * (lh2 - 1) * (np.dot(orl2, d2eta_dr2) - np.dot(orl3, deta_dr))
                - 2 * np.dot(np.dot(orl1, d2eta_dr2) - np.dot(orl2, deta_dr), dr1),
                "p",
            )
            lmat.add_term(
                Bulk("q"), (eta_r @ ops.lapl) + d2eta_dr2 + 2 * (deta_dr @ dr1), "q"
            )
        else:
            # laplacian(Q) - RaT/r = 0
            lmat.add_term(Bulk("q"), ops.lapl, "q")
        if temp_terms:
            assert ra_num is not None
            lmat.add_term(Bulk("q"), -ra_num * orl1, "T")
        if self.composition is not None:
            assert ra_comp is not None
            lmat.add_term(Bulk("q"), -ra_comp * orl1, "c")

        if self.cooling_smo is not None:
            gamt_f, w_f = self.cooling_smo
            gam2_smo = gamt_f(gamma) ** 2
            w_smo = w_f(gamma)

        # T equations
        # laplacian(T) - u.grad(T_conductive) = sigma T
        if self.temperature is not None:
            self.temperature.add_pert_eq("T", lmat, ops)

            if not self.frozen_time and self.cooling_smo is not None:
                # FIXME: define proper operators for the moving-front approach
                grad_tcond = ops.grad_r @ self.temperature.ref_prof.eval_with(ops)
                grad_ref_temp_top = grad_tcond[0]
                lmat.add_term(
                    Bulk("T"),
                    w_smo * (np.dot(rad - one, dr1) + grad_ref_temp_top * one),
                    "T",
                )

            if not self.frozen_time and self.cooling_smo:
                rmat.add_term(Bulk("T"), one * gam2_smo, "T")
            else:
                rmat.add_term(Bulk("T"), one, "T")

        # C equations
        if self.composition is not None:
            self.composition.add_pert_eq("c", lmat, ops)
            if not self.frozen_time and self.cooling_smo is not None:
                lmat.add_term(Bulk("c"), w_smo * np.dot(rad - one, dr1), "c")
                rmat.add_term(Bulk("c"), one * gam2_smo, "c")
            else:
                rmat.add_term(Bulk("c"), one, "c")

        return EigenvalueProblem(lmat, rmat)

    def growth_rate(
        self, harm: int, ra_num: Optional[float], ra_comp: Optional[float] = None
    ) -> np.floating:
        return np.real(self.eigen_problem(harm, ra_num, ra_comp).max_eigval())

    def split_mode(
        self, eigvec: Vector, l_harm: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        p_mode = eigvec.extract("p")
        t_mode = eigvec.extract("T")

        ops = self.operators(l_harm)

        orl1 = 1 / ops.phys_coord
        ur_mode = l_harm * (l_harm + 1) * p_mode * orl1
        up_mode = 1j * l_harm * (ops.diff_r(1) @ p_mode + p_mode * orl1)

        return (p_mode, up_mode, ur_mode, t_mode)

    def neutral_ra(
        self,
        harm: int,
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
        self, ra_num: float, ra_comp: Optional[float] = None, harm: int = 2
    ) -> tuple[float, int]:
        """Find the fastest growing mode at a given Ra"""

        harms = np.array(range(max(1, harm - 10), harm + 10))

        sigma = [self.growth_rate(harm, ra_num, ra_comp) for harm in harms]
        max_found = False
        while not max_found:
            max_found = True
            if harms[0] != 1 and sigma[0] > sigma[1]:
                hs_smaller = range(max(1, harms[0] - 10), harms[0])
                s_smaller = [self.growth_rate(h, ra_num, ra_comp) for h in hs_smaller]
                harms = np.array(range(hs_smaller[0], harms[-1] + 1))
                sigma = s_smaller + sigma
                max_found = False
            if sigma[-1] > sigma[-2]:
                hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                s_greater = [self.growth_rate(h, ra_num, ra_comp) for h in hs_greater]
                harms = np.array(range(harms[0], hs_greater[-1] + 1))
                sigma = sigma + s_greater
                max_found = False
        imax = np.argmax(sigma)
        smax = sigma[imax]
        hmax = harms[imax]
        return float(smax), hmax

    def ran_l_mins(self) -> tuple[tuple[int, float], tuple[int, float]]:
        """Find neutral Rayleigh of mode giving square cells and of mode l=1"""
        lmax = 2
        rans = [self.neutral_ra(h) for h in (1, 2)]
        ranp, ran = rans
        while ran <= ranp or lmax <= np.pi / (1 - self.gamma):
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

    def critical_ra(
        self, harm: int = 2, ra_guess: float = 600, ra_comp: Optional[float] = None
    ) -> tuple[float, int]:
        """Find the harmonic with the lowest neutral Ra

        harm is an optional initial guess
        ra_guess is a guess for Ra_c
        """
        # find 3 values of Ra for 3 different harmonics
        harms = np.array(range(max(1, harm - 10), harm + 10))
        ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]

        min_found = False
        while not min_found:
            min_found = True
            if harms[0] != 1 and ray[0] < ray[1]:
                hs_smaller = range(max(1, harms[0] - 10), harms[0])
                ray_smaller = [self.neutral_ra(h, ray[0], ra_comp) for h in hs_smaller]
                harms = np.array(range(hs_smaller[0], harms[-1] + 1))
                ray = ray_smaller + ray
                min_found = False
            if ray[-1] < ray[-2]:
                hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                ray_greater = [self.neutral_ra(h, ray[-1], ra_comp) for h in hs_greater]
                harms = np.array(range(harms[0], hs_greater[-1] + 1))
                ray = ray + ray_greater
                min_found = False
        imin = np.argmin(ray)
        ra_guess = ray[imin]
        hmin = harms[imin]

        return ra_guess, hmin
