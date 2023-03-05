from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property
from itertools import repeat

import numpy as np
from numpy.linalg import lstsq, solve

from .analyzer import LinearAnalyzer
from .matrix import All, Bot, Bulk, Field, Matrix, Scalar, Single, Slices, Top, Vector

if typing.TYPE_CHECKING:
    from typing import Mapping, Sequence

    from numpy.typing import NDArray

    from .matrix import VarSpec
    from .physics import PhysicalProblem


@dataclass(frozen=True)
class ModeIndex:
    max_order: int

    @cached_property
    def _all_ord_harm(self) -> Sequence[tuple[int, int]]:
        ord_harms: list[tuple[int, int]] = []
        for n in range(1, self.max_order + 1):
            indices = range(n % 2, n + 1, 2)
            ord_harms.extend(zip(repeat(n), indices))
        return ord_harms

    @cached_property
    def _oh_to_i(self) -> Mapping[tuple[int, int], int]:
        return {oh: i for i, oh in enumerate(self._all_ord_harm)}

    @property
    def n_harmonics(self) -> int:
        return len(self._all_ord_harm)

    def ord_harm(self, index: int) -> tuple[int, int]:
        return self._all_ord_harm[index]

    def index(self, order: int, harmonic: int) -> int:
        return self._oh_to_i[(order, harmonic)]


@dataclass
class NonLinearAnalyzer:

    """Weakly non-linear analysis.

    Attributes:
        phys: physical problem.
        ncheb: degree of Chebyshev polynomials.
        nnonlin: maximum order of non-linear analysis
    """

    phys: PhysicalProblem
    ncheb: int
    nnonlin: int

    @cached_property
    def linear_analyzer(self) -> LinearAnalyzer:
        return LinearAnalyzer(self.phys, self.ncheb)

    @cached_property
    def mode_index(self) -> ModeIndex:
        return ModeIndex(max_order=self.nnonlin + 1)

    @property
    def slices(self) -> Slices:
        return self.linear_analyzer.slices

    @property
    def rad(self) -> NDArray:
        return self.linear_analyzer.rad

    @cached_property
    def _invcp(self) -> NDArray:
        """Weights for integration."""
        invcp = np.ones(self.ncheb + 1)
        invcp[0] = 1 / 2
        invcp[-1] = 1 / 2
        return invcp

    @cached_property
    def _tmat(self) -> NDArray:
        """Matrix to get pseudo-spectrum."""
        ncheb = self.ncheb
        tmat = np.zeros((ncheb + 1, ncheb + 1))
        for n in range(ncheb + 1):
            for p in range(ncheb + 1):
                tmat[n, p] = (-1) ** n * np.cos(n * p * np.pi / ncheb)
        return tmat

    def _cartesian_lmat_0(self, ra_num: float) -> Matrix:
        """LHS matrix for x-independent forcing

        When the RHS is independent of x, the solution also is,
        and the velocity is uniform and only vertical, possibly null.
        Only the pressure, temperature and uniform vertical velocity
        are solved for
        """
        nnodes = self.linear_analyzer.rad.size
        dz1, dz2 = self.linear_analyzer.diff_mat(1), self.linear_analyzer.diff_mat(2)
        one = np.identity(nnodes)  # identity

        # only in that case a translating vertical velocity is possible
        solve_for_w = self.phys.phi_top is not None and self.phys.phi_bot is not None

        var_specs: list[VarSpec] = [
            Field(var="p", include_top=solve_for_w, include_bot=solve_for_w),
            Field(
                var="T",
                include_top=self.phys.heat_flux_top is not None
                or self.phys.biot_top is not None,
                include_bot=self.phys.heat_flux_bot is not None
                or self.phys.biot_bot is not None,
            ),
        ]
        if solve_for_w:
            # translation velocity
            var_specs.append(Scalar(var="w0"))

        lmat = Matrix(slices=Slices(var_specs=var_specs, nnodes=nnodes))

        # pressure equation (z momentum)
        lmat.add_term(Bulk("p"), -dz1, "p")
        lmat.add_term(Bulk("p"), ra_num * one, "T")
        # temperature equation
        lmat.add_term(Bulk("T"), dz2, "T")
        # FIXME: missing boundary conditions on T (non-dirichlet)
        # the case for a translating vertical velocity (mode 0)
        if solve_for_w:
            assert self.phys.phi_top is not None and self.phys.phi_bot is not None
            # Uniform vertical velocity in the temperature equation
            # FIXME: depends on grad T
            one_row = np.diag(one)[:, np.newaxis]
            lmat.add_term(Bulk("T"), one_row, "w0")
            # Vertical velocity in momentum boundary conditions
            lmat.add_term(Top("p"), -one, "p")
            lmat.add_term(Top("p"), self.phys.phi_top * one_row, "w0")
            lmat.add_term(Bot("p"), one, "p")
            lmat.add_term(Bot("p"), self.phys.phi_bot * one_row, "w0")
            # equation for the uniform vertical velocity
            lmat.add_term(
                Single("w0"), np.asarray(self.phys.phi_top + self.phys.phi_bot), "w0"
            )
            lmat.add_term(Single("w0"), one[:1], "p")
            lmat.add_term(Single("w0"), -one[-1:], "p")

        return lmat

    def integz(self, prof: NDArray) -> np.complexfloating:
        """Integral on the -1/2 <= z <= 1/2 interval"""
        # pseudo-spectrum
        spec = np.dot(self._tmat, prof * self._invcp)
        spec *= 2 / self.ncheb * self._invcp
        intz = (
            -1
            / 2
            * np.sum(spec[i] * 2 / (i**2 - 1) for i in range(len(spec)) if i % 2 == 0)  # type: ignore
        )
        # factor 1/2 is to account for the interval -1/2 < z < 1/2
        return intz

    def dotprod(self, ord1: int, ord2: int, harm: int) -> np.complexfloating:
        """dot product of two modes in the full solution

        serves to remove the first order solution from any mode
        of greater order
        ord1: order of the mode on the left
        ord2: order of the mode on the right
        harm: common harmonics (otherwise zero)
        ord1 and ord2 must have the same parity or the product is zero
        (no common harmonics)
        """
        # TYPE SAFETY: there is an implicit assumption on call order to other methods
        # so that ratot and full_* modes are known when calling this function.
        dprod = np.complex128(0.0)

        if ord1 % 2 == ord2 % 2:
            rac = self.ratot[0]
            # get both harmonics indices
            mode1 = self.full_sol[self.mode_index.index(ord1, harm)]
            mode2 = self.full_sol[self.mode_index.index(ord2, harm)]
            for var in ("p", "u", "w", "T"):
                prof = np.conj(mode1.extract(var)) * mode2.extract(var)
                dprod += (rac if var == "T" else 1) * self.integz(prof)
        # Complex conjugate needed to get the full dot product. CHECK!
        # no because otherwise we loose the phase, needed to remove the contribution
        # from mode 1 in other modes
        return dprod  # + np.conj(dprod)

    def ntermprod(self, mle: int, mri: int, harm: float) -> None:
        """One non-linear term on the RHS

        input : orders of the two modes to be combined, mle (left) and mri (right)
        input values for the modes themselves are taken from the predefined
        full_sol array, results are added to the nterm array.
        ordered by wavenumber
        """
        # get indices
        tall = self.slices.collocation(All("T"))

        # create local profiles
        uloc = np.zeros(self.slices.nnodes, dtype=np.complex64)
        wloc = np.zeros_like(uloc)

        # order of the produced mode
        nmo = mle + mri
        # decompose mxx = 2 lxx + yxx
        (lle, yle) = divmod(mle, 2)
        (lri, yri) = divmod(mri, 2)
        # compute the sum
        # TYPE SAFETY: there is an implicit assumption on call order to other methods
        # so that nterm and full_* modes are known when calling this function.
        for lr in range(lri + 1):
            # outer loop on the right one to avoid computing
            # radial derivative several times
            # index for the right mode in matrix
            indri = self.mode_index.index(mri, 2 * lr + yri)
            tloc = self.full_sol[indri].extract("T")
            dtr = self.linear_analyzer.diff_mat(1) @ tloc
            for ll in range(lle + 1):
                # index for the left mode in matrix
                indle = self.mode_index.index(mle, 2 * ll + yle)
                # index for nmo and ll+lr
                nharmm = 2 * (ll + lr) + yle + yri
                iind = self.mode_index.index(nmo, nharmm)

                uloc = self.full_sol[indle].extract("u")
                wloc = self.full_sol[indle].extract("w")
                ntermt = 1j * harm * (2 * lr + yri) * uloc * tloc + wloc * dtr
                # FIXME: in-place modification of ntermt
                self.nterm[iind].arr[self.slices.span(All("T"))] += ntermt[tall]

                # index for nmo and ll-lr
                nharmm = 2 * (ll - lr) + yle - yri
                iind = self.mode_index.index(nmo, abs(nharmm))
                if nharmm > 0:
                    ntermt = -1j * harm * (2 * lr + yri) * uloc * np.conj(
                        tloc
                    ) + wloc * np.conj(dtr)
                elif nharmm == 0:
                    ntermt = -1j * harm * (2 * lr + yri) * uloc * np.conj(
                        tloc
                    ) + wloc * np.conj(dtr)
                else:
                    ntermt = (
                        1j * harm * (2 * lr + yri) * np.conj(uloc) * tloc
                        + np.conj(wloc) * dtr
                    )
                # FIXME: in-place modification of ntermt
                self.nterm[iind].arr[self.slices.span(All("T"))] += ntermt[tall]

    def nonlinana(self) -> tuple[float, NDArray, list[Vector], NDArray, NDArray]:
        """Ra2 and X2"""
        nnonlin = self.nnonlin

        # First compute the linear mode and matrix
        ana = self.linear_analyzer
        ra_c, harm_c = ana.critical_ra()
        lmat_c, rmat = ana.matrices(harm_c, ra_c)
        nnodes = lmat_c.slices.total_size
        _, mode_c = ana.eigvec(harm_c, ra_c)
        mode_c = mode_c.normalize_by_max_of("w")

        # setup matrices for the non linear solution
        nmodes = self.mode_index.n_harmonics
        self.full_sol = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        self.full_w0 = np.zeros(nmodes)  # translation velocity
        # non-linear term, only the temperature part is non-null
        self.nterm = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        rhs = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        # the suite of Rayleigh numbers
        self.ratot = np.zeros(nnonlin + 1)
        self.ratot[0] = ra_c
        # coefficient for the average temperature
        meant = np.zeros(nnonlin + 1)
        meant[0] = 1 / 2
        # coefficient for the nusselt number
        qtop = np.zeros(nnonlin + 1)
        qtop[0] = 1
        # coefficients for the velocity RMS. More complex. To be done.

        # devide by 2 to get the same value as for a sin, cos representation.
        mode_c = mode_c.normalize_by(2)
        self.full_sol[0] = mode_c
        w_c = mode_c.extract("w")
        t_c = mode_c.extract("T")

        # denominator in Ra_i
        xcmxc = self.integz(np.real(w_c * t_c))

        # norm of the linear mode
        norm_x1 = self.dotprod(1, 1, 1)

        lmat = np.zeros((nnonlin + 1, nnodes, nnodes), dtype=np.complex128)
        lmat0 = self._cartesian_lmat_0(ra_c)
        lmat[0] = lmat_c.array()
        # loop on the orders
        for ii in range(2, nnonlin + 2):
            # also need the linear problem for wnk up to nnonlin*harm_c
            lmat[ii - 1] = ana.matrices(ii * harm_c, ra_c)[0].array()
            (lii, yii) = divmod(ii, 2)
            # compute the N terms
            for ll in range(1, ii):
                self.ntermprod(ll, ii - ll, harm_c)

            # compute Ra_{ii-1} if ii is odd (otherwise keep it 0).
            if yii == 1:
                # only term in harm_c from nterm contributes
                # nterm already summed by harmonics.
                ind = self.mode_index.index(ii, 1)
                prof = np.real(
                    self.full_sol[0].extract("T")
                    * np.conj(self.nterm[ind].extract("T"))
                )
                # <X_1|N(X_l, X_2n+1-l)>
                # Beware: do not forget to multiply by Rac since this is
                # the temperature part of the dot product.
                self.ratot[ii - 1] = self.ratot[0] * self.integz(prof)
                for jj in range(1, lii):
                    # only the ones in harm_c can contribute for each degree
                    ind = self.mode_index.index(2 * (lii - jj) + 1, 1)
                    # + sign because denominator takes the minus.
                    wwloc = self.full_sol[0].extract("w")
                    ttloc = self.full_sol[ind].extract("T")
                    prof = np.real(wwloc * ttloc)
                    self.ratot[ii - 1] += self.ratot[2 * jj] * self.integz(prof)
                self.ratot[ii - 1] /= xcmxc

            # add mterm to nterm to get rhs
            imin = self.mode_index.index(ii, yii)
            imax = self.mode_index.index(ii, ii)
            for irhs in range(imin, imax + 1):
                rhs[irhs] = self.nterm[irhs]
            jmax = lii if yii == 0 else lii + 1
            for jj in range(2, jmax, 2):
                # jj is index for Ra
                # index for MX is ii-jj
                (lmx, ymx) = divmod(ii - jj, 2)
                for kk in range(lmx + 1):
                    indjj = self.mode_index.index(ii, 2 * kk + ymx)
                    temp_mode = self.full_sol[indjj].extract("T")
                    # FIXME: handling of boundaries?
                    wgint = self.slices.span(Bulk("w"))
                    rhs[indjj].arr[wgint] -= self.ratot[jj] * temp_mode[1:-1]

            # note that rhs contains only the positive harmonics of the total rhs
            # which contains an additional complex conjugate. Therefore, the solution
            # should also be complemented by its complex conjugate

            # invert matrix for each harmonic number
            for jj in range(lii + 1):
                # index to fill in: same parity as ii
                harmjj = 2 * jj + yii
                ind = self.mode_index.index(ii, harmjj)
                # FIXME: full_sol is modified in place
                sol_arr = self.full_sol[ind].arr
                if harmjj == 0:  # special treatment for 0 modes.
                    # should be possible to avoid the use of a rhs0
                    rhs0 = np.zeros(lmat0.slices.total_size, dtype=np.complex128)
                    # FIXME: handling of boundaries?  These don't necessarily match.
                    rhs0[lmat0.slices.span(All("T"))] = rhs[ind].arr[
                        self.slices.span(All("T"))
                    ]

                    sol = Vector(slices=lmat0.slices, arr=solve(lmat0.array(), rhs0))
                    # FIXME: how to handle boundaries?  Potentially,
                    # sol has boundary points that full_sol hasn't
                    sol_arr[self.slices.span(Bulk("p"))] = sol.extract("p")[1:-1]
                    sol_arr[self.slices.span(Bulk("T"))] = sol.extract("T")[1:-1]
                    # compute coefficient ii in meant
                    # factor to account for the complex conjugate
                    prot = 2 * np.real(sol.extract("T"))
                    meant[ii] = self.integz(prot)
                    dprot = ana.diff_mat(1) @ prot
                    qtop[ii] = -dprot[0]
                    if self.phys.phi_top is not None and self.phys.phi_bot is not None:
                        # translation velocity possible
                        self.full_w0[ind] = np.real(sol.extract("w0").item())
                    else:
                        self.full_w0[ind] = 0
                else:
                    # Only the positive power of exp(1j k x)
                    # eigva = linalg.eigvals(lmat[harmjj - 1], rmat)
                    # print('harm, eig lmat0 =', harmjj, np.sort(np.real(eigva)))
                    if harmjj == 1:
                        # matrix is singular. Solve using least square
                        sol_arr[:] = lstsq(
                            lmat[harmjj - 1],
                            rhs[ind].arr,
                            rcond=None,
                        )[0]
                        # remove the contribution proportional to X1, if it exists
                        for jj in range(2):
                            dp1 = self.dotprod(1, ii, 1)
                            sol_arr[:] -= dp1 / norm_x1 * self.full_sol[0].arr
                    else:
                        sol_arr[:] = solve(lmat[harmjj - 1], rhs[ind].arr)

        return harm_c, self.ratot, self.full_sol, meant, qtop
