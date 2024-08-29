from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property
from itertools import repeat

import numpy as np
from numpy.linalg import lstsq, solve

from .cartesian import CartStability
from .matrix import All, Bot, Bulk, Field, Matrix, Scalar, Single, Slices, Top, Vector

if typing.TYPE_CHECKING:
    from typing import Mapping, Sequence

    from numpy.typing import NDArray

    from .matrix import VarSpec


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


@dataclass(frozen=True)
class IntegralCheb:
    """Integral on the -1/2 <= z <= 1/2 interval of profile at Chebyshev nodes."""

    max_degree: int

    @property
    def nnodes(self) -> int:
        return self.max_degree + 1

    @cached_property
    def _invcp(self) -> NDArray:
        """Weights for integration."""
        invcp = np.ones(self.nnodes)
        invcp[0] = 1 / 2
        invcp[-1] = 1 / 2
        return invcp

    @cached_property
    def _tmat(self) -> NDArray:
        """Matrix to get pseudo-spectrum."""
        n, p = np.meshgrid(range(self.nnodes), range(self.nnodes), indexing="ij")
        return (-1) ** n * np.cos(n * p * np.pi / self.max_degree)

    def apply(self, prof: NDArray) -> np.complexfloating:
        assert prof.shape == (self.nnodes,)
        # pseudo-spectrum
        spec = self._tmat @ (prof * self._invcp)
        spec *= 2 / self.max_degree * self._invcp
        indices = np.arange(0, self.nnodes, 2)
        # factor 1/2 is to account for the interval -1/2 < z < 1/2
        return -1 / 2 * np.sum(spec[::2] * 2 / (indices**2 - 1))


@dataclass(frozen=True)
class NonLinearAnalyzer:
    """Weakly non-linear analysis.

    Attributes:
        phys: physical problem.
        ncheb: degree of Chebyshev polynomials.
        nnonlin: maximum order of non-linear analysis
    """

    linear_pblm: CartStability
    nnonlin: int

    @cached_property
    def mode_index(self) -> ModeIndex:
        return ModeIndex(max_order=self.nnonlin + 1)

    @cached_property
    def z_integ(self) -> IntegralCheb:
        return IntegralCheb(max_degree=self.linear_pblm.chebyshev_degree)

    @property
    def slices(self) -> Slices:
        return self.linear_pblm.slices

    @property
    def nodes(self) -> NDArray:
        return self.linear_pblm.nodes

    def _cartesian_lmat_0(self, ra_num: float) -> Matrix:
        """LHS matrix for x-independent forcing

        When the RHS is independent of x, the solution also is,
        and the velocity is uniform and only vertical, possibly null.
        Only the pressure, temperature and uniform vertical velocity
        are solved for
        """
        nnodes = self.linear_pblm.nodes.size
        ops = self.linear_pblm.operators(0.0)

        # only in that case a translating vertical velocity is possible
        solve_for_w = (
            self.linear_pblm.bc_mom_top.flow_through
            and self.linear_pblm.bc_mom_bot.flow_through
        )

        var_specs: list[VarSpec] = [Field(var="p"), Field(var="T")]
        if solve_for_w:
            # translation velocity
            var_specs.append(Scalar(var="w0"))

        lmat = Matrix(slices=Slices(var_specs=var_specs, nnodes=nnodes))

        # pressure equation (z momentum)
        lmat.add_term(Bulk("p"), -ops.grad_r, "p")
        lmat.add_term(Bulk("p"), ra_num * ops.identity, "T")
        # temperature equation
        self.linear_pblm.temperature.bc_top.add_top("T", lmat, ops)
        self.linear_pblm.temperature.bc_bot.add_bot("T", lmat, ops)
        lmat.add_term(Bulk("T"), ops.lapl_r, "T")
        # the case for a translating vertical velocity (mode 0)
        if solve_for_w:
            # FIXME: more general handling of boundary condition
            phi_top = self.linear_pblm.bc_mom_top.phase_number  # type: ignore
            phi_bot = self.linear_pblm.bc_mom_bot.phase_number  # type: ignore
            # Uniform vertical velocity in the temperature equation
            tref = self.linear_pblm.temperature.ref_prof.eval_with(ops)
            grad_tref = (ops.grad_r @ tref)[:, np.newaxis]
            lmat.add_term(Bulk("T"), -grad_tref, "w0")
            # Vertical velocity in momentum boundary conditions
            one_row = np.diag(ops.identity)[:, np.newaxis]
            lmat.add_term(Top("p"), -ops.identity, "p")
            lmat.add_term(Top("p"), phi_top * one_row, "w0")
            lmat.add_term(Bot("p"), ops.identity, "p")
            lmat.add_term(Bot("p"), phi_bot * one_row, "w0")
            # equation for the uniform vertical velocity
            lmat.add_term(Single("w0"), np.asarray(phi_top + phi_bot), "w0")
            lmat.add_term(Single("w0"), ops.identity[:1], "p")
            lmat.add_term(Single("w0"), -ops.identity[-1:], "p")
        else:
            lmat.add_term(Top("p"), ops.identity, "p")
            lmat.add_term(Bot("p"), ops.identity, "p")

        return lmat

    def _dotprod(
        self, rac: float, full_sol: list[Vector], ord1: int, ord2: int, harm: int
    ) -> np.complexfloating:
        """dot product of two modes in the full solution

        serves to remove the first order solution from any mode
        of greater order
        ord1: order of the mode on the left
        ord2: order of the mode on the right
        harm: common harmonics (otherwise zero)
        ord1 and ord2 must have the same parity or the product is zero
        (no common harmonics)
        """
        dprod = np.complex128(0.0)

        if ord1 % 2 == ord2 % 2:
            # get both harmonics indices
            mode1 = full_sol[self.mode_index.index(ord1, harm)]
            mode2 = full_sol[self.mode_index.index(ord2, harm)]
            for var in ("p", "u", "w", "T"):
                prof = np.conj(mode1.extract(var)) * mode2.extract(var)
                dprod += (rac if var == "T" else 1) * self.z_integ.apply(prof)
        # Complex conjugate needed to get the full dot product. CHECK!
        # no because otherwise we loose the phase, needed to remove the contribution
        # from mode 1 in other modes
        return dprod  # + np.conj(dprod)

    def _ntermprod(
        self,
        full_sol: list[Vector],
        nterm: list[Vector],
        mle: int,
        mri: int,
        harm: float,
    ) -> None:
        """One non-linear term on the RHS

        input : orders of the two modes to be combined, mle (left) and mri (right)
        input values for the modes themselves are taken from the predefined
        full_sol array, results are added to the nterm array.
        ordered by wavenumber
        """
        # get indices
        tbulk = self.slices.collocation(Bulk("T"))

        # create local profiles
        uloc = np.zeros(self.slices.nnodes, dtype=np.complex64)
        wloc = np.zeros_like(uloc)

        grad_r = self.linear_pblm.operators(0.0).grad_r

        # order of the produced mode
        nmo = mle + mri
        # decompose mxx = 2 lxx + yxx
        (lle, yle) = divmod(mle, 2)
        (lri, yri) = divmod(mri, 2)
        # compute the sum
        for lr in range(lri + 1):
            # outer loop on the right one to avoid computing
            # radial derivative several times
            # index for the right mode in matrix
            indri = self.mode_index.index(mri, 2 * lr + yri)
            tloc = full_sol[indri].extract("T")
            dtr = grad_r @ tloc
            for ll in range(lle + 1):
                factor = 1j * harm * (2 * lr + yri)
                # index for the left mode in matrix
                indle = self.mode_index.index(mle, 2 * ll + yle)
                # index for nmo and ll+lr
                nharmm = 2 * (ll + lr) + yle + yri
                iind = self.mode_index.index(nmo, nharmm)

                uloc = full_sol[indle].extract("u")
                wloc = full_sol[indle].extract("w")
                ntermt = factor * uloc * tloc + wloc * dtr
                # FIXME: in-place modification of ntermt
                nterm[iind].arr[self.slices.span(Bulk("T"))] += ntermt[tbulk]

                # index for nmo and ll-lr
                nharmm = 2 * (ll - lr) + yle - yri
                iind = self.mode_index.index(nmo, abs(nharmm))
                if nharmm >= 0:
                    ntermt = -factor * uloc * np.conj(tloc) + wloc * np.conj(dtr)
                else:
                    ntermt = factor * np.conj(uloc) * tloc + np.conj(wloc) * dtr
                # FIXME: in-place modification of ntermt
                nterm[iind].arr[self.slices.span(Bulk("T"))] += ntermt[tbulk]

    @cached_property
    def _nonlinana(self) -> tuple[float, NDArray, list[Vector], NDArray, NDArray]:
        """Ra2 and X2"""
        nnonlin = self.nnonlin

        # First compute the linear mode and matrix
        ana = self.linear_pblm
        ra_c, harm_c = ana.critical_ra()
        eig_pblm_c = ana.eigen_problem(harm_c, ra_c)
        lmat_c = eig_pblm_c.lmat
        nnodes = lmat_c.slices.total_size
        _, mode_c = eig_pblm_c.max_eigvec()
        mode_c = mode_c.normalize_by_max_of("w")

        grad_r = ana.operators(0.0).grad_r

        # setup matrices for the non linear solution
        nmodes = self.mode_index.n_harmonics
        full_sol = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        full_w0 = np.zeros(nmodes)  # translation velocity
        # non-linear term, only the temperature part is non-null
        nterm = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        rhs = [
            Vector(slices=self.slices, arr=np.zeros(nnodes, dtype=np.complex128))
            for _ in range(nmodes)
        ]
        # the suite of Rayleigh numbers
        ratot = np.zeros(nnonlin + 1)
        ratot[0] = ra_c
        # coefficient for the average temperature
        meant = np.zeros(nnonlin + 1)
        meant[0] = 1 / 2
        # coefficient for the nusselt number
        qtop = np.zeros(nnonlin + 1)
        qtop[0] = 1
        # coefficients for the velocity RMS. More complex. To be done.

        # devide by 2 to get the same value as for a sin, cos representation.
        mode_c = mode_c.normalize_by(2)
        full_sol[0] = mode_c
        w_c = mode_c.extract("w")
        t_c = mode_c.extract("T")

        # denominator in Ra_i
        xcmxc = self.z_integ.apply(np.real(w_c * t_c))

        # norm of the linear mode
        norm_x1 = self._dotprod(ratot[0], full_sol, 1, 1, 1)

        lmat = np.zeros((nnonlin + 1, nnodes, nnodes), dtype=np.complex128)
        lmat0 = self._cartesian_lmat_0(ra_c)
        lmat[0] = lmat_c.array()
        # loop on the orders
        for ii in range(2, nnonlin + 2):
            # also need the linear problem for wnk up to nnonlin*harm_c
            lmat[ii - 1] = ana.eigen_problem(ii * harm_c, ra_c).lmat.array()
            (lii, yii) = divmod(ii, 2)
            # compute the N terms
            for ll in range(1, ii):
                self._ntermprod(full_sol, nterm, ll, ii - ll, harm_c)

            # compute Ra_{ii-1} if ii is odd (otherwise keep it 0).
            if yii == 1:
                # only term in harm_c from nterm contributes
                # nterm already summed by harmonics.
                ind = self.mode_index.index(ii, 1)
                prof = np.real(
                    full_sol[0].extract("T") * np.conj(nterm[ind].extract("T"))
                )
                # <X_1|N(X_l, X_2n+1-l)>
                # Beware: do not forget to multiply by Rac since this is
                # the temperature part of the dot product.
                ratot[ii - 1] = ratot[0] * self.z_integ.apply(prof)
                for jj in range(1, lii):
                    # only the ones in harm_c can contribute for each degree
                    ind = self.mode_index.index(2 * (lii - jj) + 1, 1)
                    # + sign because denominator takes the minus.
                    wwloc = full_sol[0].extract("w")
                    ttloc = full_sol[ind].extract("T")
                    prof = np.real(wwloc * ttloc)
                    ratot[ii - 1] += ratot[2 * jj] * self.z_integ.apply(prof)
                ratot[ii - 1] /= xcmxc

            # add mterm to nterm to get rhs
            imin = self.mode_index.index(ii, yii)
            imax = self.mode_index.index(ii, ii)
            for irhs in range(imin, imax + 1):
                rhs[irhs] = nterm[irhs]
            jmax = lii if yii == 0 else lii + 1
            for jj in range(2, jmax, 2):
                # jj is index for Ra
                # index for MX is ii-jj
                (lmx, ymx) = divmod(ii - jj, 2)
                for kk in range(lmx + 1):
                    indjj = self.mode_index.index(ii, 2 * kk + ymx)
                    temp_mode = full_sol[indjj].extract("T")
                    # FIXME: handling of boundaries?
                    wgint = self.slices.span(Bulk("w"))
                    rhs[indjj].arr[wgint] -= ratot[jj] * temp_mode[1:-1]

            # note that rhs contains only the positive harmonics of the total rhs
            # which contains an additional complex conjugate. Therefore, the solution
            # should also be complemented by its complex conjugate

            # invert matrix for each harmonic number
            for jj in range(lii + 1):
                # index to fill in: same parity as ii
                harmjj = 2 * jj + yii
                ind = self.mode_index.index(ii, harmjj)
                # FIXME: full_sol is modified in place
                sol_arr = full_sol[ind].arr
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
                    meant[ii] = self.z_integ.apply(prot)
                    dprot = grad_r @ prot
                    qtop[ii] = -dprot[0]
                    if (
                        self.linear_pblm.bc_mom_top.flow_through
                        and self.linear_pblm.bc_mom_bot.flow_through
                    ):
                        # translation velocity possible
                        full_w0[ind] = np.real(sol.extract("w0").item())
                    else:
                        full_w0[ind] = 0
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
                            dp1 = self._dotprod(ratot[0], full_sol, 1, ii, 1)
                            sol_arr[:] -= dp1 / norm_x1 * full_sol[0].arr
                    else:
                        sol_arr[:] = solve(lmat[harmjj - 1], rhs[ind].arr)

        return harm_c, ratot, full_sol, meant, qtop

    @property
    def harm_c(self) -> float:
        """Critical linear mode."""
        return self._nonlinana[0]

    @property
    def ratot(self) -> NDArray:
        """Coefficients of development in Rayleigh."""
        return self._nonlinana[1]

    @property
    def all_modes(self) -> list[Vector]:
        """Harmonic modes."""
        return self._nonlinana[2]

    @property
    def meant(self) -> NDArray:
        """Coefficients for the mean temperature."""
        return self._nonlinana[3]

    @property
    def qtop(self) -> NDArray:
        """Coefficients for the heat flux at the top."""
        return self._nonlinana[4]
