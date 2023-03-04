from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.linalg import lstsq, solve

from .analyzer import LinearAnalyzer
from .matrix import All, Bot, Bulk, Field, Matrix, Scalar, Single, Slices, Top, Vector

if typing.TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from .matrix import VarSpec
    from .physics import PhysicalProblem


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

    @property
    def slices(self) -> Slices:
        return self.linear_analyzer.slices

    @property
    def rad(self) -> NDArray:
        return self.linear_analyzer.rad

    def _insert_boundaries(self, mode: NDArray, im0: int, imn: int) -> NDArray:
        """Insert zero at boundaries of mode if needed

        This need to be done when Dirichlet BCs are applied
        """
        if im0 == 1:
            mode = np.insert(mode, [0], [0])
        if imn == self.ncheb - 1:
            mode = np.append(mode, 0)
        return mode

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

    @cached_property
    def _nmodes_tot(self) -> int:
        order_max = self.nnonlin + 1
        return sum(n // 2 + 1 for n in range(1, order_max + 1))

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

    def integz(self, prof: NDArray) -> NDArray:
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

    def indexmat(
        self, order: int, ind: int = 1, harmm: Optional[int] = None
    ) -> tuple[None, int, int, int]:
        """Indices of the matrix of modes for non-linear analysis

        Returns
        ordn, harm: order and harmonic number corresponding to ind in matrix.
        ind: matrix index corresponding to order and harmm
        """
        # if ordnn > order:
        # raise ValueError("in indexmat, ordnn > order")
        ordn = 0
        harm = 0
        harms = np.array([], dtype=np.int64)
        ordns = np.array([], dtype=np.int64)
        index = 0
        for n in range(1, order + 1):
            if n % 2 == 0:
                indices = np.array([i for i in range(0, n + 1, 2)])
                harms = np.concatenate((harms, indices))
                ordns = np.concatenate(
                    (ordns, n * np.ones(indices.shape, dtype=np.int64))
                )
            else:
                indices = np.array([i for i in range(1, n + 1, 2)])
                harms = np.concatenate((harms, indices))
                ordns = np.concatenate(
                    (ordns, n * np.ones(indices.shape, dtype=np.int64))
                )
            if ordn == 0 and ordns.shape[0] >= ind + 1:
                ordn = ordns[ind]
                harm = harms[ind]
            if harmm is not None:
                if n == order:
                    index += np.where(np.array(indices) == harmm)[0][0]
                else:
                    index += len(indices)
        return None, ordn, harm, index

    def dotprod(self, ord1: int, ord2: int, harm: int) -> NDArray:
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
        rac = self.ratot[0]  # type: ignore

        # get indices
        pall = self.slices.collocation(All("p"))
        uall = self.slices.collocation(All("u"))
        wall = self.slices.collocation(All("w"))
        tall = self.slices.collocation(All("T"))

        if (ord1 % 2 == 0 and ord2 % 2 == 0) or (ord1 % 2 == 1 and ord2 % 2 == 1):
            # create local profiles
            prof = np.zeros(self.slices.nnodes, dtype=np.complex64)
            # get indices in the global matrix
            ind1 = self.indexmat(ord1, harmm=harm)[3]
            ind2 = self.indexmat(ord2, harmm=harm)[3]
            # pressure part
            prof[pall] = np.conj(self.full_p[ind1]) * self.full_p[ind2]  # type: ignore
            dprod = self.integz(prof)
            # horizontal velocity part
            prof.fill(0)
            prof[uall] = np.conj(self.full_u[ind1]) * self.full_u[ind2]  # type: ignore
            dprod += self.integz(prof)
            # vertical velocity part
            prof.fill(0)
            prof[wall] = np.conj(self.full_w[ind1]) * self.full_w[ind2]  # type: ignore
            dprod += self.integz(prof)
            # temperature part
            prof.fill(0)
            prof[tall] = np.conj(self.full_t[ind1]) * self.full_t[ind2]  # type: ignore
            dprod += rac * self.integz(prof)
        else:
            dprod = np.asarray(0)
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
        uall = self.slices.collocation(All("u"))
        wall = self.slices.collocation(All("w"))
        tall = self.slices.collocation(All("T"))

        # create local profiles
        uloc = np.zeros(self.slices.nnodes, dtype=np.complex64)
        wloc = np.zeros_like(uloc)
        tloc = np.zeros_like(uloc)

        # order of the produced mode
        nmo = mle + mri
        # decompose mxx = 2 lxx + yxx
        (lle, yle) = divmod(mle, 2)
        (lri, yri) = divmod(mri, 2)
        # compute the sum
        # TYPE SAFETY: there is an implicit assumption on call order to other methods
        # so that ntermt and full_* modes are known when calling this function.
        for lr in range(lri + 1):
            # outer loop on the right one to avoid computing
            # radial derivative several times
            # index for the right mode in matrix
            indri = self.indexmat(mri, harmm=2 * lr + yri)[3]
            tloc[tall] = self.full_t[indri]  # type: ignore
            dtr = self.linear_analyzer.diff_mat(1) @ tloc
            for ll in range(lle + 1):
                # index for the left mode in matrix
                indle = self.indexmat(mle, harmm=2 * ll + yle)[3]
                # index for nmo and ll+lr
                nharmm = 2 * (ll + lr) + yle + yri
                iind = self.indexmat(nmo, harmm=nharmm)[3]
                # reduce to shape of t
                uloc[uall] = self.full_u[indle]  # type: ignore
                wloc[wall] = self.full_w[indle]  # type: ignore
                self.ntermt[iind] += (  # type: ignore
                    1j * harm * (2 * lr + yri) * uloc[tall] * tloc[tall]
                    + wloc[tall] * dtr[tall]
                )
                # index for nmo and ll-lr
                nharmm = 2 * (ll - lr) + yle - yri
                iind = self.indexmat(nmo, harmm=np.abs(nharmm))[3]
                if nharmm > 0:
                    self.ntermt[iind] += -1j * harm * (2 * lr + yri) * uloc[  # type: ignore
                        tall
                    ] * np.conj(
                        tloc[tall]
                    ) + wloc[
                        tall
                    ] * np.conj(
                        dtr[tall]
                    )
                elif nharmm == 0:
                    self.ntermt[iind] += -1j * harm * (2 * lr + yri) * uloc[  # type: ignore
                        tall
                    ] * np.conj(
                        tloc[tall]
                    ) + wloc[
                        tall
                    ] * np.conj(
                        dtr[tall]
                    )
                else:
                    self.ntermt[iind] += (  # type: ignore
                        1j * harm * (2 * lr + yri) * np.conj(uloc[tall]) * tloc[tall]
                        + np.conj(wloc[tall]) * dtr[tall]
                    )

    def nonlinana(self) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
        """Ra2 and X2"""
        nnonlin = self.nnonlin
        # global indices and slices
        # FIXME: use Vector and Matrix to avoid manipulating those by hand
        pgall = self.slices.span(All("p"))
        ugall = self.slices.span(All("u"))
        wgall = self.slices.span(All("w"))
        tgall = self.slices.span(All("T"))
        pgint = self.slices.span(Bulk("p"))
        wgint = self.slices.span(Bulk("w"))
        tgint = self.slices.span(Bulk("T"))
        iw0 = self.slices.collocation(All("w")).start
        iwn = self.slices.collocation(All("w")).stop - 1
        it0 = self.slices.collocation(All("T")).start
        itn = self.slices.collocation(All("T")).stop - 1

        # First compute the linear mode and matrix
        ana = self.linear_analyzer
        ra_c, harm_c = ana.critical_ra()
        lmat_c, rmat = ana.matrices(harm_c, ra_c)
        nnodes = lmat_c.slices.total_size
        _, mode_c = ana.eigvec(harm_c, ra_c)
        mode_c = mode_c.normalize_by_max_of("w")

        # setup matrices for the non linear solution
        self.full_sol = np.zeros((self._nmodes_tot, nnodes), dtype=np.complex128)
        self.full_p = self.full_sol[:, pgall]
        self.full_u = self.full_sol[:, ugall]
        self.full_w = self.full_sol[:, wgall]
        self.full_t = self.full_sol[:, tgall]
        self.full_w0 = np.zeros(self._nmodes_tot)  # translation velocity
        self.nterm = np.zeros_like(self.full_sol)
        self.rhs = np.zeros_like(self.full_sol)

        # temperature part, the only non-null one in the non-linear term
        self.ntermt = self.nterm[:, tgall]
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
        self.full_sol[0] = mode_c.arr
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
                ind = self.indexmat(ii, harmm=1)[3]
                prof = self._insert_boundaries(
                    np.real(self.full_t[0] * np.conj(self.ntermt[ind])), it0, itn
                )
                # <X_1|N(X_l, X_2n+1-l)>
                # Beware: do not forget to multiply by Rac since this is
                # the temperature part of the dot product.
                self.ratot[ii - 1] = self.ratot[0] * self.integz(prof)
                for jj in range(1, lii):
                    # only the ones in harm_c can contribute for each degree
                    ind = self.indexmat(2 * (lii - jj) + 1, harmm=1)[3]
                    # + sign because denominator takes the minus.
                    wwloc = self._insert_boundaries(self.full_w[0], iw0, iwn)
                    ttloc = self._insert_boundaries(self.full_t[ind], it0, itn)
                    prof = np.real(wwloc * ttloc)
                    self.ratot[ii - 1] += self.ratot[2 * jj] * self.integz(prof)
                self.ratot[ii - 1] /= xcmxc

            # add mterm to nterm to get rhs
            imin = self.indexmat(ii, harmm=yii)[3]
            imax = self.indexmat(ii, harmm=ii)[3]
            self.rhs[imin : imax + 1] = self.nterm[imin : imax + 1]
            jmax = lii if yii == 0 else lii + 1
            for jj in range(2, jmax, 2):
                # jj is index for Ra
                # index for MX is ii-jj
                (lmx, ymx) = divmod(ii - jj, 2)
                for kk in range(lmx + 1):
                    indjj = self.indexmat(ii, harmm=2 * kk + ymx)[3]
                    self.rhs[indjj, wgint] -= self.ratot[jj] * self.full_t[indjj]

            # note that rhs contains only the positive harmonics of the total rhs
            # which contains an additional complex conjugate. Therefore, the solution
            # should also be complemented by its complex conjugate

            # invert matrix for each harmonic number
            for jj in range(lii + 1):
                # index to fill in: same parity as ii
                harmjj = 2 * jj + yii
                ind = self.indexmat(ii, harmm=harmjj)[3]
                if harmjj == 0:  # special treatment for 0 modes.
                    # should be possible to avoid the use of a rhs0
                    rhs0 = np.zeros(lmat0.slices.total_size, dtype=np.complex128)
                    rhs0[lmat0.slices.span(All("T"))] = self.rhs[ind, tgall]

                    sol = Vector(slices=lmat0.slices, arr=solve(lmat0.array(), rhs0))
                    self.full_sol[ind, pgint] = sol.extract("p")[1:-1]
                    self.full_sol[ind, tgint] = sol.extract("T")[1:-1]
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
                        self.full_sol[ind] = lstsq(
                            lmat[harmjj - 1], self.rhs[ind], rcond=None
                        )[0]
                        # remove the contribution proportional to X1, if it exists
                        for jj in range(2):
                            dp1 = self.dotprod(1, ii, 1)
                            self.full_sol[ind] -= dp1 / norm_x1 * self.full_sol[0]
                    else:
                        self.full_sol[ind] = solve(lmat[harmjj - 1], self.rhs[ind])

        return harm_c, self.ratot, self.full_sol, meant, qtop
