from __future__ import annotations

import typing

import numpy as np
from numpy.linalg import lstsq, solve

from .analyzer import Analyser, LinearAnalyzer
from .misc import build_slices, normalize_modes

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence

    from numpy.typing import NDArray


def cartesian_matrices_0(
    self: NonLinearAnalyzer, ra_num: float
) -> tuple[NDArray, slice, slice, slice, slice, int]:
    """LHS matrix for x-independent forcing

    When the RHS is independent of x, the solution also is,
    and the velocity is uniform and only vertical, possibly null.
    Only the pressure, temperature and uniform vertical velocity
    are solved for
    """
    ncheb = self._ncheb
    dz1, dz2 = self.dr1, self.dr2
    one = np.identity(ncheb + 1)  # identity
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    heat_flux_top = self.phys.heat_flux_top
    biot_top = self.phys.biot_top
    biot_bot = self.phys.biot_bot
    heat_flux_bot = self.phys.heat_flux_bot

    # pressure
    if phi_top is not None and phi_bot is not None:
        # only in that case a translating vertical velocity is possible
        ip0 = 0
        ipn = ncheb
    else:
        ip0 = 1
        ipn = ncheb - 1
    # temperature
    it0 = 0 if (heat_flux_top is not None or biot_top is not None) else 1
    itn = ncheb if (heat_flux_bot is not None or biot_bot is not None) else ncheb - 1
    # global indices and slices in the solution containing only P and T
    _, igf, slall, slint, slgall, slgint = build_slices([(ip0, ipn), (it0, itn)], ncheb)
    ipg, itg = igf
    pall, tall = slall
    pint, tint = slint
    pgall, tgall = slgall
    pgint, tgint = slgint

    # index for vertical velocity
    if phi_top is not None and phi_bot is not None:
        igw = itg(itn + 1)
    else:
        igw = itg(itn)

    # global indices and slices in the total solution

    # initialize matrix
    lmat = np.zeros((igw + 1, igw + 1))
    # pressure equation (z momentum)
    lmat[pgint, pgall] = -dz1[pint, pall]
    lmat[pgint, tgint] = ra_num * one[pint, tint]
    # temperature equation
    lmat[tgint, tgall] = dz2[tint, tall]
    # the case for a translating vertical velocity (mode 0)
    if phi_top is not None and phi_bot is not None:
        # Uniform vertical velocity in the temperature equation
        lmat[tgint, igw] = 1
        # Vertical velocity in momentum boundary conditions
        lmat[0, 0] = -1
        lmat[0, igw] = phi_top
        lmat[ipn, ipn] = 1
        lmat[ipn, igw] = phi_bot
        # equation for the uniform vertical velocity
        lmat[igw, igw] = phi_top + phi_bot
        lmat[igw, 0] = 1
        lmat[igw, ipn] = -1

    return lmat, pgint, tgint, pgall, tgall, igw


class NonLinearAnalyzer(Analyser):

    """Perform non-linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def integz(self, prof: NDArray) -> NDArray:
        """Integral on the -1/2 <= z <= 1/2 interval"""
        ncheb = self._ncheb
        invcp = self._invcp
        tmat = self._tmat

        # pseudo-spectrum
        spec = np.dot(tmat, prof * invcp)
        spec *= 2 / ncheb * invcp
        intz = (
            -1
            / 2
            * np.sum(spec[i] * 2 / (i**2 - 1) for i in range(len(spec)) if i % 2 == 0)  # type: ignore
        )
        # factor 1/2 is to account for the interval -1/2 < z < 1/2
        return intz

    def indexmat(
        self, order: int, ind: int = 1, harmm: Optional[int] = None
    ) -> tuple[int, int, int, int]:
        """Indices of the matrix of modes for non-linear analysis

        Returns
        nmax: the max number of matrix element to solve up to order.
        ordn, harm: order and harmonic number corresponding to ind in matrix.
        ind: matrix index corresponding to order and harmm
        """
        # if ordnn > order:
        # raise ValueError("in indexmat, ordnn > order")
        nmax = 0
        ordn = 0
        harm = 0
        harms = np.array([], dtype=np.int64)
        ordns = np.array([], dtype=np.int64)
        index = 0
        for n in range(1, order + 1):
            if n % 2 == 0:
                jj = int(n / 2)
                indices = np.array([i for i in range(0, n + 1, 2)])
                harms = np.concatenate((harms, indices))
                ordns = np.concatenate(
                    (ordns, n * np.ones(indices.shape, dtype=np.int64))
                )
            else:
                jj = int((n - 1) / 2)
                indices = np.array([i for i in range(1, n + 1, 2)])
                harms = np.concatenate((harms, indices))
                ordns = np.concatenate(
                    (ordns, n * np.ones(indices.shape, dtype=np.int64))
                )
            nmax += jj + 1
            if ordn == 0 and ordns.shape[0] >= ind + 1:
                ordn = ordns[ind]
                harm = harms[ind]
            if harmm is not None:
                if n == order:
                    index += np.where(np.array(indices) == harmm)[0][0]
                else:
                    index += len(indices)
        return nmax, ordn, harm, index

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
        ncheb = self._ncheb

        # get indices
        slall = self._slices()[2]
        pall, uall, wall, tall = slall

        if (ord1 % 2 == 0 and ord2 % 2 == 0) or (ord1 % 2 == 1 and ord2 % 2 == 1):
            # create local profiles
            prof = np.zeros(ncheb + 1) * (1 + 0j)
            # get indices in the global matrix
            ind1 = self.indexmat(ord1, harmm=harm)[3]
            ind2 = self.indexmat(ord2, harmm=harm)[3]
            # pressure part
            prof[pall] = np.conj(self.full_p[ind1]) * self.full_p[ind2]  # type: ignore
            dprod = self.integz(prof)
            # horizontal velocity part
            prof = np.zeros(ncheb + 1) * (1 + 0j)
            prof[uall] = np.conj(self.full_u[ind1]) * self.full_u[ind2]  # type: ignore
            dprod += self.integz(prof)
            # vertical velocity part
            prof = np.zeros(ncheb + 1) * (1 + 0j)
            prof[wall] = np.conj(self.full_w[ind1]) * self.full_w[ind2]  # type: ignore
            dprod += self.integz(prof)
            # temperature part
            prof = np.zeros(ncheb + 1) * (1 + 0j)
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
        ncheb = self._ncheb
        # global indices and slices
        slall = self._slices()[2]
        pall, uall, wall, tall = slall

        # create local profiles
        uloc = np.zeros(ncheb + 1) * (1 + 0j)
        wloc = np.zeros(ncheb + 1) * (1 + 0j)
        tloc = np.zeros(ncheb + 1) * (1 + 0j)

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
            dtr = np.dot(self.dr1, tloc)
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

    def symmetrize(self, ind: int) -> None:
        """Make the solution symmetric with respect to z -> -z

        ind: index of the mode in the full solution
        """
        # TYPE SAFETY: there is an implicit assumption on call order to other methods
        # so that full_* modes are known when calling this function.
        self.full_p[ind] = 0.5 * (self.full_p[ind] + np.flipud(self.full_p[ind]))  # type: ignore
        self.full_u[ind] = 0.5 * (self.full_u[ind] - np.flipud(self.full_u[ind]))  # type: ignore
        self.full_w[ind] = 0.5 * (self.full_w[ind] + np.flipud(self.full_w[ind]))  # type: ignore
        self.full_t[ind] = 0.5 * (self.full_t[ind] + np.flipud(self.full_t[ind]))  # type: ignore

    def nonlinana(self) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
        """Ra2 and X2"""
        nnonlin = self._nnonlin
        # global indices and slices
        i0n, igf, slall, slint, slgall, slgint = self._slices()
        i_0s, i_ns = zip(*i0n)
        iu0, iw0, it0 = i_0s[1:]
        iun, iwn, itn = i_ns[1:]
        pall, uall, wall, tall = slall
        pgall, ugall, wgall, tgall = slgall
        pgint, ugint, wgint, tgint = slgint

        # First compute the linear mode and matrix
        ana = LinearAnalyzer(self.phys, self._ncheb)
        ra_c, harm_c = ana.critical_ra()
        lmat_c, rmat = self.matrices(harm_c, ra_c)
        _, mode_c = self.eigvec(harm_c, ra_c)
        modec: Sequence[NDArray] = self.split_mode(mode_c, harm_c, apply_bc=True)
        modec, _ = normalize_modes(modec, norm_mode=2, full_norm=False)

        # setup matrices for the non linear solution
        nmodez = np.shape(mode_c)
        nkmax = self.indexmat(nnonlin + 1)[0]
        self.full_sol = np.zeros((nkmax, nmodez[0])) * (1 + 1j)
        self.full_p = self.full_sol[:, pgall]
        self.full_u = self.full_sol[:, ugall]
        self.full_w = self.full_sol[:, wgall]
        self.full_t = self.full_sol[:, tgall]
        self.full_w0 = np.zeros(nkmax)  # translation velocity
        self.nterm = np.zeros(self.full_sol.shape) * (1 + 1j)
        self.rhs = np.zeros(self.full_sol.shape) * (1 + 1j)

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

        (p_c, u_c, w_c, t_c) = modec
        # devide by 2 to get the same value as for a sin, cos representation.
        p_c /= 2
        u_c /= 2
        w_c /= 2
        t_c /= 2

        self.full_sol[0] = self._join_mode_cartesian((p_c, u_c, w_c, t_c))
        # denominator in Ra_i
        xcmxc = self.integz(np.real(w_c * t_c))

        # norm of the linear mode
        norm_x1 = self.dotprod(1, 1, 1)

        lmat = np.zeros((nnonlin + 1, lmat_c.shape[0], lmat_c.shape[1])) * (1 + 1j)
        lmat0, pgint0, tgint0, pgall0, tgall0, igw0 = cartesian_matrices_0(self, ra_c)
        lmat[0] = lmat_c
        # loop on the orders
        for ii in range(2, nnonlin + 2):
            # also need the linear problem for wnk up to nnonlin*harm_c
            lmat[ii - 1] = self.matrices(ii * harm_c, ra_c)[0]
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
                    rhs0 = np.zeros(lmat0.shape[1], dtype=complex)
                    rhs0[tgall0] = self.rhs[ind, tgall]

                    sol = solve(lmat0, rhs0)
                    self.full_sol[ind, pgint] = sol[pgint0]
                    self.full_sol[ind, tgint] = sol[tgint0]
                    # compute coefficient ii in meant
                    # factor to account for the complex conjugate
                    prot = self._insert_boundaries(2 * np.real(sol[tgint0]), it0, itn)
                    meant[ii] = self.integz(prot)
                    dprot = np.dot(self.dr1, prot)
                    qtop[ii] = -dprot[0]
                    # if phi_top is not None and phi_bot is not None:
                    # translation velocity possible
                    # self.full_w0[ind] = np.real(sol[igw0])
                    # else:
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
