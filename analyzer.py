import dmsuite.dmsuite as dm
import numpy as np
import numpy.ma as ma
from scipy import linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt
# import seaborn as sns
from physics import PhysicalProblem, wtran
from misc import build_slices, normalize_modes


def cartesian_matrices_0(self, ra_num):
    """LHS matrix for x-independent forcing

    When the RHS is independent of x, the also solution is,
    and the velocity is uniform and only vertical, possibly null.
    Only the pressure, temperature and uniform vertical velocity
    are solved"""
    ncheb = self._ncheb
    dz1, dz2 = self.dr1, self.dr2
    one = np.identity(ncheb + 1)  # identity
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot

    # pressure
    if phi_top is not None and phi_bot is not None:
        # only that case a translating vertical velocity is possible
        ip0 = 0
        ipn = ncheb
    else:
        ip0 = 1
        ipn = ncheb - 1
    # temperature
    it0 = 0 if (heat_flux_top is not None) else 1
    itn = ncheb if (heat_flux_bot is not None) else ncheb - 1
    # global indices and slices in the solution containing only P and T
    _, igf, slall, slint, slgall, slgint = build_slices([(ip0, ipn),
                                                         (it0, itn)],
                                                        ncheb)
    ipg, itg = igf
    pall, tall = slall
    pint, tint = slint
    pgall, tgall = slgall
    pgint, tgint = slgint

    # index for vertical velocity
    if phi_top is not None and phi_bot is not None:
        igw = itg(itn+1)
    else:
        igw = itg(itn)

    # global indices and slices in the total solution
    

    # initialize matrix
    lmat = np.zeros((igw + 1, igw + 1))
    # pressure equation (z momentum)
    lmat[pgint, pgall] = - dz1[pint, pall]
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
        lmat[ipn, 0] = 1
        lmat[0, igw] = phi_bot
        # equation for the uniform vertical velocity
        lmat[igw, igw] = phi_top + phi_bot
        lmat[igw, 0] = 1
        lmat[igw, ipn] = -1

    return lmat, pgint, tgint, pgall, tgall, igw

def cartesian_matrices(self, wnk, ra_num, ra_comp=None):
    """Build left and right matrices in cartesian case"""
    ncheb = self._ncheb
    zphys = self.rad
    h_int = self.phys.h_int
    dz1, dz2 = self.dr1, self.dr2
    one = np.identity(ncheb+1)  # identity
    dh1 = 1.j * wnk * one  # horizontal derivative
    lapl = dz2 - wnk**2 * one  # laplacian
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    lewis = self.phys.lewis
    composition = self.phys.composition
    comp_terms = lewis is not None or composition is not None
    translation = self.phys.ref_state_translation
    if comp_terms and ra_comp is None:
        raise ValueError("ra_comp must be specified for compositional problem")

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
        rtr = 12*(phi_top+phi_bot)
        wtrans = wtran((ra_num-rtr)/rtr)[0]

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
    lmat[wgint, tgall] = ra_num * one[wint, tall]
    if comp_terms:
        lmat[wgint, cgall] = ra_comp * one[wint, call]
    if phi_bot is not None:
        # phase change at bot
        lmat[iwg(iwn), pgall] = -one[iwn, pall]
        lmat[iwg(iwn), wgall] = -phi_bot * one[iwn, wall] + 2 * dz1[iwn, wall]

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T

    # Neumann boundary condition if imposed flux
    if heat_flux_top is not None:
        lmat[itg(it0), tgall] = dz1[it0, tall]
    elif heat_flux_bot is not None:
        lmat[itg(itn), tgall] = dz1[itn, tall]

    lmat[tgint, tgall] = lapl[tint, tall]

    # need to take heat flux into account in T conductive
    if translation:
        # only written for Dirichlet BCs on T and without internal heating
        lmat[tgint, tgall] -= wtrans * dz1[tint, tall]
        lmat[tgint, wgall] = \
            np.diag(np.exp(wtrans * self.rad[wall]))[tint, wall]
        if np.abs(wtrans)>1.e-3:
            lmat[tgint, wgall] *= wtrans/(2*np.sinh(wtrans/2))
        else:
            # use a limited development
            lmat[tgint, wgall] *= (1-wtrans**2/24)
    else:
        grad_tcond = - h_int * zphys
        if heat_flux_bot is not None:
            grad_tcond += heat_flux_bot - h_int / 2
        elif heat_flux_top is not None:
            grad_tcond += heat_flux_top + h_int / 2
        else:
            grad_tcond += 1
        lmat[tgint, wgall] = np.diag(grad_tcond)[tint, wall]

    rmat[tgint, tgall] = one[tint, tall]

    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if composition is not None:
        lmat[cgint, wgall] = -np.diag(
            np.dot(dz1, composition(zphys)))[cint, wall]
    elif lewis is not None:
        lmat[cgint, wgall] = one[cint, wall]
        lmat[cgint, cgall] = lapl[cint, call] / lewis
    if comp_terms:
        rmat[cgint, cgall] = one[cint, call]
    return lmat, rmat


def spherical_matrices(self, l_harm, ra_num, ra_comp=None):
    """Build left and right matrices in spherical case"""
    rad = self.rad
    ncheb = self._ncheb
    dr1, dr2 = self.dr1, self.dr2
    orad1 = 1 / rad
    orad2 = 1 / rad**2
    orad3 = 1 / rad**3
    rad = np.diag(rad)
    orad1 = np.diag(orad1)
    orad2 = np.diag(orad2)
    orad3 = np.diag(orad3)
    one = np.identity(ncheb + 1)  # identity
    lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
    lapl = dr2 + 2 * np.dot(orad1, dr1) - lh2 * orad2  # laplacian
    gamma = self.phys.gamma
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    h_int = self.phys.h_int
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    if self.phys.eta_r is not None:
        eta_r = np.diag(np.vectorize(self.phys.eta_r)(np.diag(rad)))
    else:
        eta_r = one
    lewis = self.phys.lewis
    composition = self.phys.composition
    comp_terms = lewis is not None or composition is not None
    translation = self.phys.ref_state_translation

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

    lmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1))
    rmat = np.zeros((igf[-1](i_ns[-1]) + 1, igf[-1](i_ns[-1]) + 1))

    # Poloidal potential equations
    if phi_top is not None:
        # free-slip at top
        lmat[ipg(ip0), pgall] = dr2[ip0, pall] + \
            (lh2 - 2) * orad2[ip0, pall]
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
        lmat[ipg(ipn), pgall] = dr2[ipn, pall] + \
            (lh2 - 2) * orad2[ipn, pall]
    else:
        lmat[ipg(ipn), pgall] = one[ipn, pall]

    # Q equations
    # normal stress continuity at top
    if phi_top is not None:
        lmat[iqg(iq0), pgall] = lh2 * (self.phys.phi_top *
            orad1[iq0, pall] - 2 * np.dot(eta_r, orad2)[iq0, pall] +
            2 * np.dot(eta_r, np.dot(orad1, dr1))[iq0, pall])
        lmat[iqg(iq0), qgall] = -eta_r[iq0, qall] - \
            np.dot(eta_r, np.dot(rad, dr1))[iq0, qall]
    elif freeslip_top:
        lmat[iqg(iq0), pgall] = dr2[iq0, pall]
    else:
        # rigid
        lmat[iqg(iq0), pgall] = dr1[iq0, pall]
    if self.phys.eta_r is not None:
        deta_dr = np.diag(np.dot(dr1, np.diag(eta_r)))
        d2eta_dr2 = np.diag(np.dot(dr2, np.diag(eta_r)))
        lmat[qgint, pgall] = (2 * (lh2 - 1) *
            (np.dot(orad2, d2eta_dr2) - np.dot(orad3, deta_dr)) -
            2 * np.dot(np.dot(orad1, d2eta_dr2) - np.dot(orad2, deta_dr), dr1)
            )[qint, pall]
        lmat[qgint, qgall] = (np.dot(eta_r, lapl) + d2eta_dr2 +
            2 * np.dot(deta_dr, dr1))[qint, qall]
    else:
        # laplacian(Q) - RaT/r = 0
        lmat[qgint, qgall] = lapl[qint, qall]
    lmat[qgint, tgall] = - ra_num * orad1[qint, tall]
    if comp_terms:
        lmat[qgint, cgall] = - ra_comp * orad1[qint, call]
    # normal stress continuity at bot
    if phi_bot is not None:
        lmat[iqg(iqn), pgall] = lh2 * (-self.phys.phi_bot *
            orad1[iqn, pall] - 2 * np.dot(eta_r, orad2)[iqn, pall] +
            2 * np.dot(eta_r, np.dot(orad1, dr1))[iqn, pall])
        lmat[iqg(iqn), qgall] = -eta_r[iqn, qall] - \
            np.dot(eta_r, np.dot(rad, dr1))[iqn, qall]
    elif freeslip_bot:
        lmat[iqg(iqn), pgall] = dr2[iqn, pall]
    else:
        # rigid
        lmat[iqg(iqn), pgall] = dr1[iqn, pall]

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T

    # Neumann boundary condition if imposed flux
    if heat_flux_top is not None:
        lmat[itg(it0), tgall] = dr1[it0, tall]
    elif heat_flux_bot is not None:
        lmat[itg(itn), tgall] = dr1[itn, tall]

    lmat[tgint, tgall] = lapl[tint, tall]

    # advection of conductive profile
    # using u_r = l(l+1)/r P
    # first compute 1/r * nabla T
    # then multiply by l(l+1)
    grad_tcond = - h_int / 3
    if heat_flux_bot is not None:
        grad_tcond += (gamma**2 * heat_flux_bot +
                       h_int * gamma**3 / 3) * np.diag(orad3)
    elif heat_flux_top is not None:
        grad_tcond += (heat_flux_top + h_int / 3) * np.diag(orad3)
    else:
        grad_tcond += (1 + (1 - gamma**2) * h_int / 6) * gamma / (1 - gamma) \
            * np.diag(orad3)
    lmat[tgint, pgall] = np.diag(lh2 * grad_tcond)[tint, pall]

    rmat[tgint, tgall] = one[tint, tall]

    # C equations
    # 1/Le lapl(C) - u.grad(C_reference) = sigma C
    if composition is not None:
        grad_comp = np.diag(np.dot(dr1, composition(np.diag(rad))))
        lmat[cgint, pgall] = -lh2 * np.dot(orad1, grad_comp)[cint, pall]
    elif lewis is not None:
        raise ValueError('Finite Lewis not implemented in spherical')
    if comp_terms:
        rmat[cgint, cgall] = one[cint, call]

    return lmat, rmat


class Analyser:
    """Define various elements common to both analysers"""

    def __init__(self, phys, ncheb=15, nnonlin=2):
        """Create a generic analyzer
        
        phys is the PhysicalProblem
        ncheb is the number of Chebyshev nodes
        nnonlin is the maximum order of non-linear analysis
        """
        # get differentiation matrices
        self._ncheb = ncheb
        self._nnonlin = nnonlin
        # dm should be modified to go from 0 to ncheb
        self._zcheb, self._ddm = dm.chebdif(self._ncheb+1, 2)

        # weights
        self._invcp = np.ones(ncheb+1)
        self._invcp[0] = 1/2
        self._invcp[-1] = 1/2
        # matrix to get the pseudo-spectrum
        self._tmat = np.zeros((ncheb+1, ncheb+1))
        for n in range(ncheb+1):
            for p in range(ncheb+1):
                self._tmat[n, p] = (-1)**n * np.cos(n * p *np.pi / ncheb)

        self.phys = phys
        self.phys.bind_to(self)


    def _insert_boundaries(self, mode, im0, imn):
        """Insert zero at boundaries of mode if needed
        
        This need to be done when Dirichlet BCs are applied
        """
        if im0 == 1:
            mode = np.insert(mode, [0], [0])
        if imn == self._ncheb - 1:
            mode = np.append(mode, 0)
        return mode

    @property
    def phys(self):
        """Property holding the physical problem"""
        return self._phys

    @phys.setter
    def phys(self, phys_obj):
        """Change analyzed physical problem"""
        # Chebyshev polynomials are -1 < z < 1
        if phys_obj.spherical:
            # physical space is gamma < r < 1
            shrink_geom = (1 - phys_obj.gamma) / 2
            self.rad = self._zcheb * shrink_geom + (1 + phys_obj.gamma) / 2
            self.dr1 = self._ddm[0,:,:] / shrink_geom  # first r-derivative
            self.dr2 = self._ddm[1,:,:] / shrink_geom**2  # second r-derivative
        else:
            # physical space is -1/2 < z < 1/2
            self.rad = self._zcheb / 2
            self.dr1 = self._ddm[0,:,:] * 2  # first r-derivative
            self.dr2 = self._ddm[1,:,:] * 4  # second r-derivative
        self._phys = phys_obj

    def _slices(self):
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
        i_0 = 0 if (heat_flux_top is not None) else 1
        i_n = ncheb if (heat_flux_bot is not None) else ncheb - 1
        i0n.append((i_0, i_n))
        if self.phys.composition is not None or self.phys.lewis is not None:
            i0n.append((1, ncheb - 1))
        return build_slices(i0n, ncheb)

    def matrices(self, harm, ra_num, ra_comp=None):
        """Build left and right matrices"""
        if self.phys.spherical:
            return spherical_matrices(self, harm, ra_num, ra_comp)
        else:
            return cartesian_matrices(self, harm, ra_num, ra_comp)

    def eigval(self, harm, ra_num, ra_comp=None):
        """Compute the max eigenvalue

        harm: wave number
        ra_num: thermal Rayleigh number
        ra_comp: compositional Ra
        """
        lmat, rmat = self.matrices(harm, ra_num, ra_comp)
        eigvals = linalg.eigvals(lmat, rmat)
        return np.max(np.real(ma.masked_invalid(eigvals)))

    def eigvec(self, harm, ra_num, ra_comp=None):
        """Compute the max eigenvalue and associated eigenvector

        harm: wave number
        ra_num: thermal Rayleigh number
        ra_comp: compositional Ra
        """
        lmat, rmat = self.matrices(harm, ra_num, ra_comp)
        eigvals, eigvecs = linalg.eig(lmat, rmat)
        iegv = np.argmax(np.real(ma.masked_invalid(eigvals)))
        return eigvals[iegv], eigvecs[:, iegv]

    def _split_mode_cartesian(self, eigvec, apply_bc=False):
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

    def _split_mode_spherical(self, eigvec, l_harm, apply_bc=False):
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
        q_mode = eigvec[qgall]
        t_mode = eigvec[tgall]

        if apply_bc:
            p_mode = self._insert_boundaries(p_mode, ip0, ipn)
            t_mode = self._insert_boundaries(t_mode, it0, itn)

        ur_mode = l_harm * (l_harm + 1) * p_mode / self.rad
        up_mode = 1j * l_harm * (np.dot(self.dr1, p_mode) + p_mode / self.rad)

        return (p_mode, up_mode, ur_mode, t_mode)

    def split_mode(self, eigvec, harm, apply_bc=False):
        """Generic splitting function"""
        if self.phys.spherical:
            return self._split_mode_spherical(eigvec, harm, apply_bc)
        else:
            return self._split_mode_cartesian(eigvec, apply_bc)

    def _join_mode_cartesian(self, mode):
        """concatenate (p, u, w, t) mode into 1D cartesian eigvec"""
        # global indices and slices
        i0n, igf, _, slint, slgall, slgint = self._slices()
        i_0s, i_ns = zip(*i0n)
        if self.phys.composition is None and self.phys.lewis is None:
            ip0, iu0, iw0, it0 = i_0s
            ipn, iun, iwn, itn = i_ns
            pint, uint, wint, tint = slint
            pgall, ugall, wgall, tgall = slgall
            pgint, ugint, wgint, tgint = slgint
        else:
            ip0, iu0, iw0, it0, ic0 = i_0s
            ipn, iun, iwn, itn, icn = i_ns
            pint, uint, wint, tint, cint = slint
            pgall, ugall, wgall, tgall, cgall = slgall
            pgint, ugint, wgint, tgint, cgint = slgint

        (pmod, umod, wmod, tmod) = mode

        eigvec = np.zeros(igf[-1](i_ns[-1]) + 1) + 0j
        eigvec[pgint] = pmod[pint]
        eigvec[ugint] = umod[uint]
        eigvec[wgint] = wmod[wint]
        eigvec[tgint] = tmod[tint]

            # c_mode should be added in case of composition
        return eigvec

        
class LinearAnalyzer(Analyser):

    """Perform linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def neutral_ra(self, harm, ra_guess=600, ra_comp=None, eps=1.e-8):
        """Find Ra which gives neutral stability of a given harmonic

        harm is the wave number k or spherical harmonic degree
        """
        ra_min = ra_guess / 2
        ra_max = ra_guess * 2
        sigma_min = np.real(self.eigval(harm, ra_min, ra_comp))
        sigma_max = np.real(self.eigval(harm, ra_max, ra_comp))

        while sigma_min > 0. or sigma_max < 0.:
            if sigma_min > 0.:
                ra_max = ra_min
                ra_min /= 2
            if sigma_max < 0.:
                ra_min = ra_max
                ra_max *= 2
            sigma_min = np.real(self.eigval(harm, ra_min, ra_comp))
            sigma_max = np.real(self.eigval(harm, ra_max, ra_comp))

        while (ra_max - ra_min) / ra_max > eps:
            ra_mean = (ra_min + ra_max) / 2
            sigma_mean = np.real(self.eigval(harm, ra_mean, ra_comp))
            if sigma_mean < 0.:
                sigma_min = sigma_mean
                ra_min = ra_mean
            else:
                sigma_max = sigma_mean
                ra_max = ra_mean

        return (ra_min*sigma_max - ra_max*sigma_min) / (sigma_max - sigma_min)

    def fastest_mode(self, ra_num, ra_comp=None, harm=2):
        """Find the fastest growing mode at a given Ra"""
        if self.phys.spherical:
            harms = range(max(1, harm - 10), harm + 10)
        else:
            eps = [0.1, 0.01]
            harms = np.linspace(harm * (1 - 2 * eps[0]), harm * (1 + eps[0]), 3)

        sigma = [self.eigval(harm, ra_num, ra_comp) for harm in harms]
        if self.phys.spherical:
            max_found = False
            while not max_found:
                max_found = True
                if harms[0] != 1 and sigma[0] > sigma[1]:
                    hs_smaller = range(max(1, harms[0]-10), harms[0])
                    s_smaller = [self.eigval(h, ra_num, ra_comp)
                                 for h in hs_smaller]
                    harms = range(hs_smaller[0], harms[-1] + 1)
                    sigma = s_smaller + sigma
                    max_found = False
                if sigma[-1] > sigma[-2]:
                    hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                    s_greater = [self.eigval(h, ra_num, ra_comp)
                                 for h in hs_greater]
                    harms = range(harms[0], hs_greater[-1] + 1)
                    sigma = sigma + s_greater
                    max_found = False
            imax = np.argmax(sigma)
            smax = sigma[imax]
            hmax = harms[imax]
        else:
            pol = np.polyfit(harms, sigma, 2)
            # maximum value
            hmax = -0.5*pol[1]/pol[0]
            for i, err in enumerate([0.03, 1.e-3]):
                while np.abs(hmax-harms[1]) > err*hmax:
                    harms = np.linspace(hmax * (1 - eps[i]), hmax * (1 + eps[i]), 3)
                    sigma  = [self.eigval(h, ra_num, ra_comp) for h in harms]
                    pol = np.polyfit(harms, sigma, 2)
                    hmax = -0.5*pol[1]/pol[0]
                    smax = sigma[1]

        return smax, hmax

    def ran_l_mins(self):
        """Find neutral Rayleigh of mode giving square cells and of mode l=1"""
        if not self.phys.spherical:
            raise ValueError('ran_l_mins expects a spherical problem')
        lmax = 2
        rans = [self.neutral_ra(h) for h in (1,2)]
        ranp, ran = rans
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

    def critical_ra(self, harm=2, ra_guess=600, ra_comp=None):
        """Find the harmonic with the lower neutral Ra

        harm is an optional initial guess
        ra_guess is a guess for Ra_c
        """
        # find 3 values of Ra for 3 different harmonics
        eps = [0.1, 0.01]
        if self.phys.spherical:
            harms = range(max(1, harm - 10), harm + 10)
        else:
            harms = np.linspace(harm*(1-eps[0]), harm*(1+2*eps[0]), 3)
        ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]

        if self.phys.spherical:
            min_found = False
            while not min_found:
                min_found = True
                if harms[0] != 1 and ray[0] < ray[1]:
                    hs_smaller = range(max(1, harms[0]-10), harms[0])
                    ray_smaller = [self.neutral_ra(h, ray[0], ra_comp)
                                   for h in hs_smaller]
                    harms = range(hs_smaller[0], harms[-1] + 1)
                    ray = ray_smaller + ray
                    min_found = False
                if ray[-1] < ray[-2]:
                    hs_greater = range(harms[-1] + 1, harms[-1] + 10)
                    ray_greater = [self.neutral_ra(h, ray[-1], ra_comp)
                                   for h in hs_greater]
                    harms = range(harms[0], hs_greater[-1] + 1)
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
            kmin = -0.5*pol[1]/pol[0]
            for i, err in enumerate([0.03, 1.e-3]):
                while np.abs(kmin-harms[1]) > err*kmin and not exitloop:
                    harms = np.linspace(kmin*(1-eps[i]), kmin*(1+eps[i]), 3)
                    ray = [self.neutral_ra(h, ra_guess, ra_comp) for h in harms]
                    pol = np.polyfit(harms, ray, 2)
                    kmin = -0.5*pol[1]/pol[0]
                    ra_guess = ray[1]
            hmin = kmin

        return ra_guess, hmin


class NonLinearAnalyzer(Analyser):

    """Perform non-linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def integz(self, prof):
        """Integral on the -1/2 <= z <= 1/2 interval"""
        ncheb = self._ncheb
        invcp = self._invcp
        tmat = self._tmat

        # pseudo-spectrum
        spec = np.dot(tmat, prof * invcp)
        spec *= 2 / ncheb *invcp
        intz = - 1/2 * np.sum(spec[i]*2/(i**2-1) for i in range(len(spec)) if i % 2 == 0)
        # factor 1/2 is to account for the interval -1/2 < z < 1/2 
        return intz

    def indexmat(self, order, ind=1, harmm=None):
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
        index = 0
        for n in range(1, order+1):
            if n % 2 == 0:
                jj = np.int(n/2)
                indices = [i for i in range(0, n+1, 2)]
            else:
                jj = np.int((n-1)/2)
                indices = [i for i in range(1, n+1, 2)]
            nmax += jj + 1
            if ordn ==0 and nmax >= ind:
                ordn = n
                harm = indices[ind - nmax + jj]
            if harmm is not None:
                if n == order:
                    index += np.where(np.array(indices)==harmm)[0][0]
                else:
                    index += len(indices)
        return nmax, ordn, harm, index

    def ntermprod(self, mle, mri, harm):
        """One non-linear term on the RHS

        input : orders of the two modes to be combined, mle (left) and mri (right)
        input values for the modes themselves are taken from the predefined
        full_sol array, results are added to the nterm array.
        ordered by wavenumber
        """
        ncheb = self._ncheb
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
            indri = self.indexmat(mri, harmm=2*lr+yri)[3]
            dtr = np.dot(self.dr1[1:ncheb, 1:ncheb], self.full_t[indri])
            for ll in range(lle + 1):
                # index for the left mode in matrix
                indle = self.indexmat(mle, harmm=2*ll+yle)[3]
                # index for nmo and ll+lr
                nharmm = 2*(ll+lr)+yle+yri
                iind = self.indexmat(nmo, harmm=nharmm)[3]
                self.ntermt[iind] += 1j * harm * (2 * lr + yri) * \
                    self.full_u[indle] * self.full_t[indri] +\
                    self.full_w[indle] * dtr
                # index for nmo and ll-lr
                nharmm = 2*(ll-lr)+yle-yri
                iind = self.indexmat(nmo, harmm=np.abs(nharmm))[3]
                if nharmm > 0 :
                    self.ntermt[iind] += -1j * harm * (2 * lr + yri) * \
                        self.full_u[indle] * np.conj(self.full_t[indri]) + \
                        self.full_w[indle] * np.conj(dtr)
                elif nharmm == 0:
                    # add complex conjugate -> times 2. No! Otherwise it is counted twice later!
                    self.ntermt[iind] += np.real(-1j * harm * (2 * lr + yri) * \
                        self.full_u[indle] * np.conj(self.full_t[indri]) + \
                        self.full_w[indle] * np.conj(dtr))                    
                else:
                    self.ntermt[iind] += 1j * harm * (2 * lr + yri) * \
                        np.conj(self.full_u[indle]) * self.full_t[indri] + \
                        np.conj(self.full_w[indle]) * dtr
        return

    def nonlinana(self):
        """Ra2 and X2"""
        ncheb = self._ncheb
        nnonlin = self._nnonlin
        phi_top = self.phys.phi_top
        phi_bot = self.phys.phi_bot
        freeslip_top = self.phys.freeslip_top
        freeslip_bot = self.phys.freeslip_bot
        heat_flux_top = self.phys.heat_flux_top
        heat_flux_bot = self.phys.heat_flux_bot
        zcheb = self._zcheb
        # global indices and slices
        i0n, igf, slall, slint, slgall, slgint = self._slices()
        i_0s, i_ns = zip(*i0n)
        ip0, iu0, iw0, it0 = i_0s
        ipn, iun, iwn, itn = i_ns
        ipg, iug, iwg, itg = igf
        pall, uall, wall, tall = slall
        pint, uint, wint, tint = slint
        pgall, ugall, wgall, tgall = slgall
        pgint, ugint, wgint, tgint = slgint

        # First compute the linear mode and matrix
        ana = LinearAnalyzer(self.phys, self._ncheb)
        ra_c, harm_c = ana.critical_ra()
        lmat_c, _ = self.matrices(harm_c, ra_c)
        _, mode_c = self.eigvec(harm_c, ra_c)
        modec = self.split_mode(mode_c, harm_c, apply_bc=True)
        modec, _ = normalize_modes(modec, norm_mode=2, full_norm=False)

        # setup matrices for the non linear solution
        nmodez = np.shape(mode_c)
        nkmax = self.indexmat(nnonlin+1)[0]
        self.full_sol = np.zeros((nkmax, nmodez[0])) * (1+1j)
        self.full_u = self.full_sol[:, ugint]
        self.full_w = self.full_sol[:, wgint]
        self.full_t = self.full_sol[:, tgint]
        self.full_w0 = np.zeros(nkmax) # translation velocity
        self.nterm = np.zeros(self.full_sol.shape) * (1+1j)
        self.rhs = np.zeros(self.full_sol.shape) * (1+1j)
        # temperature part, the only non-null one in the non-linear term
        self.ntermt = self.nterm[:, tgint]
        # the suite of Rayleigh numbers
        ratot = np.zeros(nnonlin+1)
        ratot[0] = ra_c
        # coefficient for the average temperature
        meant = np.zeros(nnonlin+1)
        meant[0] = 1/2
        # coefficient for the nusselt number
        qtop = np.zeros(nnonlin+1)
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
        xcmxc =  self.integz(np.real(w_c * t_c))

        rr = zcheb/2
        # replace with the low phi expansion for testing
        # t_c = (1 - 4 * rr**2) / 16
        # w_c = np.ones(w_c.shape)/2
        # u_c = - 3 * 1j / 16 *np.sqrt(2 * phi_top) * rr
        # harm_c = np.sqrt(phi_top / 2) * 3 / 4
        # ra_c = 24 * phi_top -81 / 256 * phi_top**2

        dt_c = np.dot(self.dr1, t_c)
        lmat = np.zeros((nnonlin+1, lmat_c.shape[0], lmat_c.shape[1])) * (1+1j)
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
                _, _, _, ind = self.indexmat(ii, harmm=1)
                prof = self._insert_boundaries(np.real(self.full_t[0] *\
                                                        np.conj(self.ntermt[ind])), it0, itn)
                ratot[ii-1] = self.integz(prof)
                for jj in range(1, lii):
                    # only the ones in harm_c can contribute for each degree
                    _, _, _, ind = self.indexmat(2 * (lii - jj) + 1, harmm=1)
                    # + sign because denominator takes the minus.
                    prof = self._insert_boundaries(np.real(self.full_w[0] *\
                                                           self.full_t[ind]), it0, itn)
                    ratot[ii-1] += ratot[2 * jj] * self.integz(prof)
                ratot[ii-1] *= ratot[0] / xcmxc

            # add mterm to nterm to get rhs
            _, _, _, imin = self.indexmat(ii, harmm=yii)
            _, _, _, imax = self.indexmat(ii, harmm=ii)
            self.rhs[imin:imax+1] = self.nterm[imin:imax+1]
            jmax = lii if yii==0 else lii+1
            for jj in range(2, jmax, 2):
                # jj is index for Ra
                # index for MX is ii-jj
                (lmx, ymx) = divmod(ii - jj, 2)
                for kk in range(lmx + 1):
                    _, _, _, indjj = self.indexmat(ii, harmm=2*kk+ymx)
                    self.rhs[indjj, wgint] -= ratot[jj] * self.full_t[indjj]

            # invert matrix for each harmonic number
            for jj in range(lii+1):
                # index to fill in: same parity as ii
                harmjj = 2*jj+yii
                _, _, _, ind = self.indexmat(ii, harmm=harmjj)
                if harmjj == 0 : # special treatment for 0 modes.
                    # should be possible to avoid the use of a rhs0
                    rhs0 = np.zeros(lmat0.shape[1]) * (1 + 1j)
                    rhs0[tgall0] = self.rhs[ind, tgall]
                    sol = solve(lmat0, rhs0)
                    self.full_sol[ind, pgint] = sol[pgint0]
                    self.full_sol[ind, tgint] = sol[tgint0]
                    # compute coefficient ii in meant
                    # factor to account for the complex conjugate
                    prot = self._insert_boundaries(2 * np.real(sol[tgint0]), it0, itn)
                    meant[ii] = self.integz(prot)
                    dprot = np.dot(self.dr1, prot)
                    qtop[ii] = - dprot[0]
                    if phi_top is not None and phi_bot is not None:
                        # translation velocity possible
                        self.full_w0[ind] = np.real(sol[igw0])
                        # print('w0 =', self.full_w0[ind])
                    else:
                        self.full_w0[ind] = 0
                else:
                    self.full_sol[ind] = solve(lmat[harmjj - 1], self.rhs[ind])

        # part proportional to epsilon^2
        # moyv2 = 2 * self.integz(np.imag(u_c)**2)
        # moyv2 += 2 * self.integz(np.imag(w_c)**2)
        # part proportional to epsilon^4
        # moyv4 = 2 * self.integz(np.imag(u22)**2)
        # moyv4 += w20**2 # a constant
        # moyv4 += 2 * self.integz(np.real(w22)**2)

        return harm_c, ratot, self.full_sol, meant, qtop
