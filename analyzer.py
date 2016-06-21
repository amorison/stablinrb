import dmsuite.dmsuite as dm
import numpy as np
import numpy.ma as ma
from scipy import linalg
from scipy.optimize import brentq
from numpy.linalg import solve
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from physics import PhysicalProblem, wtran
from misc import normalize_modes


def cartesian_matrices_0(self, ra_num):
    """LHS matrix for x-independent forcing

    When the RHS is independent of x, the solution is also
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
    # if (phi_top is not None) else 1
    # ipn = ncheb if (phi_bot is not None) else ncheb - 1
    # ip0 = 1
    # ipn = ncheb-1
    # temperature
    it0 = 0 if (heat_flux_top is not None) else 1
    itn = ncheb if (heat_flux_bot is not None) else ncheb - 1
    # global indices
    ipg = lambda idx: idx - ip0
    itg = lambda idx: idx - it0 + ipg(ipn) + 1
    # slices
    # entire vector
    pall = slice(ip0, ipn + 1)
    tall = slice(it0, itn + 1)
    # interior points
    pint = slice(1, ncheb)
    tint = slice(1, ncheb)
    # entire vector with big matrix indexing
    pgall = slice(ipg(ip0), ipg(ipn + 1))
    tgall = slice(itg(it0), itg(itn + 1))
    # interior points with big matrix indexing
    pgint = slice(ipg(1), ipg(ncheb))
    tgint = slice(itg(1), itg(ncheb))
    # index for vertical velocity
    if phi_top is not None and phi_bot is not None:
        igw = itg(itn+1)
    else:
        igw = itg(itn)

    # initialize matrix
    lmat = np.zeros((igw + 1, igw + 1))
    # pressure equation (z momentum)
    lmat[pgint, pgall] = - dz1[pint, pall]
    lmat[pgint, tgint] = ra_num * one[pint, tint]
    # temperature equation
    lmat[tgint, tgall] = dz2[tint, tall]
    # the case for a translating vertical velocity (mode 0)
    if phi_top is not None and phi_bot is not None:
        lmat[tgint, igw] = 1
        lmat[0, 0] = -1
        lmat[0, igw] = phi_top
        lmat[ipn, 0] = 1
        lmat[0, igw] = phi_bot
        lmat[igw, igw] = phi_top + phi_bot
        lmat[igw, 0] = 1
        lmat[igw, ipn] = -1

    return lmat, pgint, tgint, igw

def cartesian_matrices(self, wnk, ra_num):
    "LHS and RHS matrices for the linear stability"
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
    translation = self.phys.ref_state_translation

    # global indices and slices
    i0n, igf, slall, slint, slgall, slgint = self._slices()
    ip0, ipn, iu0, iun, iw0, iwn, it0, itn = i0n
    ipg, iug, iwg, itg = igf
    pall, uall, wall, tall = slall
    pint, uint, wint, tint = slint
    pgall, ugall, wgall, tgall = slgall
    pgint, ugint, wgint, tgint = slgint

    # For pressure. No BCs but side values needed or removed
    # depending on the BCs for W. number of lines need to be
    # the same as that of d2w and depends on bcsw.

    if translation:
        rtr = 12*(phitop+phibot)
        wtrans = wtran((ranum-rtr)/rtr)[0]

    lmat = np.zeros((itg(itn) + 1, itg(itn) + 1)) + 0j
    rmat = np.zeros((itg(itn) + 1, itg(itn) + 1))

    # Pressure equations
    # mass conservation
    lmat[pgall, ugall] = dh1[pall, uall]
    lmat[pgall, wgall] = dz1[pall, wall]

    # U equations
    # free-slip at top
    if phi_top or freeslip_top:
        lmat[iug(iu0), ugall] = dz1[iu0, uall]
    if phi_top:
        lmat[iug(iu0), wgall] = dh1[iu0, wall]
    # horizontal momentum conservation
    lmat[ugint, pgall] = -dh1[uint, pall]
    lmat[ugint, ugall] = lapl[uint, uall]
    # free-slip at bot
    if phi_bot or freeslip_bot:
        lmat[iug(iun), ugall] = dz1[iun, uall]
    if phi_top:
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
        lmat[tgint, wgall] = \
            np.diag(np.exp(wtrans * self.rad[wall]))[tint, wall]
        lmat[tgint, tgall] -= wtrans * dz1[tint, tall]
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

    return lmat, rmat

def eigval_cartesian(self, wnk, ra_num):
    """Compute the max eigenvalue and associated eigenvector

    wnk: wave number
    ra_num: Rayleigh number
    """
    ncheb = self._ncheb
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
    translation = self.phys.ref_state_translation

    # global indices and slices
    i0n, _, _, _, slgall, _ = self._slices()
    ip0, ipn, iu0, iun, iw0, iwn, it0, itn = i0n
    pgall, ugall, wgall, tgall = slgall

    lmat, rmat = cartesian_matrices(self, wnk, ra_num)

    # Find the eigenvalues
    eigvals, eigvecs = linalg.eig(lmat, rmat, right=True)
    index = np.argsort(ma.masked_invalid(eigvals))
    eigvals = ma.masked_invalid(eigvals[index])
    iegv = np.argmax(np.real(eigvals))

    # Extract modes from eigenvector
    eigvecs = eigvecs[:, index]
    eigvecs = eigvecs[:, iegv]
    p_mode = eigvecs[pgall]
    u_mode = eigvecs[ugall]
    w_mode = eigvecs[wgall]
    t_mode = eigvecs[tgall]

    p_mode = self._insert_boundaries(p_mode, ip0, ipn)
    u_mode = self._insert_boundaries(u_mode, iu0, iun)
    w_mode = self._insert_boundaries(w_mode, iw0, iwn)
    t_mode = self._insert_boundaries(t_mode, it0, itn)

    return eigvals[iegv], (p_mode, u_mode, w_mode, t_mode), (lmat, rmat)


def eigval_spherical(self, l_harm, ra_num):
    """Compute the max eigenvalue and associated eigenvector

    l_harm: spherical harmonic degree
    ra_num: Rayleigh number
    """
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
    translation = self.phys.ref_state_translation

    # index min and max
    # poloidal
    ip0 = 0
    ipn = ncheb
    # laplacian of poloidal
    iq0 = 0
    iqn = ncheb
    # temperature
    it0 = 0 if (heat_flux_top is not None) else 1
    itn = ncheb if (heat_flux_bot is not None) else ncheb - 1

    # global indices
    ipg = lambda idx: idx - ip0
    iqg = lambda idx: idx - iq0 + ipg(ipn) + 1
    itg = lambda idx: idx - it0 + iqg(iqn) + 1

    # slices
    # entire vector
    pall = slice(ip0, ipn + 1)
    qall = slice(iq0, iqn + 1)
    tall = slice(it0, itn + 1)
    # interior points
    pint = slice(1, ncheb)
    qint = slice(1, ncheb)
    tint = slice(1, ncheb)
    # entire vector with big matrix indexing
    pgall = slice(ipg(ip0), ipg(ipn + 1))
    qgall = slice(iqg(iq0), iqg(iqn + 1))
    tgall = slice(itg(it0), itg(itn + 1))
    # interior points with big matrix indexing
    pgint = slice(ipg(1), ipg(ncheb))
    qgint = slice(iqg(1), iqg(ncheb))
    tgint = slice(itg(1), itg(ncheb))

    lmat = np.zeros((itg(itn) + 1, itg(itn) + 1))
    rmat = np.zeros((itg(itn) + 1, itg(itn) + 1))

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
            orad1[iq0, pall] - 2 * orad2[iq0, pall] +
            2 * np.dot(orad1, dr1)[iq0, pall])
        lmat[iqg(iq0), qgall] = -one[iq0, qall] - np.dot(rad, dr1)[iq0, qall]
    elif freeslip_top:
        lmat[iqg(iq0), pgall] = dr2[iq0, pall]
    else:
        # rigid
        lmat[iqg(iq0), pgall] = dr1[iq0, pall]
    # laplacian(Q) - RaT/r = 0
    lmat[qgint, qgall] = lapl[qint, qall]
    lmat[qgint, tgall] = - ra_num * orad1[qint, tall]
    # normal stress continuity at bot
    if phi_bot is not None:
        lmat[iqg(iqn), pgall] = lh2 * (-self.phys.phi_bot *
            orad1[iqn, pall] - 2 * orad2[iqn, pall] +
            2 * np.dot(orad1, dr1)[iqn, pall])
        lmat[iqg(iqn), qgall] = -one[iqn, qall] - np.dot(rad, dr1)[iqn, qall]
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

    # Find the eigenvalues
    eigvals, eigvecs = linalg.eig(lmat, rmat, right=True)
    index = np.argsort(ma.masked_invalid(eigvals))
    eigvals = ma.masked_invalid(eigvals[index])
    iegv = np.argmax(np.real(eigvals))

    # Extract modes from eigenvector
    eigvecs = eigvecs[:, index]
    eigvecs = eigvecs[:, iegv]
    p_mode = eigvecs[pgall]
    q_mode = eigvecs[qgall]
    t_mode = eigvecs[tgall]

    p_mode = self._insert_boundaries(p_mode, ip0, ipn)
    t_mode = self._insert_boundaries(t_mode, it0, itn)

    ur_mode = l_harm * (l_harm + 1) * p_mode / self.rad
    up_mode = 1j * l_harm * (np.dot(dr1, p_mode) + p_mode / self.rad)

    return eigvals[iegv], (p_mode, up_mode, ur_mode, t_mode), (lmat, rmat)

class Analyser:
    """Define various elements common to both analysers"""

    def __init__(self, phys, ncheb=15):
        """Create a generic analyzer
        
        phys is the PhysicalProblem
        ncheb is the number of Chebyshev nodes
        """
        # get differentiation matrices
        self._ncheb = ncheb
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
        # pressure
        ip0 = 0
        ipn = ncheb
        # horizontal velocity
        iu0 = 0 if (phi_top is not None) or freeslip_top else 1
        iun = ncheb if (phi_bot is not None) or freeslip_bot else ncheb - 1
        # vertical velocity
        iw0 = 0 if (phi_top is not None) else 1
        iwn = ncheb if (phi_bot is not None) else ncheb - 1
        # temperature
        it0 = 0 if (heat_flux_top is not None) else 1
        itn = ncheb if (heat_flux_bot is not None) else ncheb - 1
        i0n = (ip0, ipn, iu0, iun, iw0, iwn, it0, itn)
        # global indices
        ipg = lambda idx: idx - ip0
        iug = lambda idx: idx - iu0 + ipg(ipn) + 1
        iwg = lambda idx: idx - iw0 + iug(iun) + 1
        itg = lambda idx: idx - it0 + iwg(iwn) + 1
        igf = (ipg, iug, iwg, itg)
        # slices
        # entire vector
        pall = slice(ip0, ipn + 1)
        uall = slice(iu0, iun + 1)
        wall = slice(iw0, iwn + 1)
        tall = slice(it0, itn + 1)
        slall = (pall, uall, wall, tall)
        # interior points
        pint = slice(1, ncheb)
        uint = slice(1, ncheb)
        wint = slice(1, ncheb)
        tint = slice(1, ncheb)
        slint = (pint, uint, wint, tint)
        # entire vector with big matrix indexing
        pgall = slice(ipg(ip0), ipg(ipn + 1))
        ugall = slice(iug(iu0), iug(iun + 1))
        wgall = slice(iwg(iw0), iwg(iwn + 1))
        tgall = slice(itg(it0), itg(itn + 1))
        slgall = (pgall, ugall, wgall, tgall)
        # interior points with big matrix indexing
        pgint = slice(ipg(1), ipg(ncheb))
        ugint = slice(iug(1), iug(ncheb))
        wgint = slice(iwg(1), iwg(ncheb))
        tgint = slice(itg(1), itg(ncheb))
        slgint = (pgint, ugint, wgint, tgint)
        return i0n, igf, slall, slint, slgall, slgint

class LinearAnalyzer(Analyser):

    """Perform linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def eigval(self, harm, ra_num):
        """Generic eigval function"""
        if self.phys.spherical:
            return eigval_spherical(self, harm, ra_num)
        else:
            return eigval_cartesian(self, harm, ra_num)

    def neutral_ra(self, harm, ra_guess=600, eps=1.e-8):
        """Find Ra which gives neutral stability of a given harmonic

        harm is the wave number k or spherical harmonic degree
        """
        ra_min = ra_guess / 2
        ra_max = ra_guess * 2
        sigma_min = np.real(self.eigval(harm, ra_min)[0])
        sigma_max = np.real(self.eigval(harm, ra_max)[0])

        while sigma_min > 0. or sigma_max < 0.:
            if sigma_min > 0.:
                ra_max = ra_min
                ra_min /= 2
            if sigma_max < 0.:
                ra_min = ra_max
                ra_max *= 2
            sigma_min = np.real(self.eigval(harm, ra_min)[0])
            sigma_max = np.real(self.eigval(harm, ra_max)[0])

        while (ra_max - ra_min) / ra_max > eps:
            ra_mean = (ra_min + ra_max) / 2
            sigma_mean = np.real(self.eigval(harm, ra_mean)[0])
            if sigma_mean < 0.:
                sigma_min = sigma_mean
                ra_min = ra_mean
            else:
                sigma_max = sigma_mean
                ra_max = ra_mean

        return (ra_min*sigma_max - ra_max*sigma_min) / (sigma_max - sigma_min)

    def critical_ra(self, harm=2, ra_guess=600):
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
        ray = [self.neutral_ra(h, ra_guess) for h in harms]

        if self.phys.spherical:
            min_found = False
            while not min_found:
                imin = np.argmin(ray)
                ra_guess = ray[imin]
                hmin = harms[imin]
                if imin == 0 and hmin != 1:
                    ra_guess_n = ray[1]
                    harms = range(max(1, hmin - 10), hmin)
                    ray = [self.neutral_ra(h, ra_guess) for h in harms]
                    # append local minimum and the next point to
                    # avoid oscillating around the true minimum
                    ray.append(ra_guess)
                    ray.append(ra_guess_n)
                    harms = range(max(1, hmin - 10), hmin + 2)
                elif imin == len(ray) - 1:
                    harms = range(hmin + 1, hmin + 20)
                    ray = [self.neutral_ra(h, ra_guess) for h in harms]
                else:
                    min_found = True
        else:
            # fit a degree 2 polynomial
            pol = np.polyfit(harms, ray, 2)
            # minimum value
            exitloop = False
            kmin = -0.5*pol[1]/pol[0]
            for i, err in enumerate([0.03, 1.e-3]):
                while np.abs(kmin-harms[1]) > err*kmin and not exitloop:
                    harms = np.linspace(kmin*(1-eps[i]), kmin*(1+eps[i]), 3)
                    ray = [self.neutral_ra(h, ra_guess) for h in harms]
                    pol = np.polyfit(harms, ray, 2)
                    kmin = -0.5*pol[1]/pol[0]
                    # if kmin <= 1.e-3:
                        # exitloop = True
                        # kmin = 1.e-3
                        # ray[1] = self.neutral_ra(kmin, ra_guess)
                        # not able to properly converge anymore
                    ra_guess = ray[1]
            hmin = kmin

        return ra_guess, hmin


class NonLinearAnalyzer(Analyser):

    """Perform non-linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def matrices(self, harm, ra_num):
        """Generic eigval function"""
        if self.phys.spherical:
            return spherical_matrices(self, harm, ra_num) # not yet treated beyond that point
        else:
            return cartesian_matrices(self, harm, ra_num)

    def eigval(self, harm, ra_num):
        """Generic eigval function"""
        if self.phys.spherical:
            return eigval_spherical(self, harm, ra_num) # not yet treated beyond that point
        else:
            return eigval_cartesian(self, harm, ra_num)

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

    def nonlinana(self):
        """Ra2 and X2"""
        ncheb = self._ncheb
        phi_top = self.phys.phi_top
        phi_bot = self.phys.phi_bot
        freeslip_top = self.phys.freeslip_top
        freeslip_bot = self.phys.freeslip_bot
        heat_flux_top = self.phys.heat_flux_top
        heat_flux_bot = self.phys.heat_flux_bot
        zcheb = self._zcheb
        # global indices and slices
        i0n, igf, slall, slint, slgall, slgint = self._slices()
        ip0, ipn, iu0, iun, iw0, iwn, it0, itn = i0n
        ipg, iug, iwg, itg = igf
        pall, uall, wall, tall = slall
        pint, uint, wint, tint = slint
        pgall, ugall, wgall, tgall = slgall
        pgint, ugint, wgint, tgint = slgint

        # First compute the linear mode and matrix
        ana = LinearAnalyzer(self.phys, self._ncheb)
        ra_c, harm_c = ana.critical_ra()
        _, mode_c, mats_c = self.eigval(harm_c, ra_c)
        mode_c, _ = normalize_modes(mode_c, norm_mode=2, full_norm=False)
        p_c, u_c, w_c, t_c = mode_c
        p_c /= 2
        u_c /= 2
        w_c /= 2
        t_c /= 2
        lmat_c, _ = mats_c
        dt_c = np.dot(self.dr1, t_c)
        # also need the linear problem for wnk=2*harm_c and wnk=0
        lmat2 = self.matrices(2 * harm_c, ra_c)[0]
        lmat0, pgint0, tgint0, igw0 = cartesian_matrices_0(self, ra_c)

        # theta part of the non-linear term N(Xc, Xc)
        # part constant in x
        nxcxc0 = np.zeros((igw0 + 1))
        nxcxc0[tgint0] = 2 * (np.real(w_c * dt_c) + harm_c * np.imag(u_c) * np.real(t_c))[tint]

        # Solve Lc * X2 = NXcXc to get X20
        mode20 = solve(lmat0, nxcxc0)

        p20 = mode20[pgint0]
        t20 = mode20[tgint0]
        if phi_top is not None and phi_bot is not None:
            w20 = mode20[igw0]
        else:
            w20 = 0

        p20 = self._insert_boundaries(p20, ip0, ipn)
        t20 = self._insert_boundaries(t20, it0, itn)
        dt20 = np.dot(self.dr1, t20)
        # part in 2 k x
        nxcxc2 = np.zeros((itg(itn) + 1))
        nxcxc2[tgint] = 2 * (np.real(w_c * dt_c) - harm_c * np.imag(u_c) * np.real(t_c))[tint]

        # Solve Lc * X2 = NXcXc to get X22
        mode22 = solve(lmat2, nxcxc2)
        # print(mode22)
        p22 = mode22[pgall]
        u22 = mode22[ugall]
        w22 = mode22[wgall]
        t22 = mode22[tgall]

        p22 = self._insert_boundaries(p22, ip0, ipn)
        u22 = self._insert_boundaries(u22, iu0, iun)
        w22 = self._insert_boundaries(w22, iw0, iwn)
        t22 = self._insert_boundaries(t22, it0, itn)
        dt22 = np.dot(self.dr1, t22)

        # check the profiles
        # n_rad = 40
        # cheb_space = np.linspace(-1, 1, n_rad)
        # rad = np.linspace(-1/2, 1/2, n_rad)
        # u_interp = dm.chebint(u_c, cheb_space)
        # w_interp = dm.chebint(w_c, cheb_space)
        # t_interp = dm.chebint(t_c, cheb_space)
        # n2xc = self._insert_boundaries(nxcxc0[tgint0], 1, ncheb - 1)

        # n2xc_interp = dm.chebint(n2xc, cheb_space)
        # fig, axis = plt.subplots(1, 5, sharey=True)
        # axis[0].plot(t_interp, rad)
        # axis[0].plot(t_c, zcheb/2, 'o')
        # axis[0].plot(np.cos(np.pi*rad)/2/(np.pi**2+harm_c**2), rad)
        # axis[1].plot(w_interp, rad)
        # axis[1].plot(w_c, zcheb/2, 'o')
        # axis[1].plot(np.cos(np.pi*rad)/2, rad)
        # axis[2].plot(n2xc_interp, rad)
        # axis[2].plot(n2xc, zcheb/2, 'o')
        # axis[2].plot(-np.pi/2/(np.pi**2+harm_c**2)*np.sin(2*np.pi*rad), rad)
        # axis[3].plot(np.imag(u_interp), rad)
        # axis[3].plot(t20, zcheb/2, 'o')
        # axis[3].plot(1/(np.pi**2+harm_c**2)/(8*np.pi)*np.sin(2*np.pi*rad), rad)

        # denominator in Ra2
        xcmxc =  self.integz(np.real(w_c * t_c))
        # numerator in Ra2
        xcnx2xc = 2 * harm_c * self.integz(np.real(t_c)**2 * np.imag(u22))
        xcnx2xc += 2 * self.integz(np.real(t_c * dt_c) * (np.real(w22) + np.real(w20)))
        # non-linear term xc, x2, along t, in exp(ikx)
        # nxcx2tk = w_c * dt20
        # axis[4].plot(nxcx2tk, zcheb/2, 'o')
        # axis[4].plot(np.cos(np.pi*rad)/8/(np.pi**2+harm_c**2)*np.cos(2*np.pi*rad), rad)
        # axis[4].plot(np.cos(np.pi*rad)/16/(np.pi**2+harm_c**2)*np.cos(2*np.pi*rad), rad, '.-')
        
        # plt.savefig('tw.pdf', format='PDF')

        xcnxcx20 = self.integz(t_c * w_c * dt20)
        xcnxcx22 = self.integz(t_c * w_c * np.real(dt22))
        xcnxcx22 += 2 * harm_c * self.integz(t_c * np.imag(u_c) * np.real(dt22))
        xcnxcx2 = xcnxcx20 + xcnxcx22

        ra2 = ra_c * xcnxcx2/xcmxc

        # compute some global caracteristics
        moyt = 0.5 * (1 + self.integz(t20)) # 0.5 for scaling to -1/2 ; 1/2
        qtop = - dt20[0]
        # * 0.5 * 2 : scaling and times 2 for double product of e^ikx e^-ikx
        moyv = self.integz(np.imag(u_c)**2)
        moyv += self.integz(np.abs(u22)**2) # * 0.5 * 2 : scaling and times 2
        moyv += w20**2 # a constant
        moyv += self.integz(np.imag(w_c)**2) # * 0.5 * 2 : scaling and times 2
        moyv += self.integz(np.abs(w22)**2) # * 0.5 * 2 : scaling and times 2

        return harm_c, (ra_c, ra2), mode_c, (p20, w20, t20),\
          (p22, u22, w22, t22), (moyt, moyv, qtop)
