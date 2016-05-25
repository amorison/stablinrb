import dmsuite.dmsuite as dm
import numpy as np
import numpy.ma as ma
from scipy import linalg
from scipy.optimize import brentq

class PhysicalProblem:

    """Description of the physical problem"""

    def __init__(self, gamma=None,
                 phi_top=None, phi_bot=None,
                 freeslip_top=True, freeslip_bot=True,
                 heat_flux_top=None, heat_flux_bot=None,
                 ref_state_translation=False):
        """Create a physical problem instance

        gamma is r_bot/r_top, cartesian if None

        Boundary conditions:
        phi_*: phase change number, no phase change if None
        freeslip_*: whether free-slip of rigid if no phase change
        heat_flux_*: heat flux, Dirichlet condition if None
        """
        self._observers = []
        self.gamma = gamma
        self.phi_top = phi_top
        self.phi_bot = phi_bot
        self.freeslip_top = freeslip_top
        self.freeslip_bot = freeslip_bot
        self.heat_flux_top = heat_flux_top
        self.heat_flux_bot = heat_flux_bot
        self.ref_state_translation = ref_state_translation

    def bind_to(self, analyzer):
        """Connect analyzer to physical problem

        The analyzer will be warned whenever the physical
        problem geometry has changed"""
        self._observers.append(analyzer)

    def name(self):
        """Construct a name for the current case"""
        name = ['sph'] if self.spherical else ['cart']
        if self.phi_top is not None:
            name.append('phiT')
            name.append(np.str(self.phi_top).replace('.', '-'))
        else:
            name.append(
                'freeT' if self.freeslip_top else 'rigidT')
        if self.phi_bot is not None:
            name.append('phiB')
            name.append(np.str(self.phi_bot).replace('.', '-'))
        else:
            name.append(
                'freeB' if self.freeslip_bot else 'rigidB')
        return '_'.join(name)

    @property
    def gamma(self):
        """Aspect ratio of spherical geometry"""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Set spherical according to gamma"""
        self.spherical = (value is not None) and (0 < value < 1)
        self._gamma = value if self.spherical else None
        # warn bounded analyzers that the problem geometry has changed
        for analyzer in self._observers:
            analyzer.phys = self


def wtran(eps):
    """translation velocity as function of the reduced Rayleigh number"""
    if eps <= 0:
        wtr = 0
        wtrs = 0
        wtrl = 0
    else:
        # function whose roots are the translation velocity
        func = lambda wtra, eps: \
            wtra**2 * np.sinh(wtra / 2) - 6 * (1 + eps) * \
                (wtra * np.cosh(wtra / 2) - 2 * np.sinh(wtra / 2))
        # value in the large Ra limit
        wtrl = 6*(eps+1)
        ful = func(wtrl, eps)
        # small Ra limit
        wtrs = 2*np.sqrt(15*eps)
        fus = func(wtrs, eps)
        if fus*ful > 0:
            print('warning in wtran: ',eps, wtrs, fus, wtrl, ful)
            if eps < 1.e-2:
                wtr = wtrs
                print('setting wtr = wtrs= ', wtrs)
        # complete solution
        else:
            wtr = brentq(func, wtrs, wtrl, args=(eps))
    return wtr, wtrs, wtrl


def eigval_cartesian(self, wnk, ra_num):
    """Compute the max eigenvalue and associated eigenvector

    wnk: wave number
    ra_num: Rayleigh number
    """
    ncheb = self._ncheb
    dz1, dz2 = self.dr1, self.dr2
    one = np.identity(ncheb)  # identity
    dh1 = 1.j * wnk * one  # horizontal derivative
    lapl = dz2 - wnk**2 * one  # laplacian
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    translation = self.phys.ref_state_translation
    # index min and max
    # remove boundary when Dirichlet condition
    # pressure
    ip0 = 0
    ipn = ncheb - 1
    # horizontal velocity
    iu0 = 0 if (phi_top is not None) or freeslip_top else 1
    iun = ncheb - 1 if (phi_bot is not None) or freeslip_bot else ncheb - 2
    # vertical velocity
    iw0 = 0 if (phi_top is not None) else 1
    iwn = ncheb - 1 if (phi_bot is not None) else ncheb - 2
    # temperature
    it0 = 0 if (heat_flux_top is not None) else 1
    itn = ncheb - 1 if (heat_flux_bot is not None) else ncheb - 2

    # global indices
    ipg = lambda idx: idx - ip0
    iug = lambda idx: idx - iu0 + ipg(ipn) + 1
    iwg = lambda idx: idx - iw0 + iug(iun) + 1
    itg = lambda idx: idx - it0 + iwg(iwn) + 1

    # slices
    # entire vector
    pall = slice(ip0, ipn + 1)
    uall = slice(iu0, iun + 1)
    wall = slice(iw0, iwn + 1)
    tall = slice(it0, itn + 1)
    # interior points
    pint = slice(1, ncheb - 1)
    uint = slice(1, ncheb - 1)
    wint = slice(1, ncheb - 1)
    tint = slice(1, ncheb - 1)
    # entire vector with big matrix indexing
    pgall = slice(ipg(ip0), ipg(ipn + 1))
    ugall = slice(iug(iu0), iug(iun + 1))
    wgall = slice(iwg(iw0), iwg(iwn + 1))
    tgall = slice(itg(it0), itg(itn + 1))
    # interior points with big matrix indexing
    pgint = slice(ipg(1), ipg(ncheb - 1))
    ugint = slice(iug(1), iug(ncheb - 1))
    wgint = slice(iwg(1), iwg(ncheb - 1))
    tgint = slice(itg(1), itg(ncheb - 1))

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
        lmat[iwg(iwn), wgall] = -phi_bot * one[iwn, pall] + 2 * dz1[iwn, pall]

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T
    if heat_flux_top:
        # Neumann boundary condition
        lmat[itg(it0), tgall] = dz1[it0, tall]
    if heat_flux_bot:
        # Neumann boundary condition
        lmat[itg(itn), tgall] = dz1[itn, tall]
    lmat[tgint, tgall] = lapl[tint, tall]

    # need to take heat flux into account in T conductive
    if translation:
        lmat[tgint, wgall] = \
            np.diag(np.exp(wtrans * self.rad[wall]))[tint, wall]
        lmat[tgint, tgall] -= wtrans * dz1[tint, tall]
        if np.abs(wtrans)>1.e-3:
            lmat[tgint, wgall] *= wtrans/(2*np.sinh(wtrans/2))
        else:
            # use a limited development
            lmat[tgint, wgall] *= (1-wtrans**2/24)
    else:
        lmat[tgint, wgall] = one[tint, wall]

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
    u_mode = eigvecs[ugall]
    w_mode = eigvecs[wgall]
    t_mode = eigvecs[tgall]

    p_mode = self._insert_boundaries(p_mode, ip0, ipn)
    u_mode = self._insert_boundaries(u_mode, iu0, iun)
    w_mode = self._insert_boundaries(w_mode, iw0, iwn)
    t_mode = self._insert_boundaries(t_mode, it0, itn)

    return eigvals[iegv], (p_mode, u_mode, w_mode, t_mode)


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
    one = np.identity(ncheb)  # identity
    lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
    lapl = dr2 + 2 * np.dot(orad1, dr1) - lh2 * orad2  # laplacian

    # index min and max
    # poloidal
    ip0 = 0
    ipn = ncheb - 1
    # laplacian of poloidal
    iq0 = 0
    iqn = ncheb - 1
    # temperature
    it0 = 1
    itn = ncheb - 2

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
    pint = slice(1, ncheb - 1)
    qint = slice(1, ncheb - 1)
    tint = slice(1, ncheb - 1)
    # entire vector with big matrix indexing
    pgall = slice(ipg(ip0), ipg(ipn + 1))
    qgall = slice(iqg(iq0), iqg(iqn + 1))
    tgall = slice(itg(it0), itg(itn + 1))
    # interior points with big matrix indexing
    pgint = slice(ipg(1), ipg(ncheb - 1))
    qgint = slice(iqg(1), iqg(ncheb - 1))
    tgint = slice(itg(1), itg(ncheb - 1))

    lmat = np.zeros((itg(itn) + 1, itg(itn) + 1))
    rmat = np.zeros((itg(itn) + 1, itg(itn) + 1))

    # Poloidal potential equations
    # free-slip at top (BC on P)
    if ip0 == 0:
        lmat[ipg(ip0), pgall] = dr2[ip0, pall] + \
                (lh2 - 2) * orad2[ip0, pall]
    # laplacian(P) - Q = 0
    lmat[pgint, pgall] = lapl[pint, pall]
    lmat[pgint, qgall] = -one[qint, qall]
    # free-slip at bot (BC on P)
    if ipn == ncheb - 1:
        lmat[ipg(ipn), pgall] = dr2[ipn, pall] + \
                (lh2 - 2) * orad2[ipn, pall]

    # Q equations
    # normal stress continuity at top
    if iq0 == 0:
        lmat[iqg(iq0), pgall] = lh2 * (self.phys.phi_top *
            orad1[iq0, pall] - 2 * orad2[iq0, pall] +
            2 * np.dot(orad1, dr1)[iq0, pall])
        lmat[iqg(iq0), qgall] = -one[iq0, qall] - np.dot(rad, dr1)[iq0, qall]
    # laplacian(Q) - RaT/r = 0
    lmat[qgint, qgall] = lapl[qint, qall]
    lmat[qgint, tgall] = - ra_num * orad1[qint, tall]
    # normal stress continuity at bot
    if iqn == ncheb - 1:
        lmat[iqg(iqn), pgall] = lh2 * (-self.phys.phi_bot *
            orad1[iqn, pall] - 2 * orad2[iqn, pall] +
            2 * np.dot(orad1, dr1)[iqn, pall])
        lmat[iqg(iqn), qgall] = -one[iqn, qall] - np.dot(rad, dr1)[iqn, qall]

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T
    lmat[tgint, pgall] = lh2 * self.phys.gamma / (1 - self.phys.gamma) \
        * orad3[tint, pall]
    lmat[tgint, tgall] = lapl[tint, tall]
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

    return eigvals[iegv], (p_mode, up_mode, ur_mode, t_mode)


class LinearAnalyzer:

    """Perform linear analysis

    The studied problem is the one of Rayleigh-Benard convection with
    phase change at either or both boundaries.
    """

    def __init__(self, phys, ncheb=15):
        """Create a linear analyzer
        
        phys is the PhysicalProblem
        ncheb is the number of Chebyshev nodes
        """
        # get differentiation matrices
        self._ncheb = ncheb
        self._zcheb, self._ddm = dm.chebdif(self._ncheb, 2)
        self.phys = phys
        self.phys.bind_to(self)

    def _insert_boundaries(self, mode, im0, imn):
        """Insert zero at boundaries of mode if needed
        
        This need to be done when Dirichlet BCs are applied
        """
        if im0 == 1:
            mode = np.insert(mode, [0], [0])
        if imn == self._ncheb - 2:
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
