import dmsuite.dmsuite as dm
import numpy as np
import numpy.ma as ma
from scipy import linalg
from scipy.optimize import brentq
from numpy.linalg import solve

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
        name = []
        if self.spherical:
            name.append('sph')
            name.append(np.str(self.gamma).replace('.', '-'))
        else:
            name.append('cart')
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


def normalize(arr, norm=None):
    """Normalize complex array with element of higher modulus

    if norm is given, a first normalization with this norm is done
    """
    if norm is not None:
        arr = arr / norm
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax


def normalize_modes(modes, norm_mode=3, full_norm=True):
    """Normalize modes

    Since modes are eigenvectors, one can choose an arbitrary
    value to normalize them without any loss of information.

    The chose value is the component of modes[norm_mode] with
    the max modulus.
    All the other modes are then normalized by their maximum
    modulus component.
    This function return the set of modes after those two
    normalizations, and the set of normalization components.
    """
    ref_vector, norm = normalize(modes[norm_mode])
    normed_vectors = []
    norm_values = []
    for ivec in range(0, norm_mode):
        if full_norm:
            nvec, nval = normalize(modes[ivec], norm)
        else:
            nvec, nval = modes[ivec] / norm, 1
        normed_vectors.append(nvec)
        norm_values.append(nval)
    normed_vectors.append(ref_vector)
    norm_values.append(norm)
    for ivec in range(norm_mode + 1, len(modes)):
        if full_norm:
            nvec, nval = normalize(modes[ivec], norm)
        else:
            nvec, nval = modes[ivec] / norm, 1
        normed_vectors.append(nvec)
        norm_values.append(nval)
    return normed_vectors, norm_values


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
    one = np.identity(ncheb)  # identity
    lh2 = l_harm * (l_harm + 1)  # horizontal laplacian
    lapl = dr2 + 2 * np.dot(orad1, dr1) - lh2 * orad2  # laplacian
    phi_top = self.phys.phi_top
    phi_bot = self.phys.phi_bot
    freeslip_top = self.phys.freeslip_top
    freeslip_bot = self.phys.freeslip_bot
    heat_flux_top = self.phys.heat_flux_top
    heat_flux_bot = self.phys.heat_flux_bot
    translation = self.phys.ref_state_translation

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
        pint = slice(1, ncheb - 1)
        uint = slice(1, ncheb - 1)
        wint = slice(1, ncheb - 1)
        tint = slice(1, ncheb - 1)
        slint = (pint, uint, wint, tint)
        # entire vector with big matrix indexing
        pgall = slice(ipg(ip0), ipg(ipn + 1))
        ugall = slice(iug(iu0), iug(iun + 1))
        wgall = slice(iwg(iw0), iwg(iwn + 1))
        tgall = slice(itg(it0), itg(itn + 1))
        slgall = (pgall, ugall, wgall, tgall)
        # interior points with big matrix indexing
        pgint = slice(ipg(1), ipg(ncheb - 1))
        ugint = slice(iug(1), iug(ncheb - 1))
        wgint = slice(iwg(1), iwg(ncheb - 1))
        tgint = slice(itg(1), itg(ncheb - 1))
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

    def eigval(self, harm, ra_num):
        """Generic eigval function"""
        if self.phys.spherical:
            return eigval_spherical(self, harm, ra_num) # not yet treated beyond that point
        else:
            return eigval_cartesian(self, harm, ra_num)

    def integz(self, prof):
        """Integral on the z interval with Chebyshev weight"""
        intz = np.sum(prof)
        intz = intz - (prof[0] + prof[-1])/2
        intz *=np.pi / self._ncheb
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
        # self.name = ana.phys.name()
        ra_c, harm_c = ana.critical_ra()
        _, mode_c, mats_c = self.eigval(harm_c, ra_c)
        mode_c, _ = normalize_modes(mode_c, full_norm=False)
        p_c, u_c, w_c, t_c = mode_c
        lmat_c, _ = mats_c
        dt_c = np.dot(self.dr1, t_c)
        # theta part of the non-linear term N(Xc, Xc)
        # part constant in x
        nxcxc0 = np.zeros((itg(itn) + 1))
        nxcxc0[tgint] = 2 * (w_c * dt_c + harm_c * np.imag(u_c) * t_c)[tint]
        # part in 2 k x
        nxcxc2 = np.zeros((itg(itn) + 1))
        nxcxc2[tgint] = 2 * (w_c * dt_c - harm_c * np.imag(u_c) * t_c)[tint]
        # Solve Lc * X2 = NXcXc to get X2
        mode20 = solve(lmat_c, nxcxc0)

        p20 = mode20[pgall]
        u20 = mode20[ugall]
        w20 = mode20[wgall]
        t20 = mode20[tgall]

        p20 = self._insert_boundaries(p20, ip0, ipn)
        u20 = self._insert_boundaries(u20, iu0, iun)
        w20 = self._insert_boundaries(w20, iw0, iwn)
        t20 = self._insert_boundaries(t20, it0, itn)
        dt20 = np.dot(self.dr1, t20)

        mode22 = solve(lmat_c, nxcxc2)

        p22 = mode22[pgall]
        u22 = mode22[ugall]
        w22 = mode22[wgall]
        t22 = mode22[tgall]

        p22 = self._insert_boundaries(p22, ip0, ipn)
        u22 = self._insert_boundaries(u22, iu0, iun)
        w22 = self._insert_boundaries(w22, iw0, iwn)
        t22 = self._insert_boundaries(t22, it0, itn)
        dt22 = np.dot(self.dr1, t22)

        # denominator in Ra2
        xcmxc =  self.integz(w_c * t_c)

        # numerator in Ra2
        xcnx2xc = 2 * harm_c * self.integz(t_c**2 * np.imag(u22))
        xcnx2xc += 2 * self.integz(t_c * dt_c * (np.real(w22) + w20))
        xcnxcx2 = -4 * harm_c * 1j * self.integz(t_c * u_c * np.real(t22))
        xcnxcx2 += 2 * self.integz(t_c * w_c * (np.real(dt22) + dt20))

        # Ra2
        ra2 = (xcnx2xc + xcnxcx2)/xcmxc

        return harm_c, (ra_c, ra2), mode_c, (p20, u20, w20, t20), (p22, u22, w22, t22)
