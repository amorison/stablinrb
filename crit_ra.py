#!/usr/bin/env python3
"""
Finds critical Rayleigh number.

Plots the Critical Ra as function of wave-number and finds the minimum.
Based on the matlab code provided by Thierry Alboussiere.
Can do both the no-slip and free-slip BCs, applying to both boundaries.
Also treats the phase change boundary conditions.
"""
import math
import numpy as np
import numpy.ma as ma
from scipy import linalg
from scipy import integrate
import dmsuite.dmsuite as dm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from scipy.special import sph_harm

# for contour
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, \
    MaxNLocator, DictFormatter

# -------------------------------------------------------------------

# Spherical geometry
SPHERICAL = True
GAMMA = 0.5

# Free slip (rigid otherwise)
# (unused if PHI_* is not None)
FREESLIP_TOP = True
FREESLIP_BOT = True

# Phase change number
# (precedence on FREESLIP_*)
# set to None if no phase change
PHI_TOP = 3.e-2
PHI_BOT = 3.e-2

# Whether to plot the theoretical profiles for translation mode
PLOT_THEORY = False

# Number of Chebyshev points
NCHEB = 15

# Font and markers size
FTSZ = 14
MSIZE = 3

# Explore phi space
COMPUTE_PHASECHANGE = False

# -------------------------------------------------------------------

if FREESLIP_TOP or (PHI_TOP is not None):
    BCSU = [[0, 1, 0]]
else:
    BCSU = [[1, 0, 0]]

if FREESLIP_BOT or (PHI_BOT is not None):
    BCSU.append([0, 1, 0])
else:
    BCSU.append([1, 0, 0])

BCSU = np.array(BCSU)

title = ['sph'] if SPHERICAL else ['cart']
if PHI_TOP is not None:
    title.append('Ptop')
else:
    title.append(
        'free' if FREESLIP_TOP else 'rigid')
if PHI_BOT is not None:
    title.append('Pbot')
else:
    title.append(
        'free' if FREESLIP_BOT else 'rigid')
title = '_'.join(title)

def ndprint(a, format_string='{0:.2f}'):
    """pretty print array for debugging"""
    for ili in range(a.shape[0]):
        for icol in range(a.shape[1]):
            if icol == a.shape[1]-1:
                print(format_string.format(a[ili, icol]))
            else:
                print(format_string.format(a[ili, icol]), ' ', end='')

def func(wtra, eps):
    """function whose roots are the translation velocity"""
    fff = wtra**2*np.sinh(wtra/2)-6*(1+eps)*(wtra*np.cosh(wtra/2)-2*np.sinh(wtra/2))
    return fff

def wtran(eps):
    """translation velocity as function of the reduced Rayleigh number"""
    if eps <= 0:
        wtr = 0
        wtrs = 0
        wtrl = 0
    else:
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

def eigval_spherical(l_harm, ranum, ncheb, gamma,
                     phitop=None, phibot=None,
                     output_eigvec=False):
    """
    Eigenvalue for spherical geometry

    l_harm: spherical harmonics degree
    ranum: Rayleigh number
    ncheb: number of Chebyshev collocation points
    gamma: r_bot/r_top
    phitop: phase change number at top, None if no phase change
    phibot: phase change number at bot, None if no phase change
    output_eigvec: whether to return eigenvector
    """
    # indices for each field
    # (Dirichlet BC for temperature)

    if not(0<gamma<1):
        raise ValueError

    if phitop is None or phibot is None:
        raise ValueError

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

    # get differentiation matrices
    zcheb, ddm = dm.chebdif(ncheb, 2)

    # physical space is gamma<r<1
    # whereas Chebyshev polynomials are -1<z<1
    shrink_geom = (1 - gamma) / 2
    rad = zcheb * shrink_geom + (1 + gamma) / 2
    rad1 = 1 / rad
    rad2 = 1 / rad**2
    rad3 = 1 / rad**3
    rad = np.diag(rad)
    rad1 = np.diag(rad1)
    rad2 = np.diag(rad2)
    rad3 = np.diag(rad3)
    one = np.identity(ncheb)  # identity
    dr1 = ddm[0,:,:] / shrink_geom  # first r-derivative
    dr2 = ddm[1,:,:] / shrink_geom**2  # second r-derivative
    lh2 = l_harm * (l_harm + 1)
    lapl = dr2 + 2 * np.dot(rad1, dr1) - lh2 * rad2

    lmat = np.zeros((itg(itn) + 1, itg(itn) + 1))
    rmat = np.zeros((itg(itn) + 1, itg(itn) + 1))

    # Poloidal potential equations
    # free-slip at top (BC on P)
    if ip0 == 0:
        lmat[ipg(ip0), pgall] = dr2[ip0, pall] + \
                (lh2 - 2) * rad2[ip0, pall]
    # laplacian(P) - Q = 0
    lmat[pgint, pgall] = lapl[pint, pall]
    lmat[pgint, qgall] = -one[qint, qall]
    # free-slip at bot (BC on P)
    if ipn == ncheb - 1:
        lmat[ipg(ipn), pgall] = dr2[ipn, pall] + \
                (lh2 - 2) * rad2[ipn, pall]

    # Q equations
    # normal stress continuity at top
    if iq0 == 0:
        lmat[iqg(iq0), pgall] = lh2 * (phitop * rad1[iq0, pall] -
                2 * rad2[iq0, pall] + 2 * np.dot(rad1, dr1)[iq0, pall])
        lmat[iqg(iq0), qgall] = -one[iq0, qall] - np.dot(rad, dr1)[iq0, qall]
    # laplacian Q - RaT/r = 0
    lmat[qgint, qgall] = lapl[qint, qall]
    lmat[qgint, tgall] = - ranum * rad1[qint, tall]
    # normal stress continuity at bot
    if iqn == ncheb - 1:
        lmat[iqg(iqn), pgall] = lh2 * (-phibot * rad1[iqn, pall] -
                2 * rad2[iqn, pall] + 2 * np.dot(rad1, dr1)[iqn, pall])
        lmat[iqg(iqn), qgall] = -one[iqn, qall] - np.dot(rad, dr1)[iqn, qall]

    # T equations
    # laplacian(T) - u.grad(T_conductive) = sigma T
    lmat[tgint, pgall] = lh2 * gamma / (1 - gamma) * rad3[tint, pall]
    lmat[tgint, tgall] = lapl[tint, tall]
    rmat[tgint, tgall] = one[tint, tall]

    # Find the eigenvalues
    if output_eigvec:
        eigvals, eigvecs = linalg.eig(lmat, rmat, right=True)
    else:
        eigvals = linalg.eig(lmat, rmat, right=False)
    index = np.argsort(ma.masked_invalid(eigvals))
    eigvals = ma.masked_invalid(eigvals[index])
    iegv = np.argmax(np.real(eigvals))
    if output_eigvec:
        eigvecs = eigvecs[:, index]
        eigvecs = eigvecs[:, iegv]
        p_mode = eigvecs[pgall]
        q_mode = eigvecs[qgall]
        t_mode = eigvecs[tgall]
        if ip0 == 1:
            p_mode = np.insert(p_mode, [0], [0])
        if ipn == ncheb - 2:
            p_mode = np.append(p_mode, 0)
        if iq0 == 1:
            q_mode = np.insert(q_mode, [0], [0])
        if iqn == ncheb - 2:
            q_mode = np.append(q_mode, 0)
        if it0 == 1:
            t_mode = np.insert(t_mode, [0], [0])
        if itn == ncheb - 2:
            t_mode = np.append(t_mode, 0)

        rad = np.diag(rad)
        ur_mode = l_harm * (l_harm + 1) * p_mode / rad
        up_mode = 1j * l_harm * (np.dot(dr1, p_mode) + p_mode / rad)

        return eigvals[iegv], p_mode, t_mode, ur_mode, up_mode, rad
    else:
        return eigvals[iegv]

def eigval_cartesian(wnk, ranum, ncheb,
                     bcsu=np.array([[1, 0, 0], [1, 0, 0]], float),
                     bcsw=np.array([[1, 0, 0], [1, 0, 0]], float),
                     bcst=np.array([[1, 0, 0], [1, 0, 0]], float),
                     phase_change_top=False,
                     phase_change_bot=False,
                     phitop = 100,
                     phibot = 100,
                     penalty = 0.,
                     output_eigvec=False,
                     translation=False,
                     mode=0):
    """
    Eigenvalue for given wavenumber and Rayleigh number with general BCs

    The boundary conditions are
    a_1 f(1) + b_1 f'(1)  = c_1
    a_N f(-1) + b_N f'(-1) = c_N
    for each of the following variables: u (horizontal velocity),
    w (vertical velocity), t (temperature)
    The values are passed as part of the **kwargs, with keywords
    bcsX for variable X.
    Default values are the Dirichlet conditions for all variables at both
    boundaries, ie: bcsX = np.array([[1, 0, 0], [1, 0, 0]], float)

    INPUT
    wnk = horizontal wavenumber of the perturbation
    ranum = Rayleigh number
    ncheb = number of Chebyshev points in vertical direction
    bcsu=array of boundary conditions for u (default Dirichlet)
    bcsw=array of boundary conditions for w (default Dirichlet)
    bcst=array of boundary conditions for t (default Dirichlet)
    plot_eigvec=True|False (default) whether to return the (right) eigenvector
    translation=True|False (default) controls whether the base state is translation

    OUTPUT
    eigv = eigenvalue with the largest real part (fastest growing if positive).
    eigvec = corresponding eigen vector
    """
    # setup indices for each field depending on BCs
    # local and global
    iu0 = 0
    iun = ncheb
    iw0 = 0
    iwn = ncheb
    it0 = 0
    itn = ncheb

    # CHECK WHETHER TOP IS REALLY TOP
    if not phase_change_top:
        if bcsu[0, 1] == 0:
            # Dirichlet at z=-1/2
            iu0 = 1
        if bcsw[0, 1] == 0:
            # Dirichlet at z=-1/2
            iw0 = 1

    if not phase_change_bot:
        if bcsu[1, 1] == 0:
            # Dirichlet ar z=1/2
            iun = ncheb-1
        if bcsw[1, 1] == 0:
            # Dirichlet ar z=1/2
            iwn = ncheb-1

    ju0 = ncheb
    jw0 = ju0+iun-iu0
    jun = jw0-1
    jwn = jw0+iwn-iw0-1

    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        it0 = 1
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        itn = ncheb-1
    # For pressure. No BCs but side values needed or removed
    # depending on the BCs for W. number of lines need to be
    # the same as that of d2w and depends on bcsw.
    
    if translation:
        rtr = 12*(phitop+phibot)
        wtrans = wtran((ranum-rtr)/rtr)[0]
    # All z derivatives must be multiplied by 2 to scale [-1, 1] to [-0.5, 0.5]
    if output_eigvec or translation:
        xxt, ddm = dm.chebdif(ncheb, 2)
    else:
        ddm = dm.chebdif(ncheb, 2)[1]
    if phase_change_top or phase_change_bot:
        dd1 = 2.*ddm[0, :, :]
        dd2 = 4.*ddm[1, :, :]
        # take care of Dirichlet BCs naturally.
        d2u = dd2[iu0:iun, iu0:iun]
        d2w = dd2[iw0:iwn, iw0:iwn]
    else:
        # compute differentiation matrices
        # For horizontal velocity
        d2u = 4.*dm.cheb2bc(ncheb, bcsu)[1]
        # For vertical velocity
        d2w = 4.*dm.cheb2bc(ncheb, bcsw)[1]
        
    # For temperature. d1t used for the stability of the translation mode
    ddmt = dm.cheb2bc(ncheb, bcst)[1:3]
    d2t = 4*ddmt[0]
    d1t = 2*ddmt[1]

    d1p = 2.*ddm[0, iw0:iwn, :]
    d1w = 2.*ddm[0, :, iw0:iwn]

    # identity
    ieye = np.eye(ncheb, ncheb)

    # Prandtl number. Could be an argument of the functiojn if needed.
    pra = 1.
    # square of the wave number
    wn2 = wnk**2.
    # construction of the lhs matrix. Submatrices' dimensions must match
    # first line: p
    app = np.zeros((ncheb, ncheb))
    apu = 1j*wnk*ieye[:, iu0:iun]
    apw = d1w
    apt = np.zeros((ncheb, itn-it0))

    # second line: u
    aup = -1j*wnk*ieye[iu0:iun, :]
    auu = d2u-wn2*ieye[iu0:iun, iu0:iun]
    auw = np.zeros((iun-iu0, iwn-iw0))+0.*1j
    aut = np.zeros((iun-iu0, itn-it0))
    # special boundary conditions
    if phase_change_top:
        # zero shear stress
        aup[0, :] = 0.
        auu[0, :] = dd1[0, iu0:iun]
        auw[0, 0] = 1j*wnk
        if not phase_change_bot and bcsu[1, 1] !=0:
            # need to deal with the other side in case of Robin at z=-1/2
            if bcsu[1, 2] != 0:
                print('Warning: bcsu[1, 2] not taken into account', bcsu[1, 2])
            aup[-1, :] = 0.
            auu[-1, :]=bcsu[1, 1]*dd1[-1, iu0:iun]
            auu[-1, -1] += bcsu[1, 0]
    if phase_change_bot:
        # zero shear stress
        aup[-1, :] = 0.
        auu[-1, :] = dd1[-1, iu0:iun]
        auw[-1, -1] = 1j*wnk
        if not phase_change_top and bcsu[0, 1] != 0:
            # need to deal with the other side in case of Robin at z=1/2
            if bcsu[0, 2] != 0:
                print('Warning: bcsu[0, 2] not taken into account', bcsu[0, 2])
            aup[0, :] = 0.
            auu[0, :] = bcsu[0, 1]*dd1[0, iu0:iun]
            auu[0, 0] += bcsu[0, 0]
    # third line: w
    awp = -d1p
    awu = np.zeros((iwn-iw0, iun-iu0))
    aww = d2w-wn2*ieye[iw0:iwn, iw0:iwn]
    awt = ranum*ieye[iw0:iwn, it0:itn]
    if phase_change_top:
        awp[0, :] = 0.
        awt[0, :] = 0.
        if phitop < 1.:
            awp[0, 0] = -1.
            aww[0, :] = 2.*dd1[0, iw0:iwn]
            aww[0, 0] += phitop
        else:
            awp[0, 0] = -1./phitop
            aww[0, :] = 2./phitop*dd1[0, iw0:iwn]
            aww[0, 0] += 1.
        if not phase_change_bot and bcsw[1, 1] !=0:
            # need to deal with the other side in case of Robin at z=-1/2
            if bcsw[1, 2] != 0:
                print('Warning: bcsw[1, 2] not taken into account', bcsw[1, 2])
            awp[-1, :] = 0.
            aww[-1, :] = bcsw[1, 1]*dd1[-1, iw0:iwn]
            aww[-1, -1] += bcsw[1, 0]
            awt[-1, :] = 0.
    if phase_change_bot:
        awp[-1, :] = 0.
        awt[-1, :] = 0.
        if phibot < 1.:
            awp[-1, -1] = -1.
            aww[-1, :] = 2.*dd1[-1, iw0:iwn]
            aww[-1, -1] -= phibot
        else:
            awp[-1, -1] = -1./phibot
            aww[-1, :] = 2./phibot*dd1[-1, iw0:iwn]
            aww[-1, -1] -= 1.
        if not phase_change_top and bcsw[0, 1] != 0:
            # need to deal with the other side in case of Robin at z=1/2
            if bcsw[0, 2] != 0:
                print('Warning: bcsw[0, 2] not taken into account', bcsw[0, 2])
            awp[0, :] = 0.
            aww[0, :] = bcsw[0, 1]*dd1[0, iw0:iwn]
            aww[0, 0] += bcsw[0, 0]
            awt[0, :] = 0.

    # fourth line: t
    atp = np.zeros((itn-it0, ncheb))
    atu = np.zeros((itn-it0, iun-iu0))
    # Assume here the basic state to be linear of z
    att = d2t-wn2*ieye[it0:itn, it0:itn]
    if translation:
        att -= wtrans*d1t
        atw = np.diag(np.exp(wtrans*xxt[iw0:iwn]))[it0:itn, iw0:iwn]
        if np.abs(wtrans)>1.e-3:
            atw *= wtrans/(2*np.sinh(wtrans/2))
        else:
            # use a limited development
            atw *= (1-wtrans**2/24)
    else:
        atw = ieye[it0:itn, iw0:iwn]
    # assemble the big matrix
    abig = np.concatenate((np.concatenate((app, apu, apw, apt), axis=1),
                           np.concatenate((aup, auu, auw, aut), axis=1),
                           np.concatenate((awp, awu, aww, awt), axis=1),
                           np.concatenate((atp, atu, atw, att), axis=1),
                          ))

    # construction of the rhs matrix: identity almost everywhere
    bbig = np.eye(abig.shape[0])
    # penalty to enforce pressure (mass conservation)
    bbig[range(d1p.shape[1]), range(d1p.shape[1])] = penalty
    # penalty to enforce boundary conditions
    if phase_change_top:
        # u
        bbig[ju0, ju0] = penalty
        # w
        bbig[jw0, jw0] = penalty
    else:
        if iu0 == 0:
            # u
            bbig[ju0, ju0] = penalty
        if iw0 == 0:
            # w
            bbig[jw0, jw0] = penalty
    if phase_change_bot:
        # u
        bbig[jun, jun] = penalty
        # w
        bbig[jwn, jwn] = penalty
    else:
        if iun == ncheb:
            # u
            bbig[jun, jun] = penalty
        if iwn == ncheb:
            # w
            bbig[jwn, jwn] = penalty
    if translation:
        # penalty to enforce infinite Prandtl number
        bbig[ju0:jwn, ju0:jwn] = penalty
    # Find the eigenvalues
    if output_eigvec:
        egv, rvec = linalg.eig(abig, bbig, right=True)
    else:
        egv = linalg.eig(abig, bbig, right=False)
    #print(egv)
    # egv = np.sort(egv)
    # print(egv)
    if penalty == 0.:
        index = np.argsort(ma.masked_invalid(egv))
        egv2 = ma.masked_invalid(egv[index])
    else:
        index = np.argsort(ma.masked_where(np.real(egv)>0.1/penalty, egv))
        egv2 = ma.masked_where(np.real(egv[index])>0.1/penalty, egv[index])
    if output_eigvec:
        rvec2 = rvec[:, index]
    leig = np.argmax(np.real(egv2))-mode
    if output_eigvec:
        return egv2[leig], rvec2[:, leig], xxt
    else:
        return egv2[leig]


def search_ra(wnk, ray, ncheb, **kwargs):
    """find rayleigh number ray which gives neutral stability"""
    ray0 = ray/math.sqrt(1.2)
    ray1 = ray*math.sqrt(1.2)
    la0 = np.real(eigval_cartesian(wnk, ray0, ncheb, **kwargs))
    la1 = np.real(eigval_cartesian(wnk, ray1, ncheb, **kwargs))

    while la0 > 0. or la1 < 0.:
        if la0 > 0.:
            ray1 = ray0
            ray0 = ray0/2.
        if la1 < 0.:
            ray0 = ray1
            ray1 = 2.*ray1
        la0 = np.real(eigval_cartesian(wnk, ray0, ncheb, **kwargs))
        la1 = np.real(eigval_cartesian(wnk, ray1, ncheb, **kwargs))
    # while np.abs(lam) > 1.e-7:
    while la1-la0 > 1.e-3:
        raym = (ray0+ray1)/2.
        lam = np.real(eigval_cartesian(wnk, raym, ncheb, **kwargs))
        if lam < 0.:
            la0 = lam
            ray0 = raym
        else:
            la1 = lam
            ray1 = raym
        # raym = (ray0+ray1)/2.
        # lam = np.real(eigval_cartesian(wnk, raym, ncheb, **kwargs))
    return (ray0*la1-ray1*la0)/(la1-la0)

def ra_ks(rag, wng, ncheb, **kwargs):
    """finds the minimum in the Ra-wn curve"""
    # find 3 values of Ra for 3 different wave numbers
    eps = [0.1, 0.01]
    wns = np.linspace(wng*(1-eps[0]), wng*(1+2*eps[0]), 3)
    ray = [search_ra(kkx, rag, ncheb, **kwargs) for kkx in wns]

    # fit a degree 2 polynomial
    pol = np.polyfit(wns, ray, 2)

    # minimum value
    exitloop = False
    kmin = -0.5*pol[1]/pol[0]
    for i, err in enumerate([0.03, 1.e-3]):
        while np.abs(kmin-wns[1]) > err*kmin and not exitloop:
            wns = np.linspace(kmin*(1-eps[i]), kmin*(1+eps[i]), 3)
            ray = [search_ra(kkx, rag, ncheb, **kwargs) for kkx in wns]
            pol = np.polyfit(wns, ray, 2)
            kmin = -0.5*pol[1]/pol[0]
            # if kmin <= 1.e-3:
                # exitloop = True
                # kmin = 1.e-3
                # ray[1] = search_ra(kmin, rag, ncheb, **kwargs)
                # not able to properly converge anymore
            rag = ray[1]

    return rag, kmin

def plot_mode(wnk, ranum, ncheb, title,
                   npoints=100, plotfig=True, **kwargs):
    """
    Plots the fastest growing mode for wavenumber wnk and ranum

    INPUT
    wnk : wavenumber
    ranum : Rayleigh number
    ncheb : number of Chebyshev points
    title : string to use in the pdf file name
    npoints : number of interpolation points for the plot
                  otherwise just used to compute mode caracteristics
    plotfig : whether to actually plot and save the figure
    **kwargs : passed on to eigval_cartesian
    """
    # egv, eigvec, zzr = eigval_cartesian(wnk, ranum, ncheb, bcsu=bcsu,
    #                           bcsw=bcsw, bcst=bcst, output_eigvec=output_eigvec)
    if 'output_eigvec' not in kwargs:
        kwargs['output_eigvec'] = True

    if 'phase_change_top' in kwargs:
        phase_change_top = kwargs['phase_change_top']
    else:
        phase_change_top = False

    if 'phase_change_bot' in kwargs:
        phase_change_bot = kwargs['phase_change_bot']
    else:
        phase_change_bot = False

    bcsu=np.array([[1, 0, 0], [1, 0, 0]], float)
    bcsw=np.array([[1, 0, 0], [1, 0, 0]], float)
    bcst=np.array([[1, 0, 0], [1, 0, 0]], float)

    if kwargs != {}:
        if 'bcsu' in kwargs:
            bcsu = kwargs['bcsu']
        if 'bcsw' in kwargs:
            bcsw = kwargs['bcsw']
        if 'bcst' in kwargs:
            bcst = kwargs['bcst']
        if 'output_rakx' in kwargs:
            kwargs.pop('output_rakx', 0)
        if 'output_eigvec' not in kwargs:
            kwargs['output_eigvec'] = True
    else:
        kwargs = {}
        kwargs['output_eigvec'] = True

    egv, eigvec, zzr = eigval_cartesian(wnk, ranum, ncheb, **kwargs)
    print('Eigenvalue = ', egv)

    # setup indices for each field depending on BCs
    # local and global
    iu0 = 0
    iun = ncheb
    iw0 = 0
    iwn = ncheb
    it0 = 0
    itn = ncheb

    if not phase_change_top:
        if bcsu[0, 1] == 0:
            # Dirichlet at z=-1/2
            iu0 = 1
        if bcsw[0, 1] == 0:
            # Dirichlet at z=-1/2
            iw0 = 1

    if not phase_change_bot:
        if bcsu[1, 1] == 0:
            # Dirichlet ar z=1/2
            iun = ncheb-1
        if bcsw[1, 1] == 0:
            # Dirichlet ar z=1/2
            iwn = ncheb-1

    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        it0 = 1
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        itn = ncheb-1

    ju0 = ncheb
    jw0 = ju0+iun-iu0
    jt0 = jw0+iwn-iw0
    jun = jw0-1
    jwn = jt0-1
    jtn = jwn+itn-it0
            
    # now split the eigenvector into the different fields
    # do not normalize now
    pmod = eigvec[0:ncheb] #/np.max(np.abs(eigvec[0:ncheb]))
    umod = eigvec[ju0:jun+1] #/np.max(np.abs(eigvec[ju0:jun+1]))
    # add boundary values in Dirichlet case
    if bcsu[0, 1] == 0 and iu0 != 0:
        # Dirichlet at z=-1/2
        umod = np.insert(umod, 0, bcsu[0, 2])
    if bcsu[1, 1] == 0 and iun != ncheb:
        # Dirichlet ar z=1/2
        umod = np.append(umod, bcsu[1, 2])

    wmod = eigvec[jw0:jwn+1] #/np.max(np.abs(eigvec[jw0:jwn+1]))
    if bcsw[0, 1] == 0 and iw0 != 0:
        # Dirichlet at z=-1/2
        wmod = np.insert(wmod, [0], [bcsw[0, 2]])
    if bcsw[1, 1] == 0 and iwn != ncheb:
        # Dirichlet ar z=1/2
        wmod = np.append(wmod, bcsw[1, 2])
    tmod = eigvec[jt0:jtn+1] #/np.max(np.abs(eigvec[jt0:jtn+1]))
    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        tmod = np.insert(tmod, 0, bcst[0, 2])
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        tmod = np.append(tmod, bcst[1, 2])

    # use max of T to normalize all profiles
    tmax = tmod[np.argmax(np.abs(tmod))]
    tmod = tmod/tmax
    wmod = wmod/tmax
    wmax = wmod[np.argmax(np.abs(wmod))]
    umod = umod/tmax
    umax = umod[np.argmax(np.abs(umod))]
    pmod = pmod/tmax
    pmax = pmod[np.argmax(np.abs(pmod))]
    # Use the max of w to normalize all parts
    # wmax = wmod[np.argmax(np.abs(wmod))]
    # wmod = wmod/wmax
    # umod = umod/wmax
    # umax = umod[np.argmax(np.abs(umod))]
    # pmod = pmod/wmax
    # pmax = pmod[np.argmax(np.abs(pmod))]
    # tmod = tmod/wmax
    # tmax = tmod[np.argmax(np.abs(tmod))]

    if plotfig:
        # define the z values on which to interpolate modes
        # between -1 and 1 (Chebyshev) then rescaled to -1./2 to 1/2
        zpl = np.linspace(-1, 1, npoints)
        # interpolate
        upl = dm.chebint(umod, zpl)
        wpl = dm.chebint(wmod, zpl)
        tpl = dm.chebint(tmod, zpl)
        ppl = dm.chebint(pmod, zpl)

        # plot the mode profiles
        fig, axe = plt.subplots(1, 4, sharey=True)
        plt.setp(axe, xlim=[-1.1, 1.1], ylim=[-0.5, 0.5],
                xticks=[-1, -0.5, 0., 0.5, 1])
        # U
        if PLOT_THEORY:
            axe[0].plot(-zpl, zpl/2)
        else:            
            axe[0].plot(upl/(1j*np.abs(umax)), zpl/2)
        axe[0].plot(umod/(1j*np.abs(umax)), zzr/2, "o", label=r'$U$')
        axe[0].set_ylabel(r'$z$', fontsize=FTSZ)
        axe[0].set_xlabel(r'$U/(\mathrm{i} %.2e)$' %(np.abs(umax)), fontsize=FTSZ)
        # axe[1].plot(wpl, zpl/2)
        # axe[1].plot(wmod, zzr/2, "o", label=r'$W$')
        # axe[1].set_xlabel(r'$W$', fontsize=FTSZ)
        #W
        if PLOT_THEORY:
            axe[1].plot(np.ones(zpl.shape), zpl/2)
        else:
            axe[1].plot(wpl/np.abs(wmax), zpl/2)
        axe[1].plot(wmod/np.abs(wmax), zzr/2, "o", label=r'$W$')
        axe[1].set_xlabel(r'$W/(%.3f)$' %(wmax), fontsize=FTSZ)
        # axe[2].plot(tpl/np.abs(tmax), zpl/2)
        # axe[2].plot(tmod/np.abs(tmax), zzr/2, "o", label=r'$\theta$')
        # axe[2].set_xlabel(r'$\Theta/(%.3f)$' %(tmax), fontsize=FTSZ)
        # Theta
        if PLOT_THEORY:
            axe[2].plot(1-4*(zpl/2)**2, zpl/2)
        else:
            axe[2].plot(tpl, zpl/2)
        axe[2].plot(tmod, zzr/2, "o", label=r'$\theta$')
        axe[2].set_xlabel(r'$\Theta$', fontsize=FTSZ)
        # P
        if PLOT_THEORY:
            axe[3].plot(zpl*2/(13*np.sqrt(13))*(39-64*(zpl/2)**2), zpl/2)
        else:
            axe[3].plot(ppl/np.abs(pmax), zpl/2)
        axe[3].plot(pmod/np.abs(pmax), zzr/2, "o", label=r'$P$')
        axe[3].set_xlabel(r'$P/(%.2e)$' %(pmax), fontsize=FTSZ)
        if PLOT_THEORY:
            filename = '_'.join((title, 'mode_prof', 'theory.pdf'))
        else:
            filename = '_'.join((title, 'mode_prof.pdf'))
        plt.savefig(filename, format='PDF')
        plt.close(fig)

        # now plot the modes in 2D
        # make a version with the total temperature
        xvar = np.linspace(0, 2*np.pi/wnk, npoints)
        xgr, zgr = np.meshgrid(xvar, zpl)
        zgr = 0.5*zgr
        # temperature
        modx = np.exp(1j*wnk*xvar)
        t2d1, t2d2 = np.meshgrid(modx, tpl)
        t2d = np.real(t2d1*t2d2)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        if wnk>0.2:
            plt.figure(figsize=(6*np.pi/wnk, 3), dpi=300)
        else:
            plt.figure()
        axe = fig.add_subplot(111)
        # if wnk>0.2:
        # plt.axis('equal')
        # plt.setp(axe, xlim=[0, 2*np.pi/wnk], ylim=[-0.5, 0.5],
                 # xticks=np.linspace(0, np.floor(2*np.pi/wnk), 5))
        surf=plt.pcolormesh(xgr, zgr, t2d, cmap='RdBu_r', linewidth=0,)
        plt.axis([xgr.min(), xgr.max(), zgr.min(), zgr.max()])
        # stream function
        u2d1, u2d2 = np.meshgrid(modx, upl)
        u2d = np.real(u2d1*u2d2)
        w2d1, w2d2 = np.meshgrid(modx, wpl)
        w2d = np.real(w2d1*w2d2)
        speed = np.sqrt(u2d**2+w2d**2)
        lw = 2*speed/speed.max()
        plt.streamplot(xgr, zgr, u2d, w2d, linewidth=lw, density=0.7)
        plt.xlabel(r'$x$', fontsize=FTSZ)
        plt.ylabel(r'$y$', fontsize=FTSZ)
        cbar = plt.colorbar(surf)
        # save image
        filename = '_'.join((title, 'mode.pdf'))
        plt.savefig(filename, format='PDF', bbox_inches='tight')
    return pmax, umax, wmax, tmax

def findplot_rakx(ncheb, title, **kwargs):
    """
    Finds the minimum and plots Ra(kx)

    Inputs
    ----------
    ncheb  = number of Chebyshev points in the calculation
    title  = string variable to use in figure name
    **kwargs: most are just to be passed on to eigval_cartesian
    output_eigvec (True|False) controls whether to plot the eigenvector
    of the first ustable mode. Default is False
    """
    if kwargs != {}:
        if 'output_rakx' in kwargs:
            output_rakx = kwargs['output_rakx']
            kwargs.pop('output_rakx', 0)
        else:
            output_rakx = True
        # alternate kwargs to not output eigenvector
        kwargs2 = kwargs.copy()
        if 'plotfig' in kwargs:
            kwargs2.pop('plotfig', 0)
        if 'output_eigvec' in kwargs:
            plot_eigvec = kwargs['output_eigvec']
            kwargs2['output_eigvec'] = False
    else:
        plot_eigvec = False
        kwargs2 = {}
        output_rakx = True
        kwargs2['output_eigvec'] = False

    ramin, kxmin = ra_ks(600, 2, ncheb, **kwargs2)
    # print(title+': Ra=', ramin, 'kx=', kxmin)

    if output_rakx:
        # plot Ra as function of wavenumber
        wnum = np.linspace(kxmin/2, kxmin*1.5, 50)
        rayl = [search_ra(wnum[0], ramin, ncheb, **kwargs2)]
        for i, kk in enumerate(wnum[1:]):
            ra2 = search_ra(kk, rayl[i], ncheb, **kwargs2)
            rayl = np.append(rayl, ra2)

        fig = plt.figure()
        plt.plot(wnum, rayl, linewidth=2)
        if ramin < 1:
            plt.plot(kxmin, ramin, 'o', label=r'$Ra_{min}=%.2e ; k=%.2e$' %(ramin, kxmin))
        else:
            plt.plot(kxmin, ramin, 'o', label=r'$Ra_{min}=%.2f ; k=%.2f$' %(ramin, kxmin))
        plt.xlabel('Wavenumber', fontsize=FTSZ)
        plt.ylabel('Rayleigh number', fontsize=FTSZ)
        plt.xticks(fontsize=FTSZ)
        plt.yticks(fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        filename = '_'.join((title, 'Ra_kx.pdf'))
        plt.savefig(filename, format='PDF')
        plt.close(fig)

    if plot_eigvec:
        pmax, umax, wmax, tmax = plot_mode(kxmin, ramin,
                                           ncheb, title, **kwargs)
    return ramin, kxmin, pmax, umax, wmax, tmax

def crit_ra_l(l_harm, ncheb, gamma, phitop=None, phibot=None, eps=1.e-3):
    """Find critical Rayleigh of a given harmonic"""
    ra_min = 200
    ra_max = 1000
    sigma_min = np.real(
            eigval_spherical(l_harm, ra_min, ncheb, gamma,
                             phitop, phibot, output_eigvec=False))
    sigma_max = np.real(
            eigval_spherical(l_harm, ra_max, ncheb, gamma,
                             phitop, phibot, output_eigvec=False))

    while sigma_min > 0. or sigma_max < 0.:
        if sigma_min > 0.:
            ra_max = ra_min
            ra_min /= 2
        if sigma_max < 0.:
            ra_min = ra_max
            ra_max *=2
        sigma_min = np.real(
                eigval_spherical(l_harm, ra_min, ncheb, gamma,
                                 phitop, phibot, output_eigvec=False))
        sigma_max = np.real(
                eigval_spherical(l_harm, ra_max, ncheb, gamma,
                                 phitop, phibot, output_eigvec=False))

    while ra_max - ra_min > eps:
        ra_mean = (ra_min + ra_max) / 2
        sigma_mean = np.real(
                eigval_spherical(l_harm, ra_mean, ncheb, gamma,
                                 phitop, phibot, output_eigvec=False))
        if sigma_mean < 0.:
            sigma_min = sigma_mean
            ra_min = ra_mean
        else:
            sigma_max = sigma_mean
            ra_max = ra_mean

    return (ra_min * sigma_max - ra_max * sigma_min) / (sigma_max - sigma_min)

def normalize(arr):
    """Normalize complex array with element of higher modulus"""
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax

def plot_l_mode(l_harm, gamma, p_mode, t_mode, ur_mode, up_mode, rad_cheb, title):
    """Plot on spherical annulus"""
    # stream function
    psi_mode = 1.j * l_harm * p_mode

    # interpolation
    n_rad = 100
    n_phi = 400
    cheb_space = np.linspace(-1, 1, n_rad)
    rad = np.linspace(gamma, 1, n_rad)
    p_interp = dm.chebint(p_mode, cheb_space)
    psi_interp = dm.chebint(psi_mode, cheb_space)
    t_interp = dm.chebint(t_mode, cheb_space)
    ur_interp = dm.chebint(ur_mode, cheb_space)
    up_interp = dm.chebint(up_mode, cheb_space)

    # normalization
    p_norm, p_max = normalize(p_mode)
    t_norm, t_max = normalize(t_mode)
    ur_norm, ur_max = normalize(ur_mode)
    up_norm, up_max = normalize(up_mode)

    # profiles
    fig, axis = plt.subplots(1, 4, sharey=True)
    plt.setp(axis, xlim=[-1.1, 1.1], ylim=[gamma, 1],
             xticks=[-1, -0.5, 0., 0.5, 1])
    axis[0].plot(p_interp / p_max, rad)
    axis[0].plot(p_norm, rad_cheb, 'o')
    axis[0].set_xlabel(r'$P$', fontsize=FTSZ)
    axis[1].plot(ur_interp / ur_max, rad)
    axis[1].plot(ur_norm, rad_cheb, 'o')
    axis[1].set_xlabel(r'$u_r$', fontsize=FTSZ)
    axis[2].plot(up_interp / up_max, rad)
    axis[2].plot(up_norm, rad_cheb, 'o')
    axis[2].set_xlabel(r'$u_\phi$', fontsize=FTSZ)
    axis[3].plot(t_interp / t_max, rad)
    axis[3].plot(t_norm, rad_cheb, 'o')
    axis[3].set_xlabel(r'$\Theta$', fontsize=FTSZ)
    filename = '_'.join((title, 'mode_prof.pdf'))
    plt.savefig(filename, format='PDF')
    plt.close(fig)

    # 2D plot on annulus
    # mesh construction
    theta = np.pi/2
    phi = np.linspace(0, 2*np.pi, n_phi)
    rad_mesh, phi_mesh = np.meshgrid(rad, phi)

    # spherical harmonic
    harm = sph_harm(l_harm, l_harm, phi_mesh, theta)
    t_field = (t_interp * harm).real
    ur_field = (ur_interp * harm).real
    up_field = (up_interp * harm).real
    psi_field = (psi_interp * harm).real

    # normalization
    t_min, t_max = t_field.min(), t_field.max()
    t_field = 2 * (t_field - t_min) / (t_max - t_min) - 1

    # create annulus frame
    fig = plt.figure()
    tr = PolarAxes.PolarTransform()

    angle_ticks = [(0, r"$0$"),
                   (.5*np.pi, r"$\frac{\pi}{2}$"),
                   (1.*np.pi, r"$\pi$"),
                   (1.5*np.pi, r"$\frac{3\pi}{2}$")]
    grid_locator1 = FixedLocator([v for v, _ in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))

    radius_ticks = [(gamma, r'$\gamma$'), (1, '')]
    grid_locator2 = FixedLocator([v for v, _ in radius_ticks])
    tick_formatter2 = DictFormatter(dict(radius_ticks))

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(2.*np.pi, 0, 1, gamma),
        grid_locator1=grid_locator1, tick_formatter1=tick_formatter1,
        grid_locator2=grid_locator2, tick_formatter2=tick_formatter2)

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    axis = ax1.get_aux_axes(tr)

    # plot Temperature
    surf = axis.pcolormesh(phi_mesh, rad_mesh, t_field,
                           cmap='RdBu_r', shading='gouraud')
    cbar = plt.colorbar(surf, shrink=0.8)
    cbar.set_label(r'Temperature $\Theta$')
    # plot stream lines
    axis.contour(phi_mesh, rad_mesh, psi_field)
    axis.set_axis_off()
    plt.tight_layout()
    filename = '_'.join((title, 'mode.pdf'))
    plt.savefig(filename, format='PDF')
    plt.close(fig)

def findplot_ra_l(title, ncheb, gamma, lmin=1, lmax=8,
                  phibot=None, phitop=None,
                  plot_rac_l=True, plot_eigvec=True):
    """
    Find the minimum and plot Ra(l)

    Inputs
    -----
    title: string variable to use in figure name
    ncheb: number of Chebyshev points
    lmax: max harmonic degree
    gamma: r_bot/r_top
    phitop and phibot: phase change numbers
    plot_rac_l: whether to plot Ra_c vs l-mode
    plot_eigvec: whether to plot eigenvectors
    """
    rac_l = []
    for l_harm in range(lmin, lmax+1):
        # look for Ra_c of each harmonic
        rac_l.append(crit_ra_l(l_harm, ncheb, gamma, phitop, phibot))

    l_c, ra_c = min(enumerate(rac_l), key=lambda tpl: tpl[1])
    l_c += lmin

    if plot_rac_l:
        fig = plt.figure()
        plt.plot(range(lmin, lmax+1), rac_l, linewidth=2)
        plt.plot(l_c, ra_c, 'o', label=r'$Ra_{min}=%.2f ; l=%d$' %(ra_c, l_c))
        plt.xlabel(r'Spherical harmonic $l$', fontsize=FTSZ)
        plt.ylabel(r'Critical Rayleigh number $Ra_c$', fontsize=FTSZ)
        plt.xticks(fontsize=FTSZ)
        plt.yticks(fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        filename = '_'.join((title, 'Ra_l.pdf'))
        plt.savefig(filename, format='PDF')
        plt.close(fig)

    if plot_eigvec:
        _, p_mode, t_mode, ur_mode, up_mode, radius = eigval_spherical(
            l_c, ra_c, ncheb, gamma, phitop, phibot, output_eigvec=True)
        plot_l_mode(l_c, gamma, p_mode, t_mode,
                    ur_mode, up_mode, radius, title)

if SPHERICAL:
    findplot_ra_l(title, ncheb=NCHEB, gamma=GAMMA, lmin=1, lmax=8,
                  phibot=PHI_BOT, phitop=PHI_TOP)
else:
    ramin, kxmin, pmax, umax, wmax, tmax = findplot_rakx(
        NCHEB, title, plotfig=True, output_rakx=True, bcsu=BCSU,
        phase_change_top=(PHI_TOP is not None), phitop=PHI_TOP,
        phase_change_bot=(PHI_BOT is not None), phibot=PHI_BOT,
        output_eigvec=True)
    print('max =', ramin, kxmin, pmax, umax, wmax, tmax)

if COMPUTE_PHASECHANGE:
    nphi = 100
    phinum = np.flipud(np.power(10, np.linspace(-3, 5, nphi)))
    # Limit case for infinite phi
    rac = 27*np.pi**4/4
    kxc = np.pi/np.sqrt(2)
    # NCHEB = 30

    EQUAL_PHI = False
    # Computes properties as function of phi, equal for both boundaries
    if EQUAL_PHI:
        # First unstable mode
        # rrm, kkx = ra_ks(500, 2, NCHEB,
                        # phase_change_top=True, phase_change_bot=True,
                        # phitop=phinum[0], phibot=phinum[0])
        rrm, kkx, pmx, umx, wmx, tmx = findplot_rakx(NCHEB,
                        'PhaseChangeTop',
                        plotfig=False,
                        output_rakx=False,
                        phase_change_top=True, phase_change_bot=True,
                        phibot=phinum[0], phitop=phinum[0],
                        output_eigvec=True)
        ram = [rrm]
        kwn = [kkx]
        pmax = [pmx]
        umax = [umx]
        wmax = [wmx]
        tmax = [tmx]
        print(phinum[0], ram, kwn)
        for i, phi in enumerate(phinum[1:]):
            # rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
            #             phase_change_top=True, phase_change_bot=True,
            #             phitop=phi, phibot=phi)
            rrm, kkx, pmx, umx, wmx, tmx = findplot_rakx(NCHEB,
                        'PhaseChangeTop',
                        plotfig=False,
                        output_rakx=False,
                        phase_change_top=True, phase_change_bot=True,
                        phibot=phi, phitop=phi,
                        output_eigvec=True)
            print(i, phi, rrm, kkx)
            ram = np.append(ram, rrm)
            kwn = np.append(kwn, kkx)
            pmax = np.append(pmax, pmx)
            umax = np.append(umax, umx)
            wmax = np.append(wmax, wmx)
            tmax = np.append(tmax, tmx)
        # save
        with open('EqualTopBotPhi.dat', 'w') as fich:
            fmt = '{:13}'*7 + '\n'
            fich.write(fmt.format(' phi', 'kx', 'Ra', 'Pmax', 'Umax', 'Tmax', 'Wmax'))
            fmt = '{:15.3e}'*6 + '{:15.3}' + '\n'
            for i in range(nphi):
                fich.write(fmt.format(phinum[i], kwn[i], ram[i], pmax[i],
                                      umax[i], tmax[i], wmax[i]))

        # plot kx and ra as function of phi
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Theoretical prediction for translation
        axe[0].semilogx(phinum, 24*phinum, '--', c='k', label=r'Translation mode')
        # Theoretical for low phi development
        axe[0].semilogx(phinum, 24*phinum-81*phinum**2/256, '-', c='k',
                        label=r'Small $\Phi$ prediction')
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                        label=r'$\frac{27\pi^4}{4}$')
        # col0 = p0.get_color()
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE,
                              label=r'Fastest growing mode')
        col1 = p1.get_color()
        # General case
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=4)
        axe[1].loglog([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Small phi prediction
        axe[1].loglog(phinum, np.sqrt(9*phinum/32), '-', c='k',
                    label=r'Small $\Phi$ prediction')
        axe[1].loglog(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'Fastest growing mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^+=\Phi^-$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_EqualPhi.pdf", format='PDF')
        plt.close(fig)

        # plot Pmax, Umax, Wmax and (24phi - Ra) as function of phi
        fig, axe = plt.subplots(4, 1, sharex=True)
        axe[0].loglog(phinum, np.abs(pmax), marker='.', linestyle='-')
        axe[0].loglog(phinum, 13*np.sqrt(13)/8*phinum, linestyle='--', c='k',
                        label=r'$13\sqrt{13}\Phi/8$')
        axe[0].legend(loc='upper left', fontsize=FTSZ)
        axe[0].set_ylabel(r'$P_{max}$', fontsize=FTSZ)
        axe[1].loglog(phinum, np.abs(umax), marker='.', linestyle='-')
        axe[1].loglog(phinum, 3*np.sqrt(phinum/2), linestyle='--', c='k',
                        label=r'$3\sqrt{\Phi/2}$')
        axe[1].legend(loc='upper left', fontsize=FTSZ)
        axe[1].set_ylabel(r'$U_{max}$', fontsize=FTSZ)
        axe[2].semilogx(phinum, np.abs(wmax), marker='.', linestyle='-')
        axe[2].semilogx(phinum, 8*np.ones(phinum.shape), linestyle='--', c='k',
                        label=r'$8$')
        axe[2].set_ylim(ymin=7)
        axe[2].legend(loc='upper left', fontsize=FTSZ)
        axe[2].set_ylabel(r'$W_{max}$', fontsize=FTSZ)
        axe[3].loglog(phinum, 24*phinum-ram, marker='.', linestyle='-')
        axe[3].loglog(phinum, 81*phinum**2/256,  linestyle='--', c='k',
                        label=r'$81\Phi^2/256$')
        axe[3].set_ylabel(r'$24\Phi-Ra$', fontsize=FTSZ)
        axe[3].legend(loc='upper left', fontsize=FTSZ)
        axe[3].set_xlabel(r'$\Phi$', fontsize=FTSZ)
        # axe[3].set_ylim([1.e-6, 1.e2])
        plt.savefig('Phi_ModeMax.pdf', format='PDF')
        plt.close(fig)

    PHASETOPBOT = False
    if PHASETOPBOT:
        phib = 1.e-3
        phit = 1.e-3
        ramin, kxmin, pmax, umax, wmax, tmax = findplot_rakx(NCHEB,
                      'PhaseChangeTop' + np.str(phit).replace('.', '-') + 'Bot' + np.str(phib).replace('.', '-'),
                      plotfig=True,
                      output_rakx=True,
                      phase_change_top=True, phase_change_bot=True,
                      phibot=phib, phitop=phit,
                      output_eigvec=True)
        print('max =', ramin, kxmin, pmax, umax, wmax, tmax)

    PHASEBOTONLY = False
    if PHASEBOTONLY:
        # phib = 1.e-2
        # ramin, kxmin, pmax, umax, wmax, tmax = findplot_rakx(NCHEB,
                      # 'FreeTopPhaseChangeBot' + np.str(phib).replace('.', '-'),
                      # plotfig=True,
                      # output_rakx=True,
                      # phase_change_top=False, phase_change_bot=True,
                      # bcsu=np.array([[0, 1, 0], [0, 1, 0]], float),
                      # phibot=phib,
                      # output_eigvec=True)
        # print('max =', ramin, kxmin, pmax, umax, wmax, tmax)
        nphi = 20
        phinum = np.flipud(np.power(10, np.linspace(-2, 4, nphi)))
        ram = np.zeros(phinum.shape)
        kwn = np.zeros(phinum.shape)
        pmax = np.zeros(phinum.shape)
        umax = np.zeros(phinum.shape)
        wmax = np.zeros(phinum.shape)
        tmax = np.zeros(phinum.shape)
        # print(ram, kwn)
        for i, phi in enumerate(phinum):
            ram[i], kwn[i], pmax[i], umx, wmax[i], tmax[i] = findplot_rakx(
                NCHEB, 'PhaseChangeBot', plotfig=False, output_rakx=False,
                phase_change_top=False, phase_change_bot=True, phibot=phi,
                bcsu=np.array([[0, 1, 0], [0, 1, 0]], float),
                output_eigvec=True)
            umax[i] = np.imag(umx)
            print(i, phi, ram[i], kwn[i])
            # save in file
        with open('FreeTopBotPhase.dat', 'w') as fich:
            fmt = '{:15}'*7 + '\n'
            fich.write(fmt.format(' phi', 'kx', 'Ra', 'Pmax', 'Umax', 'Tmax', 'Wmax'))
            fmt = '{:15.3e}'*6 + '{:15.3}' + '\n'
            for i in range(nphi):
                fich.write(fmt.format(phinum[i], kwn[i], ram[i], pmax[i],
                                      umax[i], tmax[i], wmax[i]))
        # Now plot
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Ra
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                               label=r'$\frac{27\pi^4}{4}$')
        # general case
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE, label=r'$\Phi^+=\infty$, varying $\Phi^-$')
        col1 = p1.get_color()
        # axe[0].semilogx(phinum, ram2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=7)
        # kx
        # classical RB case
        axe[1].semilogx([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Free top, phase change at bottom
        axe[1].semilogx(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'$\Phi^+=\infty$, varying $\Phi^-$')
        # axe[1].loglog(phinum, kwn2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^-$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_VaryingPhiBotFreeTop.pdf", format='PDF')
        plt.close(fig)
        # plot Pmax, Umax, Wmax and (24phi - Ra) as function of phi
        fig, axe = plt.subplots(3, 1, sharex=True)
        axe[0].semilogx(phinum, np.abs(pmax), marker='.', linestyle='-')
        axe[0].legend(loc='upper left', fontsize=FTSZ)
        axe[0].set_ylabel(r'$P_{max}$', fontsize=FTSZ)
        axe[1].semilogx(phinum, np.abs(umax), marker='.', linestyle='-')
        axe[1].legend(loc='upper left', fontsize=FTSZ)
        axe[1].set_ylabel(r'$U_{max}$', fontsize=FTSZ)
        axe[2].semilogx(phinum, np.abs(wmax), marker='.', linestyle='-')
        axe[2].legend(loc='upper left', fontsize=FTSZ)
        axe[2].set_ylabel(r'$W_{max}$', fontsize=FTSZ)
        axe[2].set_xlabel(r'$\Phi$', fontsize=FTSZ)
        # axe[3].set_ylim([1.e-6, 1.e2])
        plt.savefig('Phi_ModeMaxFreeTopPhaseBot.pdf', format='PDF')
        plt.close(fig)

    DIFFERENT_PHI = False
    # Compute solution properties as function of both phitop and phibot
    if DIFFERENT_PHI:
        # Botphase = False and varying phitop
        rrm, kkx = ra_ks(500, 2, NCHEB,
                        phase_change_top=True, phase_change_bot=False,
                        bcsu=np.array([[1, 0, 0], [0, 1, 0]], float),
                        phitop=phinum[0])
        ram = [rrm]
        kwn = [kkx]
        print(ram, kwn)
        for i, phi in enumerate(phinum[1:]):
            rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
                        phase_change_top=True, phase_change_bot=False,
                        bcsu=np.array([[1, 0, 0], [0, 1, 0]], float),
                        phitop=phi)
            print(i, phi, rrm, kkx)
            ram = np.append(ram, rrm)
            kwn = np.append(kwn, kkx)
        # now keep top to the lowest value and change phibot
        ram2 = [ram[-1]]
        kwn2 = [kwn[-1]]
        print(ram2, kwn2)
        for i, phi in enumerate(phinum):
            rrm, kkx = ra_ks(ram[i], kwn[i], NCHEB,
                        phase_change_top=True, phase_change_bot=True,
                        phitop=phinum[-1], phibot=phi)
            print(i, phi, rrm, kkx)
            ram2 = np.append(ram2, rrm)
            kwn2 = np.append(kwn2, kkx)

        # Now plot
        fig, axe = plt.subplots(2, 1, sharex=True)
        # Ra
        # Theoretical prediction for translation
        axe[0].semilogx(phinum, 24*phinum, '--', c='k', label='Translation mode')
        # classical RB case
        axe[0].semilogx([phinum[0], phinum[-1]], [rac, rac], '-.', c='k',
                               label=r'$\frac{27\pi^4}{4}$')
        # general case
        p1, = axe[0].semilogx(phinum, ram, 'o', markersize=MSIZE, label=r'$\Phi^-=\infty$, varying $\Phi^+$')
        p2, = axe[0].semilogx(phinum, ram2[1:], 'o', markersize=MSIZE, label='Varying $\Phi^-$, $\Phi^+=10^{-2}$')
        col1 = p1.get_color()
        col2 = p2.get_color()
        # axe[0].semilogx(phinum, ram2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[0].set_ylabel(r'$Ra$', fontsize=FTSZ)
        axe[0].set_ylim([0, 700])
        axe[0].legend(loc=7)
        # kx
        # classical RB case
        axe[1].semilogx([phinum[0], phinum[-1]], [kxc, kxc], '-.', c='k',
                    label=r'$\frac{\pi}{\sqrt{2}}$')
        # Free bottom, phase change at top
        axe[1].semilogx(phinum, kwn, 'o', markersize=MSIZE, c=col1,
                    label=r'$\Phi^-=\infty$, varying $\Phi^+$')
        # Gradually openning bottom
        axe[1].semilogx(phinum, kwn2[1:], 'o', markersize=MSIZE, c=col2,
                    label='Varying $\Phi^-$, $\Phi^+=10^{-2}$')
        # axe[1].loglog(phinum, kwn2, 'o', markersize=MSIZE, label='Second fastest mode')
        axe[1].legend(loc=4)
        axe[1].set_ylabel(r'$k$', fontsize=FTSZ)
        axe[1].set_xlabel(r'$\Phi^-,\quad \Phi^+$', fontsize=FTSZ)
        plt.savefig("Phi-Ra-kx_VaryingPhiBotTop.pdf", format='PDF')
        plt.close(fig)

STAB_TRANSLATION = False
# Computes the linear stability of the steady translation mode
if STAB_TRANSLATION:
    phib = 0.01
    phit = phib
    # epsilon = np.flipud(np.linspace(0, 1, 4))
    epsilon = np.array([5])
    wkn = np.power(10, np.linspace(-1, 4, 100))
    sigma = np.zeros(wkn.shape)
    NCHEB = 30
    # rmin, kmin = ra_ks(ran, wkn, NCHEB,
                   # phase_change_top=True,
                   # phase_change_bot=True,
                   # phitop = phit,
                   # phibot = phib,
                   # translation=True)
    # print('eps = ', (rmin-rtr)/rtr, 'kmin = ', kmin)
    # rao = search_ra(wkn, ran, NCHEB,
                   # phase_change_top=True,
                   # phase_change_bot=True,
                   # phitop = phit,
                   # phibot = phib,
                   # translation=True)
    # print('eps = ', (rao-rtr)/rtr)
    axe = plt.subplot()
    for j, eps in enumerate(epsilon):
        rtr = 12*(phib+phit)
        ran = rtr*(1+eps)
        for i, kxn in enumerate(wkn):
            sigma[i] = eigval_cartesian(kxn, ran, NCHEB,
                                        phase_change_top=True,
                                        phase_change_bot=True,
                                        phitop = phit,
                                        phibot = phib,
                                        translation=True)

        axe.semilogx(wkn, np.real(sigma), label=r'$\varepsilon = %.2f$' %(eps))
        axe.set_xlabel(r'$k$', fontsize=FTSZ)
        axe.set_ylabel(r'$Re(\sigma)$', fontsize=FTSZ)
        plt.legend(loc='upper right', fontsize=FTSZ)
        # axe.set_ylim((-500, 1500))
        # axe.set_ylim(bottom=-500)

    plt.savefig('sigmaRa'+np.str(eps)+'N'+np.str(NCHEB)+'Top'+np.str(phit).replace('.', '-') + 'Bot' + np.str(phib).replace('.', '-')+'.pdf')

