#!/usr/bin/env python3
"""Weakly non-linear development for Rayleigh-Bénard problem

The Rayleigh number Ra and the solution vectore are developed as
    Ra = Rc + R1 eps + R2 eps^2 + ...
    X = eps X1 + eps^2 X2 + ...

This script finds the suite of Ri and Xi and then plots the dimensionless
heat flux (Nusselt number) as function of Ra, as well as the weakly
non-linear solution for a given value of Ra.

Warning: Only tested up to degre 2 in eps.

Weakly non-linear analysis not implemented for spherical geometry,
volumetric heating or depth-dependent physical properties, which make the
linear operator non-hermitian.

See Labrosse et al (J. Fluid Mech. 2018) for details.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import stablinrb.plotting as stabplt
from stablinrb.geometry import Cartesian
from stablinrb.nonlin import NonLinearAnalyzer
from stablinrb.physics import PhysicalProblem

# include tex fonts in pdf
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("text", usetex=True)
mpl.rcParams["pdf.fonttype"] = 42

# Font and markers size
FTSZ = 11

pblm = PhysicalProblem(
    geometry=Cartesian(),
    freeslip_top=True,
    freeslip_bot=True,
)

ana = NonLinearAnalyzer(pblm, ncheb=30, nnonlin=2)
ray = ana.ratot
qtop = ana.qtop
moyt = ana.meant

print("Critical wavenumber kc = ", ana.harm_c)
print("Coefficients of development in Rayleigh = ", ray)
coef_nu = ray[0] * qtop[2] / ray[2]
print("Nusselt  = 1 + %.2f (Ra - Rc) / Rc" % (coef_nu))

coef_t = moyt[2] * ray[0] / ray[2]
print("Mean temperature = 0.5 + %.2f (Ra - Rc) / Rc" % (coef_t))

# plot the solution for a given finite value of the reduced Rayleigh number
epsmax = 5.58
print("Plotting the temperature and velocity for Ra = %.2f" % (ray[0] * (1 + epsmax)))
stabplt.plot_mode_image(ana, eps=epsmax, plot_ords=False)

# plot Nusselt number and mean temperature as function of the Rayleigh number
nterms = qtop.shape[0]
eps = np.linspace(0, epsmax, num=20)
vdm = np.vander(eps, nterms, increasing=True)
rayl = np.dot(vdm, ray)
nuss = np.dot(vdm, qtop)
meant = np.dot(vdm, moyt)
fig, axe = plt.subplots(2, 1, sharex=True)
axe[0].plot(rayl, nuss)
axe[0].set_ylabel(r"$\mathrm{Nu}$", fontsize=FTSZ)
axe[1].plot(rayl, meant)
axe[1].set_xlabel(r"$\mathrm{Ra}$", fontsize=FTSZ)
axe[1].set_ylabel(r"$\langle T\rangle$", fontsize=FTSZ)
axe[1].set_ylim([0.4, 0.6])
plt.savefig("Ra-Nu-Tmean.pdf", format="PDF")
