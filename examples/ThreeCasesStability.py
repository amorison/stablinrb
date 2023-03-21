"""Find critical Rayleigh number for three cases with different BCs."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from stablinrb.cartesian import CartStability
from stablinrb.physics import FreeSlip, Rigid

mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("text", usetex=True)
mpl.rcParams["pdf.fonttype"] = 42

# rigid--rigid case
ana = CartStability(
    chebyshev_degree=10,
    bc_mom_top=Rigid(),
    bc_mom_bot=Rigid(),
)

hmax = 7.0
nharm = 75
hmin = 1.0
wnum = np.linspace(hmin, hmax, nharm)

# Find the fastest growing mode
ra_crr, harm_crr = ana.critical_ra()
print("Rigid-Rigid Rac, kc = ", ra_crr, harm_crr)
print("Wavelength = ", 2 * np.pi / harm_crr)

# compute the k, Ra curve
raylrr = [ana.neutral_ra(wnum[0], ra_crr)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylrr[i])
    raylrr.append(ra2)

# and plot
plt.plot(wnum, raylrr, label=r"Rigid--Rigid BCs")
plt.plot(
    harm_crr,
    ra_crr,
    "o",
    label=r"$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$" % (ra_crr, harm_crr),
)

# Free--rigid case
ana = CartStability(
    chebyshev_degree=10,
    bc_mom_top=FreeSlip(),
    bc_mom_bot=Rigid(),
)
hmin = 0.7
wnum = np.linspace(hmin, hmax, nharm)

# Find the fastest growing mode
ra_cfr, harm_cfr = ana.critical_ra()
print("Free-Rigid Rac, kc = ", ra_cfr, harm_cfr)
print("Wavelength = ", 2 * np.pi / harm_cfr)

# compute the k, Ra curve
raylfr = [ana.neutral_ra(wnum[0], ra_cfr)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylfr[i])
    raylfr.append(ra2)

# and plot
plt.plot(wnum, raylfr, label=r"Free--Rigid BCs")
plt.plot(
    harm_cfr,
    ra_cfr,
    "o",
    label=r"$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$" % (ra_cfr, harm_cfr),
)

# Free--free case
ana = CartStability(
    chebyshev_degree=10,
    bc_mom_top=FreeSlip(),
    bc_mom_bot=FreeSlip(),
)
hmin = 0.5
wnum = np.linspace(hmin, hmax, nharm)

ra_cff, harm_cff = ana.critical_ra()
print("Free-Free Rac, kc = ", ra_cff, harm_cff)
print("Wavelength = ", 2 * np.pi / harm_cff)

raylff = [ana.neutral_ra(wnum[0], ra_cff)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylff[i])
    raylff.append(ra2)

plt.plot(wnum, raylff, label=r"Free--Free BCs")
plt.plot(
    harm_cff,
    ra_cff,
    "o",
    label=r"$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$" % (ra_cff, harm_cff),
)

# Complete the figure
plt.legend(loc="upper center", fontsize=14)
plt.xlabel(r"Wavenumber, k", fontsize=14)
plt.ylabel(r"Neutral $\mbox{\textit{Ra}}$", fontsize=14)

# and save
plt.savefig("ThreeBCstability.pdf", bbox_inches="tight", format="PDF")
