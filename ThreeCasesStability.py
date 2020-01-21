#!/usr/bin/env python3
"""
Finds critical Rayleigh number for three cases, with different BCs.


"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from stablinrb.analyzer import LinearAnalyzer, NonLinearAnalyzer
from stablinrb.physics import PhysicalProblem, compo_smo, visco_Arrhenius
from stablinrb.misc import normalize_modes

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)
mpl.rcParams['pdf.fonttype'] = 42

# Font and markers size
FTSZ = 11
MSIZE = 2
GAMMA = 10 / 11

# Define the first physical problem
pblm = PhysicalProblem(
    gamma=None,
    freeslip_top=False,
    freeslip_bot=False)

# create the analyser instance
ana = LinearAnalyzer(pblm, ncheb=20)

hmax = 7
nharm = 100

# rigid--rigid case
hmin = 1
wnum = np.linspace(hmin, hmax, nharm)

# Find the fastest growing mode
ra_crr, harm_crr = ana.critical_ra()
print('Rigid-Rigid Rac, kc = ', ra_crr, harm_crr)
print('Wavelength = ', 2 * np.pi / harm_crr)

# compute the k, Ra curve
raylrr = [ana.neutral_ra(wnum[0], ra_crr)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylrr[i])
    raylrr = np.append(raylrr, ra2)

# and plot
plt.plot(wnum, raylrr, label=r'Rigid--Rigid BCs')
plt.plot(harm_crr, ra_crr, 'o',
             label=r'$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$' %(ra_crr, harm_crr))

# Free--rigid case
# Modifies the relevant physical parameter
ana.phys.freeslip_top = True
hmin = 0.7
wnum = np.linspace(hmin, hmax, nharm)

# Find the fastest growing mode
ra_cfr, harm_cfr = ana.critical_ra()
print('Free-Rigid Rac, kc = ', ra_cfr, harm_cfr)
print('Wavelength = ', 2 * np.pi / harm_cfr)

# compute the k, Ra curve
raylfr = [ana.neutral_ra(wnum[0], ra_cfr)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylfr[i])
    raylfr = np.append(raylfr, ra2)

# and plot
plt.plot(wnum, raylfr, label=r'Free--Rigid BCs')
plt.plot(harm_cfr, ra_cfr, 'o',
             label=r'$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$' %(ra_cfr, harm_cfr))

# Free--free case
ana.phys.freeslip_bot = True
hmin = 0.5
wnum = np.linspace(hmin, hmax, nharm)

ra_cff, harm_cff = ana.critical_ra()
print('Free-Free Rac, kc = ', ra_cff, harm_cff)
print('Wavelength = ', 2 * np.pi / harm_cff)

raylff = [ana.neutral_ra(wnum[0], ra_cff)]
for i, kk in enumerate(wnum[1:]):
    ra2 = ana.neutral_ra(kk, raylff[i])
    raylff = np.append(raylff, ra2)

plt.plot(wnum, raylff, label=r'Free--Free BCs')
plt.plot(harm_cff, ra_cff, 'o',
             label=r'$\mbox{\textit{Ra}}_{min}=%.2f ; k=%.2f$' %(ra_cff, harm_cff))

# Complete the figure
plt.legend(loc='upper center', fontsize=14)
plt.xlabel(r'Wavenumber, k', fontsize=14)
plt.ylabel(r'Neutral $\mbox{\textit{Ra}}$', fontsize=14)

# and save
plt.savefig('ThreeBCstability.pdf', bbox_inches='tight', format='PDF')
