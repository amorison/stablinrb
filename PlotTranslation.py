#!/usr/bin/env python3
"""
Plot results from TranslationStability.py read from data file

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

mpl.rcParams['pdf.fonttype'] = 42

# Font and markers size
FTSZ = 8
MSIZE = 3

infile = 'Phi_sigma_Ra_SAVE.dat'
numlines = sum(1 for line in open(infile))
phitot = np.zeros(numlines)
kmax = np.zeros(numlines)
smax0 = np.zeros(numlines)
kminus0 = np.zeros(numlines)
kplus0 = np.zeros(numlines)
hmax = np.zeros(numlines)
ramax = np.zeros(numlines)
epsmax = np.zeros(numlines)

print('reading from file: ' + infile)
phitot, kmax0, smax0, kminus0, kplus0, hmax, ramax, epsmax = \
  np.loadtxt(infile, unpack=True)

# Only use phi<1 data for the power law fit
mask = phitot <= 1
phis = phitot[mask]
eps = epsmax[mask]
smas = smax0[mask]
# now fit
lexs, lcos = np.polyfit(np.log10(phis), np.log10(smas), 1)
coefs = 10**lcos
sig = lambda x: coefs * x**lexs
lexr, lcor = np.polyfit(np.log10(phis), np.log10(eps), 1)
coefr = 10**lcor
ram = lambda x: coefr * x**lexr

fig, axe = plt.subplots(2, 1, sharex=True)
# growth rate
axe[0].loglog(phitot, sig(phitot), '--', label=r'$\sigma = %.3f \Phi^{%.3f}$' %(coefs, lexs))
axe[0].loglog(phitot, smax0, 'o', markersize=MSIZE, label=r'Maximum growth rate at $\varepsilon=0$')
axe[0].legend(loc='upper left', fontsize=FTSZ)
axe[0].set_ylabel('$Re(\sigma)$', fontsize=FTSZ+2)
# max Ra
axe[1].loglog(phitot, ram(phitot), '--', label=r'$\varepsilon_{max} = %.5f \Phi^{%.3f}$' %(coefr, lexr))
axe[1].loglog(phitot, epsmax, 'o', markersize=MSIZE, label=r'Maximum $Ra$ for instability')
axe[1].legend(loc='upper left', fontsize=FTSZ)
axe[1].set_ylabel(r'$\varepsilon_{max}$', fontsize=FTS+2)
axe[1].set_xlabel(r'$\Phi^+=\Phi^-$', fontsize=FTSZ+2)
plt.savefig('Phi_kx_smax_epsmax.pdf')
plt.close(fig)
