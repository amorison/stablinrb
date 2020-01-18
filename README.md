# StabLinRB
======

Computes the linear stability in the Rayleigh-Bénard problem.

The curve of critical Rayleigh number as function of wavenumber is computed
and plotted and the minimum value of the Rayleigh number is obtained along
with the corresponding value of the wavenumber. The first unstable
mode can also be plotted, both as profiles of the z dependence and as
temperature-velocity maps.

The calculation uses an implementation of DMSuite in Python available on github
as part of the pyddx package (https://github.com/labrosse/dmsuite).
DMSuite was originally developed for matlab by
Weidemann and Reddy and explained in ACM Transactions of Mathematical
Software, 4, 465-519 (2000). The present code is based on an octave code
originally developed by T. Alboussière and uses the Chebyshev differentiation
matrices.
