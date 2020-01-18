"""StabLinRB is a linear stability tool for Rayleigh-Bénard convection."""

from pkg_resources import get_distribution, DistributionNotFound
from setuptools_scm import get_version

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stablinrb').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'
