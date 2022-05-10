""" """
from pkg_resources import get_distribution, DistributionNotFound

# automatically set __version__ to the global version of the package declared
# in setup.cfg or set by the setuptools_scm package
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed
