""" """
from importlib.metadata import version, PackageNotFoundError
# automatically set __version__ to the global version of the package declared
# in setup.cfg or set by the setuptools_scm package
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass  # package is not installed

from reemission.config_registration import register_configs

# Automatically register configs at package import
register_configs()
