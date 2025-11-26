#    This file is part of Re-Emission.
#
#    Re-Emission is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Re-Emission is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Re-Emission.  If not, see <http://www.gnu.org/licenses/>.

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
