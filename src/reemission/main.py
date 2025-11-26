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

"""
Main caller of the toolbox functions.

This module runs the main emission calculation process using dummy reservoir,
catchment, and emission input data in JSON format. The results are output in
LaTeX, JSON, and Excel formats.
"""
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.presenter import LatexWriter, JSONWriter, ExcelWriter
from reemission.utils import get_package_file
from reemission import registry
# TODO: move this to tests and change paths


def run_emissions() -> EmissionModel:
    """Calculate emission factors and profiles.

    This function uses dummy reservoir, catchment, and emission input data in JSON format to calculate
    emission factors and profiles. The results are saved in LaTeX, JSON, and Excel formats.

    Returns:
        EmissionModel: An instance of the EmissionModel class with calculated emissions and results.
    """
    input_data = Inputs.fromfile(
        get_package_file('../../tests/test_data/inputs.json'))
    output_config = registry.config.get("report_outputs")
    model = EmissionModel(inputs=input_data, presenter_config=output_config)
    model.calculate()
    model.add_presenter(
        writers=[LatexWriter, JSONWriter, ExcelWriter],
        output_files=[
            get_package_file('../../outputs/', 'test_output.tex'),
            get_package_file('../../outputs/', 'test_output.json'),
            get_package_file('../../outputs/', 'test_output.xlsx')],
    )
    model.save_results()
    return model


if __name__ == '__main__':
    # Run test functions
    import pprint
    model = run_emissions()
    pprint.pprint(model.outputs)
    pprint.pprint(model.internal)
