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
    model = EmissionModel(inputs=input_data, config=output_config)
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
