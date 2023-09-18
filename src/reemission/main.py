""" Main caller of the toolbox functions """
import os
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.presenter import LatexWriter, JSONWriter, ExcelWriter
from reemission.utils import get_package_file
# TODO: move this to tests and change paths


def run_emissions() -> EmissionModel:
    """Calculate emission factors and profiles using
    dummy reservoir, catchment and emission input data in json format"""
    input_data = Inputs.fromfile(
        get_package_file('../../tests/test_data/inputs.json'))
    output_config = get_package_file('config', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config.as_posix())
    model.calculate()
    model.add_presenter(
        writers=[LatexWriter, JSONWriter, ExcelWriter],
        output_files=[
            get_package_file('../../outputs/', 'output_main_18_09.tex'),
            get_package_file('../../outputs/', 'output_main_18_09.json'),
            get_package_file('../../outputs/', 'output_main_18_09.xlsx')],
    )
    model.save_results()
    return model


if __name__ == '__main__':
    # Run test functions
    import pprint
    model = run_emissions()
    #pprint.pprint(model.outputs)
    #pprint.pprint(model.internal)
