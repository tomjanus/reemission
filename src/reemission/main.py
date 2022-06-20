""" Main caller of the toolbox functions """
import os
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.presenter import LatexWriter, JSONWriter
from reemission.utils import get_package_file
# TODO: move this to tests and change paths


def test_emissions():
    """Calculate emission factors and profiles using
    dummy reservoir, catchment and emission input data in json format"""
    input_data = Inputs.fromfile(os.path.join(
        '/home/lepton/Dropbox (The University of Manchester)/git_projects/reemission/tests/test_data/inputs.json'))
    output_config = get_package_file('config', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config.as_posix())
    model.calculate()
    model.add_presenter(
        writers=[LatexWriter, JSONWriter],
        output_files=[
            os.path.join('/home/lepton/Dropbox (The University of Manchester)/git_projects/reemission/outputs/', 'output_main.tex'),
            os.path.join('/home/lepton/Dropbox (The University of Manchester)/git_projects/reemission/outputs/', 'output_main.json')],
    )
    model.save_results()


if __name__ == '__main__':
    # Run test functions
    test_emissions()
