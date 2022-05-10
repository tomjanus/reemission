""" Main caller of the toolbox functions """
import os
import sys
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.presenter import LatexWriter, JSONWriter


def test_emissions():
    """ Calculate emission factors and profiles using
        dummy reservoir, catchment and emission input data in json format """
    input_data = Inputs.fromfile(
        os.path.join('test', 'test_data', 'inputs.json'))
    output_config = os.path.join('src', 'reemission', 'config', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config)
    model.calculate()
    model.add_presenter(
        writers=[LatexWriter, JSONWriter],
        output_files=[os.path.join('..', 'outputs', 'output_main.tex'),
                      os.path.join('..', 'outputs', 'output_main.json')])
    model.save_results()


def test_importlib():
    """ Test importing package data using importlib functionality """
    # Import the package based on Python's version
    if sys.version_info < (3, 9):
        # importlib.resources either doesn't exist or lacks the files()
        # function, so use the PyPI version:
        import importlib_resources
    else:
        # importlib.resources has files(), so use that:
        import importlib.resources as importlib_resources

    pkg = importlib_resources.files("reemission")/"config"
    for item in pkg.iterdir():
        if not item.is_dir():
            print(item)


if __name__ == '__main__':
    # Run test functions
    test_emissions()
    test_importlib()
