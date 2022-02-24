""" Main caller of the toolbox functions """
import os
from src.emissions.model import EmissionModel
from src.emissions.input import Inputs
from src.emissions.presenter import LatexWriter, JSONWriter


def test_emissions():
    """ Calculate emission factors and profiles using
        dummy reservoir, catchment and emission input data in json format """
    input_data = Inputs.fromfile(
        os.path.join('tests', 'test_data', 'inputs.json'))
    output_config = os.path.join('config', 'emissions', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config)
    model.calculate()
    model.add_presenter(
        writers=[LatexWriter, JSONWriter],
        output_files=[os.path.join('outputs', 'output_main.tex'),
                      os.path.join('outputs', 'output_main.json')])
    model.save_results()


if __name__ == '__main__':
    # Run test functions
    test_emissions()
