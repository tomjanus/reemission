""" Main caller of the toolbox functions """
from src.emissions.model import EmissionModel
from src.emissions.input import Inputs


def test_emissions():
    """ Calculate emission factors and profiles using
        dummy reservoir, catchment and emission input data in json format """
    input_data = Inputs.fromfile(
        '/home/lepton/Dropbox (The University of Manchester)/git_projects/' +
        'dam-emissions/tests/test_data/inputs.json')
    output_config = '/home/lepton/Dropbox (The University of Manchester)/' + \
        'git_projects/dam-emissions/config/emissions/outputs.yaml'
    model = EmissionModel(inputs=input_data, config=output_config)
    model.calculate()


if __name__ == '__main__':
    # Run test functions
    test_emissions()
