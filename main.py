""" Main caller of the toolbox functions """
from types import SimpleNamespace
from src.emissions.utils import read_table
from src.emissions.model import EmissionModel
from src.emissions.input import Inputs


def test_read_yaml():
    table = read_table('./data/emissions/McDowell/landscape_TP_export.yaml')
    table_ns = SimpleNamespace(**table)
    print(table_ns)


def test_emissions():
    """ Calculate emission factors and profiles using
        dummy reservoir, catchment and emission input data in json format """
    input_data = Inputs.fromfile(
        '/home/lepton/Dropbox (The University of Manchester)/git_projects/' +
        'dam-emissions/tests/test_data/inputs.json')
    model = EmissionModel()
    model.calculate(inputs=input_data)


if __name__ == '__main__':
    # Run test functions
    test_emissions()
