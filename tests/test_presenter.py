""" Tests for the Presenter and Writer classes """
from typing import Optional
import sys
sys.path.append("..")
import unittest
import json
from src.emissions.presenter import Presenter, LatexWriter
from src.emissions.input import Inputs


class TestPresenter(unittest.TestCase):
    """ Class for testing the Presenter functionality """
    input_file_path: str = './test_data/inputs.json'
    output_file_path: str = './test_data/outputs.json'
    inputs: Optional[Inputs] = None
    outputs: Optional[dict] = None

    @classmethod
    def setUpClass(cls):
        cls.inputs = Inputs.fromfile(cls.input_file_path)
        with open(cls.output_file_path, "r") as json_file:
            cls.outputs = json.load(json_file)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_presenter_init(self):
        """ Check if initializations from dict and json produce the same data
            sets """
        pres1 = Presenter(inputs=self.inputs, outputs=self.outputs,
                          author="Anonymous 1", title="Sample Title")
        pres2 = Presenter.fromfiles(
            input_file=self.input_file_path, output_file=self.output_file_path,
            author="Anonymous 2", title="Sample Title 2")
        self.assertEqual(pres1.outputs, pres2.outputs)

    def test_presenter_config(self):
        """ Check Presenter config data load in post init """
        pres1 = Presenter(inputs=self.inputs, outputs=self.outputs)
        self.assertIsInstance(pres1.input_config, dict)

    def test_latex(self):
        """ Test writing output data to .tex / .pdf using LatexWriter """
        pres_latex = Presenter(inputs=self.inputs, outputs=self.outputs,
                               author="Gallus Anonymus",
                               title="HEET Test Results")
        pres_latex.add_writer(writer=LatexWriter,
                              output_file='./test_data/output.tex')
        pres_latex.output()


if __name__ == '__main__':
    unittest.main()
