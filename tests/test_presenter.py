""" Tests for the Presenter and Writer classes """
import os
from typing import Optional
import unittest
import json
from reemission.presenter import Presenter, LatexWriter, ExcelWriter
from reemission.input import Inputs

module_dir = os.path.dirname(__file__)
input_json_file = os.path.abspath(
    os.path.join(module_dir, 'test_data', 'inputs.json'))
output_json_file = os.path.abspath(
    os.path.join(module_dir, 'test_data', 'outputs.json'))
output_tex_file = os.path.abspath(
    os.path.join(module_dir, 'test_data', 'output.tex'))
output_xls_file = os.path.abspath(
    os.path.join(module_dir, 'test_data', 'output.xlsx'))


class TestPresenter(unittest.TestCase):
    """ Class for testing the Presenter functionality """
    input_file_path: str = input_json_file
    output_file_path: str = output_json_file
    inputs: Optional[Inputs] = None
    outputs: Optional[dict] = None

    @classmethod
    def setUpClass(cls):
        cls.inputs = Inputs.fromfile(cls.input_file_path)
        with open(cls.output_file_path, "r", encoding="utf-8") as json_file:
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
                              output_file=output_tex_file)
        pres_latex.output()

    def test_excel(self):
        """ Test writing output data to .xlsx using Pandas """
        pres_xls = Presenter(inputs=self.inputs, outputs=self.outputs,
                             author="Gallus Anonymus",
                             title="HEET Test Results")
        pres_xls.add_writer(writer=ExcelWriter,
                            output_file=output_xls_file)
        pres_xls.output()


if __name__ == '__main__':
    unittest.main()
