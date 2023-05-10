"""Process parsed tabular data from HEET and create RE-EMISSION input JSON
files"""
from __future__ import annotations
import pathlib
from typing import Literal, List, Any, Dict, Union, Protocol, Optional
from pydantic import ValidationError
from dataclasses import dataclass
import pandas as pd
import json
from reemission.utils import get_package_file
from reemission.integration.heet.heet_tab_parser import TabularHeetOutput
from reemission.data_models.input_model import InputModel
from reemission.integration.heet.input_model_heet import \
    DamDataModelHeet, BuildStatusModelHeet, \
    CatchmentModelHeet, ReservoirModelHeet, BiogenicFactorsModelHeet
from reemission.integration.heet.custom_exceptions import CompositeModelValidationException
from reemission.app_logger import create_logger


logger = create_logger(logger_name=__name__)


BuildStatus = Literal["existing", "future"]
GasName = Literal["co2", "ch4", "n2o"]


class SavingStrategy(Protocol):
    """Translation of the input model into a dictionary used for creating
    input JSON files for reemission."""
    def __call__(self, input_model: InputModel) -> Dict:
        ...


@dataclass
class LegacySavingStrategy:
    """Input model saver conforming to the input model layout from v.1.0
    of RE-Emission"""
    year_vector: Optional[List[int]] = None
    gasses: Optional[List[GasName]] = None

    def __post_init__(self) -> None:
        """ """
        if self.year_vector is None:
            self.year_vector = [1, 5, 10, 20, 30, 40, 50, 65, 80, 100]
        if self.gasses is None:
            self.gasses = ["co2", "ch4", "n2o"]

    def __call__(self, input_model: InputModel) -> Dict:
        reservoir_name = input_model.dam_data.name
        legacy_input: Dict[str, Any] = {}
        legacy_input["coordinates"] = [
            input_model.dam_data.latitude, input_model.dam_data.longitude]
        legacy_input["monthly_temps"] = input_model.dam_data.monthly_temps
        legacy_input["year_vector"] = self.year_vector
        legacy_input["gasses"] = self.gasses
        # Compose the input dictionary such that it adheres to the RE-EMISSION
        # input data format
        reservoir = input_model.reservoir.dict()
        catchment = input_model.catchment.dict()
        biogenic = input_model.biogenic_factors.dict()
        catchment['biogenic_factors'] = biogenic
        legacy_input['catchment'] = catchment
        legacy_input['reservoir'] = reservoir
        return {reservoir_name: legacy_input}


@dataclass
class InputToDictConverter:
    """Conversion of HEET output data to RE-Emission input structure."""
    input: InputModel

    @classmethod
    def from_row(
            cls, row: pd.Series, r_status: BuildStatus, 
            r_construction_year: Union[str, int]) -> InputToDictConverter:
        """Instantiate InputModel from a row of tabular data output from HEET"""
        # Since objects are instantiated one by one, we need to manually accumulate
        # Pydantic's ValidationErrors and throw and re-throw validation error with
        # one big compound error message
        model_caller = {"dam_data": DamDataModelHeet,
                        "build_status": BuildStatusModelHeet,
                        "catchment": CatchmentModelHeet,
                        "reservoir": ReservoirModelHeet,
                        "biogenic_factors": BiogenicFactorsModelHeet}
        error_stack = ""
        input_model_data = {}
        for field_name, model in model_caller.items():
            try:
                # TODO: FIX THIS IF STATEMENT, CHECK OUT CUSTOM INITS FOR
                #       PYDANTIC MODELS
                if field_name == "build_status":
                    input_model_data[field_name] = model.from_row(
                        row, r_status, r_construction_year).dict()
                else:
                    input_model_data[field_name] = model.from_row(row).dict()
            except ValidationError as val_err:
                error_stack += val_err.json()
        if error_stack:
            err = CompositeModelValidationException(msg=error_stack)
            raise err
        input_model = InputModel(**input_model_data)
        return cls(input_model)

    def to_dict(self, saving_strategy: SavingStrategy) -> Dict:
        """Save the input model to JSON"""
        return saving_strategy(self.input)


@dataclass
class TabToJSONConverter:
    """ """
    tab_data_file: pathlib.Path
    saving_strategy: SavingStrategy

    def to_json(self, output_file: pathlib.Path) -> None:
        """ """
        out_dict = {}
        heet_output = TabularHeetOutput.from_csv(self.tab_data_file)
        for _, row in heet_output.data.iterrows():
            try:
                input_model = InputToDictConverter.from_row(row, 'existing', 2000)
                out_dict.update(input_model.to_dict(self.saving_strategy))
            except (CompositeModelValidationException, ValidationError) as val_err:
                err_msg_1 = \
                    f"Cannot parse input data for reservoir '{row['name']}'. "
                err_msg_2 = \
                    f"Encountered the following parsing problem(s):\n {val_err.msg}"
                logger.warning(err_msg_1 + err_msg_2)
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(out_dict, outfile, indent=4)


if __name__ == "__main__":
    """ """
    heet_output = get_package_file("../../input_data/all_heet_outputs.csv")
    reemission_input = get_package_file("../../input_data/reemission_input.json")
    converter = TabToJSONConverter(heet_output, LegacySavingStrategy())
    converter.to_json(reemission_input)