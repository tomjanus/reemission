"""Script for post-processing and visualisation of results for the Myanmar
case study"""
import pathlib
from reemission.utils import load_toml, load_shape, get_package_file
from reemission.postprocessing.data_processing import (
    append_data_to_shapes, ExtractFromJSON, TabToShpCopy)
from reemission.postprocessing.visualise import FoliumOutputMapper


config = load_toml(get_package_file('config/heet.toml'))


def process_mya_case_study_results() -> None:
    """Process (merge, clean) and visualise data for the demo case study"""
    # 1. Create shape files with extra fields with values obtained from the
    #    tabular HEET output csv file
    shp_folder = pathlib.Path("heet_outputs")
    data_file = shp_folder / "heet_outputs.csv"
    append_data_to_shapes(shp_folder, data_file, config)
    # 2. Extract data from RE-Emission output file and save it into reservoirs
    #    shape file
    output_json_file =pathlib.Path(
        "reemission_outputs/demo_GHG_outputs.json")
    reemission_outputs_df = \
        ExtractFromJSON.from_file(output_json_file).extract_outputs()
    data_columns = list(reemission_outputs_df.columns)
    data_columns.remove("name")
    # 3. "Feed" the emission data into the reservoirs shape.
    res_shape_file = shp_folder / "reservoirs_updated.shp"
    re_to_res = TabToShpCopy(
        shp_data=load_shape(res_shape_file),
        tab_data=reemission_outputs_df)
    re_to_res.transfer(
        source_key_column="name", target_key_column="name", fields=data_columns)
    re_to_res.save_shp(res_shape_file)
    # 4. Create the interactive map and save to a folder
    map_path = pathlib.Path("demo_interactive_map")
    map_path.mkdir(parents=True, exist_ok=True)

    updated_reservoirs = re_to_res.shp_data
    ifc_dams = load_shape(
        pathlib.Path("reemission_demo_dam_db")/"dam_db.shp")
    mapper = FoliumOutputMapper(updated_reservoirs, ifc_dams, location=[18.049883, 97.080650], init_zoom=9)
    mapper.create_map()
    mapper.save_map(map_path / "index.html", show=True)

if __name__ == "__main__":
    """ """
    process_mya_case_study_results()
