#!/bin/bash

INPUT_FOLDER="reemission_demo_delineations"
INPUT_FOLDER_LINK="https://drive.google.com/file/d/1PYqzy4-5P2aW8tvYZPPJ-3fDSDUoHgOv/view?usp=drive_link"
TARGET_INPUT_FOLDER_SIZE=2368  # Specify the target size in kbytes
IFC_DB_FOLDER="reemission_demo_dam_db"
IFC_DB_LINK="https://drive.google.com/file/d/1OZAVdRMOQN8J-7h3bZIMeQzIfBuo5adC/view?usp=drive_link"
IFC_DB_SIZE=72 # Size in kbytes

echo
echo "RUNNING CALCULATIONS FOR A SUBSET OF EXISTING AND FUTURE HYDROELECTRIC RESERVOIRS IN MYANMAR..."
echo

echo 
echo "1. Fetching the demo dabase of dams from external sources..."
echo 
# Download ifc database if data does not exist or data size is different from target.
if [ -d "$IFC_DB_FOLDER" ]; then
    folder_size=$(du -s "$IFC_DB_FOLDER" | awk '{print $1}')
    if [ "$folder_size" -eq "$IFC_DB_SIZE" ]; then
        echo "The DAMS database folder $IFC_DB_FOLDER exists and has the correct size."
        echo "Fetching the dam database from extrnal sources not required."
    else
        echo "The DAMS database in $IFC_DB_FOLDER exists but its size is not $IFC_DB_SIZE bytes."
        echo "Downloading the database form external sources. Please Wait..."
        python fetch_inputs.py $IFC_DB_LINK -o $IFC_DB_FOLDER/"reemission_demo_dam_db.zip"
    fi
else
    echo "The DAMS database folder $IFC_DB_FOLDER does not exist."
    mkdir "$IFC_DB_FOLDER"
    echo "Downloading the database form external sources. Please Wait..."
    python fetch_inputs.py $IFC_DB_LINK -o $IFC_DB_FOLDER/"reemission_demo_dam_db.zip"
fi

echo 
echo "2. Fetching reservoir and catchment delineations from external sources..."
echo 
# Download input data if data does not exist or data size is different from target.
if [ -d "$INPUT_FOLDER" ]; then
    folder_size=$(du -s "$INPUT_FOLDER" | awk '{print $1}')
    if [ "$folder_size" -eq "$TARGET_INPUT_FOLDER_SIZE" ]; then
        echo "The input folder $INPUT_FOLDER exists and has the correct size."
        echo "Fetching input files from extrnal sources not required."
    else
        echo "The input folder $INPUT_FOLDER exists but its size is not $TARGET_INPUT_FOLDER_SIZE bytes."
        echo "Downloading the input files form external sources. Please Wait..."
        python fetch_inputs.py $INPUT_FOLDER_LINK -o $INPUT_FOLDER/"reemission_demo_delineations.zip"
    fi
else
    echo "The input folder $INPUT_FOLDER does not exist."
    mkdir "$INPUT_FOLDER"
    echo "Downloading the input files form external sources. Please Wait..."
    python fetch_inputs.py $INPUT_FOLDER_LINK -o $INPUT_FOLDER/"reemission_demo_delineations.zip"
fi

OUTPUTS_FOLDER="heet_outputs"
echo
echo "3. Creating the outputs folder $OUTPUTS_FOLDER ..."
echo
# Create outputs folder if it does not exist already
if [ ! -d "$OUTPUTS_FOLDER" ]; then
    mkdir -p "$OUTPUTS_FOLDER"
    echo "Folder created: $OUTPUTS_FOLDER"
else
    echo "Folder already exists: $OUTPUTS_FOLDER"
fi

echo
echo "4. Merging tabular data into a single CSV file and saving to $OUTPUTS_FOLDER/heet_outputs.csv ..."
echo
# Pre-process the input data
SHP_FOLDERS=(
    "$INPUT_FOLDER/batch_1"
    "$INPUT_FOLDER/batch_2"
    "$INPUT_FOLDER/batch_3"
    "$INPUT_FOLDER/batch_4"
)
# echo ${SHP_FOLDERS[1]}
COMBINED_CSV_FILE="$OUTPUTS_FOLDER/heet_outputs.csv"
command_1="reemission-heet process-tab-outputs"
for input_folder in "${SHP_FOLDERS[@]}"; do
  command_1+=" -i $input_folder/output_parameters.csv"
done
command_1+=" -o $COMBINED_CSV_FILE"
command_1+=" -cv 'c_treatment_factor' 'primary (mechanical)'"
command_1+=" -cv 'c_landuse_intensity' 'low intensity'"
command_1+=" -cv 'type' 'unknown'"
eval $command_1

echo
echo "5. Merging shape files for individual reservoirs into combined shape files..."
echo
# Convert the csv file into a JSON input file to RE-Emission
# Write reemission tab2json CLI function
# Join shape files for individual reservoirs into combined shapes for each category of shapes
command_2="reemission-heet join-shapes"
for input_folder in "${SHP_FOLDERS[@]}"; do
  command_2+=" -i $input_folder"
done
command_2+=" -o $OUTPUTS_FOLDER \-gp 'R_*.shp, C_*.shp, MS_*.shp, PS_*.shp' -f 'reservoirs.shp, catchments.shp, rivers.shp, dams.shp'"
eval $command_2

echo
echo "6. Converting HEET tabular data to the RE-Emission input JSON file"
echo
reemission-heet tab-to-json -i $COMBINED_CSV_FILE -o $OUTPUTS_FOLDER/"reemission_inputs.json"

REEMISSION_OUTPUTS_FOLDER="reemission_outputs"
echo
echo "7. Creating the outputs folder $REEMISSION_OUTPUTS_FOLDER ..."
echo
# Create outputs folder if it does not exist already
if [ ! -d "$REEMISSION_OUTPUTS_FOLDER" ]; then
    mkdir -p "$REEMISSION_OUTPUTS_FOLDER"
    echo "Folder created: $REEMISSION_OUTPUTS_FOLDER"
else
    echo "Folder already exists: $REEMISSION_OUTPUTS_FOLDER"
fi


echo
echo "8. Calculating GHG emissions with RE-EMISSION"
echo
# Estimate gas emissions and save output files
reemission calculate $OUTPUTS_FOLDER/"reemission_inputs.json" -a "Default User" \
    -t "Demo Example Results" \
    -o $REEMISSION_OUTPUTS_FOLDER/demo_GHG_outputs.pdf \
    -o $REEMISSION_OUTPUTS_FOLDER/demo_GHG_outputs.json \
    -o $REEMISSION_OUTPUTS_FOLDER/demo_GHG_outputs_.xlsx

# Merge results into shape files and visualise on a map
echo
echo "9. Merging input and output data into shape files"
echo
python postprocess_results.py

echo
echo "DONE"
