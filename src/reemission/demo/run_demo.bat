@echo off

set "INPUT_FOLDER=reemission_demo_delineations"
set "INPUT_FOLDER_LINK=https://drive.google.com/file/d/1PYqzy4-5P2aW8tvYZPPJ-3fDSDUoHgOv/view?usp=drive_link"
set "TARGET_INPUT_FOLDER_SIZE=2368"
set "IFC_DB_FOLDER=reemission_demo_dam_db"
set "IFC_DB_LINK=https://drive.google.com/file/d/1OZAVdRMOQN8J-7h3bZIMeQzIfBuo5adC/view?usp=drive_link"
set "IFC_DB_SIZE=72"

echo.
echo "RUNNING CALCULATIONS FOR A SUBSET OF EXISTING AND FUTURE HYDROELECTRIC RESERVOIRS IN MYANMAR..."
echo.

echo.
echo "1. Fetching the demo database of dams from external sources..."
echo.
REM Download ifc database if data does not exist or data size is different from target.
if exist "%IFC_DB_FOLDER%" (
    for /f %%A in ('dir /s /b "%IFC_DB_FOLDER%\*"') do (
        set "folder_size=%%~zA"
        if %folder_size% EQU %IFC_DB_SIZE% (
            echo The DAMS database folder %IFC_DB_FOLDER% exists and has the correct size.
            echo Fetching the dam database from external sources not required.
        ) else (
            echo The DAMS database in %IFC_DB_FOLDER% exists but its size is not %IFC_DB_SIZE% bytes.
            echo Downloading the database from external sources. Please Wait...
            python fetch_inputs.py %IFC_DB_LINK% -o "%IFC_DB_FOLDER%\reemission_demo_dam_db.zip"
        )
    )
) else (
    echo The DAMS database folder %IFC_DB_FOLDER% does not exist.
    mkdir "%IFC_DB_FOLDER%"
    echo Downloading the database from external sources. Please Wait...
    python fetch_inputs.py %IFC_DB_LINK% -o "%IFC_DB_FOLDER%\reemission_demo_dam_db.zip"
)

echo.
echo "2. Fetching reservoir and catchment delineations from external sources..."
echo.
REM Download input data if data does not exist or data size is different from target.
if exist "%INPUT_FOLDER%" (
    for /f %%A in ('dir /s /b "%INPUT_FOLDER%\*"') do (
        set "folder_size=%%~zA"
        if %folder_size% EQU %TARGET_INPUT_FOLDER_SIZE% (
            echo The input folder %INPUT_FOLDER% exists and has the correct size.
            echo Fetching input files from external sources not required.
        ) else (
            echo The input folder %INPUT_FOLDER% exists but its size is not %TARGET_INPUT_FOLDER_SIZE% bytes.
            echo Downloading the input files from external sources. Please Wait...
            python fetch_inputs.py %INPUT_FOLDER_LINK% -o "%INPUT_FOLDER%\reemission_demo_delineations.zip"
        )
    )
) else (
    echo The input folder %INPUT_FOLDER% does not exist.
    mkdir "%INPUT_FOLDER%"
    echo Downloading the input files from external sources. Please Wait...
    python fetch_inputs.py %INPUT_FOLDER_LINK% -o "%INPUT_FOLDER%\reemission_demo_delineations.zip"
)

set "OUTPUTS_FOLDER=gecaret_outputs"
echo.
echo "3. Creating the outputs folder %OUTPUTS_FOLDER% ..."
echo.
REM Create outputs folder if it does not exist already
if not exist "%OUTPUTS_FOLDER%" (
    mkdir "%OUTPUTS_FOLDER%"
    echo Folder created: %OUTPUTS_FOLDER%
) else (
    echo Folder already exists: %OUTPUTS_FOLDER%
)

echo.
echo "4. Merging tabular data into a single CSV file and saving to %OUTPUTS_FOLDER%/geocaret_outputs.csv ..."
echo.
REM Pre-process the input data
set "SHP_FOLDERS=%INPUT_FOLDER%\batch_1 %INPUT_FOLDER%\batch_2 %INPUT_FOLDER%\batch_3 %INPUT_FOLDER%\batch_4"
set "COMBINED_CSV_FILE=%OUTPUTS_FOLDER%\geocaret_outputs.csv"
set "command_1=reemission-geocaret process-tab-outputs"
for %%F in (%SHP_FOLDERS%) do (
    set "command_1=!command_1! -i %%F\output_parameters.csv"
)
set "command_1=!command_1! -o %COMBINED_CSV_FILE%"
%command_1%

echo.
echo "5. Merging shape files for individual reservoirs into combined shape files..."
echo.
REM Convert the csv file into a JSON input file to RE-Emission
REM Write reemission tab2json CLI function
REM Join shape files for individual reservoirs into combined shapes for each category of shapes
set "command_2=reemission-geocaret join-shapes"
for %%F in (%SHP_FOLDERS%) do (
    set "command_2=!command_2! -i %%F"
)
set "command_2=!command_2! -o %OUTPUTS_FOLDER% -gp 'R_*.shp, C_*.shp, MS_*.shp, PS_*.shp' -f 'reservoirs.shp, catchments.shp, rivers.shp, dams.shp'"
%command_2%

echo.
echo "6. Converting GeoCARET tabular data to the RE-Emission input JSON file"
echo.
reemission-geocaret tab-to-json -i %COMBINED_CSV_FILE% -o %OUTPUTS_FOLDER%/reemission_inputs.json

set "REEMISSION_OUTPUTS_FOLDER=reemission_outputs"
echo.
echo "7. Creating the outputs folder %REEMISSION_OUTPUTS_FOLDER% ..."
echo.
REM Create outputs folder if it does not exist already
if not exist "%REEMISSION_OUTPUTS_FOLDER%" (
    mkdir "%REEMISSION_OUTPUTS_FOLDER%"
    echo Folder created: %REEMISSION_OUTPUTS_FOLDER%
) else (
    echo Folder already exists: %REEMISSION_OUTPUTS_FOLDER%
)

echo.
echo "8. Calculating GHG emissions with RE-EMISSION"
echo.
REM Estimate gas emissions and save output files
reemission calculate %OUTPUTS_FOLDER%/reemission_inputs.json -a "Default User" ^
    -t "Demo Example Results" ^
    -o %REEMISSION_OUTPUTS_FOLDER%/demo_GHG_outputs.pdf ^
    -o %REEMISSION_OUTPUTS_FOLDER%/demo_GHG_outputs.html ^
    -o %REEMISSION_OUTPUTS_FOLDER%/demo_GHG_outputs.json ^
    -o %REEMISSION_OUTPUTS_FOLDER%/demo_GHG_outputs_.xlsx

echo.
echo "9. Merging input and output data into shape files"
echo.
python postprocess_results.py

echo.
echo "DONE"

