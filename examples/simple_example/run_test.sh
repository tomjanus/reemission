#!/bin/bash
echo Running test calculation with RE-Emission ...
echo ---------------------------------------------
echo Reads input file in .json format, calculates GHG emissions for a number of reservoirs and outputs the results in three file fomrats: JSON, PDF, and XLSX. Saving ouptuts to XLSX format is still experimental.
echo ---------------------------------------------

folder="outputs"

if [ ! -d "$folder" ]; then
    mkdir -p "$folder"
    echo "Folder created: $folder"
else
    echo "Folder already exists: $folder"
fi

reemission calculate test_input.json -o ./outputs/test_output.json -o ./outputs/test_output.pdf -o ./outputs/test_output.xlsx -a "RE-Emission test run" -t "Test Results" -p g-res
