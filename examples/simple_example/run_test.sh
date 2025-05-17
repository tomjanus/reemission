#!/bin/bash

echo "Running test calculation with RE-Emission ..."
echo "---------------------------------------------"
echo "Reads input file in .json format, calculates GHG emissions for a number of reservoirs."
echo "Outputs are saved only in the formats you specify: JSON, PDF, or XLSX."
echo "---------------------------------------------"

# Ensure at least one argument was passed
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [output.json] [output.pdf] [output.xlsx]"
    exit 1
fi

folder="outputs"

if [ ! -d "$folder" ]; then
    mkdir -p "$folder"
    echo "Folder created: $folder"
else
    echo "Folder already exists: $folder"
fi

# Base command
cmd="reemission calculate test_input.json"

# Add output arguments
for output in "$@"; do
    ext="${output##*.}"
    case "$ext" in
        json|pdf|xlsx)
            cmd+=" -o ./outputs/${output}"
            ;;
        *)
            echo "Warning: Unsupported output format: $ext (skipped)"
            ;;
    esac
done

# Add other fixed arguments
cmd+=" -a \"RE-Emission test run\" -t \"Test Results\" -p g-res"

# Run the command
eval "$cmd"

