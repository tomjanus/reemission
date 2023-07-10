@echo off

echo "Running" "test" "calculation" "with" "RE-Emission" "..."
echo "---------------------------------------------"
echo "Reads" "input" "file" "in" ".json" "format," "calculates" "GHG" "emissions" "for" "a" "number" "of" "reservoirs" "and" "outputs" "the" "results" "in" "three" "file" "fomrats:" "JSON," "PDF," "and" "XLSX." "Saving" "ouptuts" "to" "XLSX" "format" "is" "still" "experimental."
echo "---------------------------------------------"
SET "folder=outputs"
IF "!" "-d" "%folder%" (
  mkdir "-p" "%folder%"
  echo "Folder created: %folder%"
) ELSE (
  echo "Folder already exists: %folder%"
)
reemission "calculate" "test_input.json" "-o" "%CD%\outputs\test_output.json" "-o" "%CD%\outputs\test_output.pdf" "-o" "%CD%\outputs\test_output.xlsx" "-a" "RE-Emission test run" "-t" "Test Results" "-p" "g-res"
