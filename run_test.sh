#!/bin/bash
echo Running test calculation with RE-Emission ...
echo ---------------------------------------------
reemission calculate ./examples/inputs.json -o ./examples/test_output.json -o ./examples/test_output.pdf -o ./examples/test_output.xlsx
