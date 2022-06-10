#!/bin/bash
echo Running test calculation with RE-Emission ...
echo ---------------------------------------------
reemission calculate ./tests/test_data/inputs.json -o cli_output.json -o cli_output.pdf
