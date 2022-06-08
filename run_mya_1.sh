#!/bin/bash
echo Running calculation for Burmese existing dams with RE-Emission ...
echo ---------------------------------------------
reemission calculate examples/MYA_inputs_existing.json -w latex -w json -o examples/MYA_outputs_existing.pdf -o examples/MYA_outputs_existing.json
