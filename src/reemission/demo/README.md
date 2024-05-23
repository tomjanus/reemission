## Demo
This is a short example of how `RE-EMISSION` works in combintion with an `upstream` reservoir and catchment delineation tool `GeoCARET`. This example showcases `RE-EMISSION`'s integration capabilities with `GeoCARET` on a small example from a larger real-life case study on estimating GHG emissions from existing and planned hydroelectric reservoirs in Myanmar. The demo runs in the following steps:
* Merging multiple tabular data files from several batches of reservoir delineations in `GeoCARET` into a single `CSV` file.
* Merging shape files for individual reservoirs and catchments into combined shape files representing reservoirs and catchments for all items in the study.
* Converting the merged tabular data from `GeoCARET` into `RE-EMISSION` JSON input file.
* Calculating GHG emissions with RE-EMISSION taking the JSON input file generated in the previous step.
* Pasting selected `GeoCARET` tabular input data and `RE-EMISSIONS` output data into the combined shape files for reservoirs and dams and presenting the updated shape files in the form of an interactive map using `Folium`.

Alternatively you can run the demo from command-line after installing the `RE-EMISSION` package:
```sh
reemission run-demo [working-directory]
```
`working-directory` is a folder in which all the inputs and outpus for the demo will be stored. If the folder structure does not exists it will be created.
The demo relies on several input datasets originating form `GeoCARET`. If not already in your `working-directory` they will be automatically downloaded. This step requires working internet connection.

If all goes well the demo should run as follows:
![demo-22-05-24-compressed](https://github.com/tomjanus/reemission/assets/8837107/b101e9d0-ac60-4f21-bbeb-a8a8ae85522b)
