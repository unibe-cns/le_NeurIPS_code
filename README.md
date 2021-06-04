# le_NeurIPS_code


### Code for Dendritic Microcircuits with and without LE (Fig.3 and 5)

The code can be found in `fig3fig5_dendritic_microcircuits/src/`.

The experiments are configured using config files.
All config files required for the production of the plotted results are in `fig3fig5_dendritic_microcircuits/experiment_configs/`.
The naming scheme of the config files is as follows `{task name}_{with LE or not}_tpres_{tpres in unit dt}.yaml` where `task name` is `bars` (Fig.3) or `mimic` (Fig.5) and `with LE or not` is either `le` or `orig`.

For each run the results will be saved in `fig3fig5_dendritic_microcircuits/experiment_results/{config file name}_{timestamp}/`.

**To run an experiment:**
```
cd fig3fig5_dendritic_microcircuits/src/
python3 run_bars.py train ../experiment_configs/{chosen_config_file}
```
For the experiment in Fig.5 replace `run_bars.py` with `run_single_mc.py`

**To plot the results of a run:**
```
cd fig3fig5_dendritic_microcircuits/src/
python3 run_bars.py eval ../experiment_results/{results_dir_of_run_to_be_evaluated}
```
This will generate plots of the results (depending on how many variables you configured to be recorded, more or less plots can be generated).
Which plots are plotted is defined in `run_X.py`

**Reproduce all data needed for Fig3:**

For the results shown in Fig.3 all config files with the name `bars_*.yaml` need to be run for 10 different seeds (configurable in the config file).
The seeds chosen for the paper were `[12345, 12346, 12347, 12348, 12349, 12350, 12351, 12352, 12353]`.
