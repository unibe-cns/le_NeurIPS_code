# le_NeurIPS_code

### Code for FC MNIST experiments (Fig.2b and 4ac)
The code can be found in `fig2b_fig4ac_mnist/src/`.

Different experiments are configured using a different config files for each
panel. There is also a bash script that starts all the necessary experiments for
different seeds, loading the corresponding parameters and/or specifing different
parameters for different runs (careful since these scripts might hundreds of
jobs at once to be executed on a cluster.)

**Running the experiments:**
For example, in order to run all the experiments needed to reproduce Fig. 2b,
execute:
```
cd fig2b_fig4ac_mnist/src/
/bin/bash 2b_jobs.sh
```

The results of each run will be saved in 
`fig2b_fig4ac_mnist/runs/{run_number}/`.

For the experiment in Fig.4 replace `2b_jobs.sh` with `4a_jobs.sh` or
`4c_jobs.sh` respectively


### Code for HIGGS, MNIST and CIFAR10 with and without LE (Fig. 2cde).

The code can be found in `fig2cde_higgs_mnist_cifar10`.

The code configuration is integrated into the main files and only a few parameters are configured via argparse.

To run the code, check the respective `submit_python_*_v100.sh` file which contains examples and all run configurations for all seeds used.

The seeds for all runs are always 1, 2, 3, 5, 7, 8, 13, 21, 34. (Fibonacci + lucky number 7), resulting in 9 seeds for each experiment.

Results can be found in the respective log file produced from the std out of the running code via `python -u *_training.py > file.log`.

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
