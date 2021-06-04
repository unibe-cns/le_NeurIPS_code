#!/bin/bash
#SBATCH -J LE_experiment
#SBATCH -o cluster/log/%j_LE_std.log
#SBATCH -e cluster/log/%j_LE_err.log
#SBATCH -t 96:00:00
#SBATCH -p gpus
#SBATCH -A insel
#SBATCH --gres=gpu:v100:1
# BATCH --gres=gpu:1080ti:1
# BATCH --gres=gpu:2080ti:1

# $1=<PYTHON_SCRIPT> $2=<PARAMS>
# echo $1 $2

source ./miniconda3/bin/activate latenteq # source venv3/bin/activate
python -u -c "import os;print(os.getcwd())"

#python -u $1 $2

# LE runs
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 1
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 2
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 3
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 5
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 7
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 8
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 13
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 21
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 34
#./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 55
python -u ./experiments/HIGGS/le_layers_higgs_training.py --epochs 100 --n_updates 20 --seed 1

# BP runs
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 1
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 2
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 3
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 5
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 7
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 8
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 13
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 21
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 34
#./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 55
python -u ./experiments/HIGGS/bp_layers_higgs_training.py --epochs 100 --seed 1

