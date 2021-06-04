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
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 1
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 2
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 3
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 5
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 7
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 8
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 13
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 21
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 34
#./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 55
python -u ./experiments/CIFAR10/le_layers_cifar10_training.py --n_updates 50 --seed 1 

# BP runs
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 1
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 2
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 3
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 5
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 7
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 8
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 13
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 21
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 34
#./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 55
python -u ./experiments/CIFAR10/bp_layers_cifar10_training.py --epochs 100 --seed 1

