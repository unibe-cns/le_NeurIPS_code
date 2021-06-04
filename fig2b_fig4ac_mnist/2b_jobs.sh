#!/bin/bash -l

for seed in 42 69 12345 98765 38274 28374 42848 48393 83475 57381
do

  # vanilla
  srun --ntasks 1 python src/sacred_mnist_training.py with 2b_params.json \
    seed=$seed &

  # feedback alignment
  srun --ntasks 1 python src/sacred_mnist_training.py with 2b_params.json \ 
    feedback_alignment=True seed=$seed &

  # big tau
  srun --ntasks 1 python src/sacred_mnist_training.py with 2b_params.json \
    tau=200 seed=$seed &

  # without prospective coding
  srun --ntasks 1 python src/sacred_mnist_training.py with 2b_params.json \
    dt=1.0 tau=0 tau_m=10 n_updates=200 prosp_rate_errors=True seed=$seed &

done

wait
