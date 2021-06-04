#!/bin/bash -l

for seed in 42 69 12345  # 98765 38274 28374 42848 48393 83475 57381
do
  for updates in 20 50 100 200 500 1000
  do

    # run with dt=0.01
    for tau_s in 0.0 0.1
    do
      srun --ntasks 1 python src/sacred_mnist_training.py with 4c_params.json \
        dt=0.01 tau_s=$tau_s seed=$seed n_updates=$updates &
    done

    # run with dt=0.1
    for tau_s in 0.0 0.5 2.0
    do
      srun --ntasks 1 python src/sacred_mnist_training.py with 4c_params.json \
        tau_s=$tau_s seed=$seed n_updates=$updates &
    done

  done
done

wait
