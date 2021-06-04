#!/bin/bash -l

for seed in 42 69 12345 38274 # 28374 42848 48393 83475 57381
do
  for dt in 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0
  do
    srun --ntasks 1 python src/sacred_mnist_training.py with 4a_params.json \
      tau_width=0.0 seed=$seed dt=$dt &

    srun --ntasks 1 python src/sacred_mnist_training.py with 4a_params.json \
      tau_width=0.2 seed=$seed dt=$dt &

    srun --ntasks 1 python src/sacred_mnist_training.py with 4a_params.json \
      tau_width=0.3 seed=$seed dt=$dt &

    # with tau_r errors and 1% noise on tau
    srun --ntasks 1 python src/sacred_mnist_training.py with 4a_params.json \
      adapt_tau=False prosp_rate_errors=True tau_width=0.01 seed=$seed dt=$dt &

    # learning time constants
    srun --ntasks 1 python src/sacred_mnist_training.py with 4a_params.json \
      adapt_tau=True prosp_rate_errors=True tau_width=0.2 seed=$seed dt=$dt &

  done
done

wait
