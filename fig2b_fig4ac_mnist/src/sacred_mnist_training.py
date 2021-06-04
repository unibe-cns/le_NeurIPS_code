#!/usr/bin/env python3
# train MNIST with latent equilibrium model
# and track experiment using sacred

import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('layeredMNIST')
ex.observers.append(FileStorageObserver('runs'))

# configuration
@ex.config
def config():
    # choose dataset
    # either mnist_784 or Fashion-MNIST
    dataset = 'mnist_784'

ex.add_config('defaults.json')

# hook for logging metrics to sacred
def after_epoch_hook(network, run):
  # metrics to track during experiment
    metrics = ['val_loss', 'val_accuracy', 'val_error',
               'test_loss', 'test_accuracy', 'test_error']

    logs = network.logs
    for key in metrics:
        if key in logs:
            run.log_scalar(key, logs[key])
        else:
            print(f'Warning: Metric "{key}" not found in logs. Skipping...')

# initialize and train the network
@ex.automain
def run(_run, _config, _seed):

    from le_layers_mnist_training import MnistTrainer
    from model.network_params import LayeredParams

    params = LayeredParams()
    params.load_params_from_dict(_config)

    import model.latent_equilibrium_layers as nn
    import model.layered_torch_utils as tu

    fc1 = nn.Linear(# this are the only things that should remain to be set here
        28 * 28, 300,
        tu.hard_sigmoid,
        tu.hard_sigmoid_deriv,
        params,
    )
    fc2 = nn.Linear(
        300, 100,
        tu.hard_sigmoid,
        tu.hard_sigmoid_deriv,
        params,
    )
    fc3 = nn.Linear(
        100, _config['classes'],
        tu.linear,
        tu.linear_deriv,
        params,
    )

    network = nn.LESequential(
        [fc1, fc2, fc3],
        params,
    )

    trainer = MnistTrainer(
        classes=_config['classes'],
        train_samples=_config['train_samples'],
        val_samples=_config['val_samples'],
        test_samples=_config['test_samples'],
        n_updates=_config['n_updates'],
        epoch_hook=after_epoch_hook,
    )

    net, test_acc = trainer.train(
        network,
        run=_run,
        batch_size=_config['batch_size'],
        epoch_qty=_config['epoch_qty'],
        with_optimizer=_config['with_optimizer'],
        verbose=1,
    )

    return test_acc
