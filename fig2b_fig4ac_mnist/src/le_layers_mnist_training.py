#!/usr/bin/env python3
# train MNIST with layered latent equilibrium implementation

import random
import numpy as np
import datetime
import argparse

import sys

from model.network_params import ModelVariant, TargetType

import torch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from utils.torch_utils import SimpleDataset

import model.latent_equilibrium_layers as nn
import model.layered_torch_utils as tu
from utils.datasets import MnistDataset

def get_prediction_accuracy(y, y_hat):
    """
    Test accuracy of predictions (average number of correct labels).

    Args:
        y: true labels.
        y_hat: predicted labels.

    Returns:
    """

    class_rates = []
    for (predicted_label, true_label) in zip(y_hat, y):  # argmax of true labels should be equal to argmax of distorted/decayed labels
        class_rates.append(np.argmax(true_label) == np.argmax(predicted_label))

    return np.mean(class_rates)


def get_confusion_matrix(y_hat, y):
    """
    Get confusion matrix to understand problematic confusions.
    """
    c = confusion_matrix(y_hat, y)
    return c


class MnistTrainer:

    def __init__(self, classes: int=10, train_samples: int=1000, val_samples: int=100,
                 test_samples: int=100, n_updates: int=10, epoch_hook=None):
        """ Initialize the mnist trainer.

        Args:
            params: Network parameters for a new network.
            classes: Number of MNIST classes for labels.
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
            epoch_hook: Function hook executed after every epoch
        """
        self.epoch_hook = epoch_hook
        self.n_updates = n_updates

        print("Preparing MNIST train images...", end=" ")
        self.train_dataset = MnistDataset('train', classes, train_samples)
        print("...Done.")

        print("Preparing MNIST validation images...", end=" ")
        self.val_dataset = MnistDataset('val', classes, val_samples)
        print("...Done.")

        print("Preparing MNIST test images...", end=" ")
        self.test_dataset = MnistDataset('test', classes, test_samples)
        print("...Done.")

        print("Shuffling MNIST train, validation & test images...", end=" ")
        random.shuffle(self.train_dataset)
        random.shuffle(self.val_dataset)
        random.shuffle(self.test_dataset)
        print("...Done.")

    def train(self, network, run=None, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3, with_optimizer=False):
        if with_optimizer:
            net, train_accuracy = self.train_with_optimizer(network, run, batch_size, epoch_qty, skip_initial_test, verbose)
        else:
            net, train_accuracy = self.train_vanilla(network, run, batch_size, epoch_qty, skip_initial_test, verbose)

        return net, train_accuracy

    def train_with_optimizer(self, network, run=None, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3):

        val_x, val_y = [x for x, _ in self.val_dataset], [y for _, y in self.val_dataset]
        accuracies = []

        # learn time constants before actual training
        if network.adapt_tau:
            network.adapt_timeconstants(val_x, batch_size=batch_size, verbose=verbose if verbose >= 3 else 0)

        # start training the network with the training set and periodically test prediction with the test set
        print("Start training network...")

        optimizer = torch.optim.Adam(network.parameters())

        for epoch in tqdm(range(epoch_qty), desc="Epochs"):

            if epoch != 0:
                print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
                random.shuffle(self.train_dataset)
                train_x, train_y = [x for x, _ in self.train_dataset], [y for _, y in self.train_dataset]

                n_samples = len(train_x)  # dataset size

                network.train()  # turn nudging on to enable training

                print("Learning with batch size {0}".format(batch_size))

                dataset = SimpleDataset(train_x, train_y)
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

                batch_qty = int(np.floor(n_samples/batch_size))
                for batch_i, (x, y) in enumerate(data_loader):
                    x = x.to(network.device)
                    y = y.to(network.device)

                    if verbose >= 1:
                        print("train:: batch ", batch_i + 1 if batch_i != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

                    for update_i in range(self.n_updates):
                        if verbose >= 2 and update_i % 10 == 0:
                            print(update_i, end=" ")

                        samples, labels = x, y
                        network.update(samples, labels)
                        optimizer.step()

                    if verbose >= 1:
                        print('')

                skip_initial_test = False

            if not skip_initial_test:
                print('Validate epoch {0}... | '.format(epoch), end='')
                y_hat = network.predict(val_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                accuracies.append(get_prediction_accuracy(val_y, y_hat))
                print(accuracies)

                print("Confusion matrix after epoch ", epoch, ": ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(val_y[:len(y_hat)], axis=1)))

                # log validation accuracy using sacred
                network.logs['val_accuracy'] = 100 * accuracies[-1]
                network.logs['val_error'] = 100 * (1 - accuracies[-1])
                # average validation loss per sample
                network.logs['val_loss'] = np.sum(np.subtract(y_hat, val_y[:len(y_hat)])**2)/len(y_hat)

            if epoch == epoch_qty - 1:
                # calculate test error if last epoch
                test_x, test_y = [x for x, _ in self.test_dataset], [y for _, y in self.test_dataset]
                print(f'Test after training {epoch} epochs... | ', end='')
                y_hat = network.predict(test_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                test_accuracy = get_prediction_accuracy(test_y, y_hat)
                network.logs['test_accuracy'] =  100 * test_accuracy
                network.logs['test_error'] = 100 * (1 - test_accuracy)
                network.logs['test_loss'] = np.sum(np.subtract(y_hat, test_y[:len(y_hat)])**2)/len(y_hat)

                print("Test accuracy after training: ", test_accuracy)
                print("Confusion matrix after training: ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(test_y[:len(y_hat)], axis=1)))

            if self.epoch_hook is not None:
                self.epoch_hook(network, run)

        return network, test_accuracy

    def train_vanilla(self, network, run=None, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3):

        val_x, val_y = [x for x, _ in self.val_dataset], [y for _, y in self.val_dataset]
        accuracies = []

        # learn time constants before actual training
        if network.adapt_tau:
            network.adapt_timeconstants(val_x, batch_size=batch_size, verbose=verbose if verbose >= 3 else 0)

        # start training the network with the training set and periodically test prediction with the test set
        print("Start training network...")

        for epoch in tqdm(range(epoch_qty), desc="Epochs"):

            if epoch != 0:
                print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
                random.shuffle(self.train_dataset)
                train_x, train_y = [x for x, _ in self.train_dataset], [y for _, y in self.train_dataset]
                network.fit(train_x, train_y, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                skip_initial_test = False

            if not skip_initial_test:
                print('Validate epoch {0}... | '.format(epoch), end='')
                y_hat = network.predict(val_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                accuracies.append(get_prediction_accuracy(val_y, y_hat))
                print(accuracies)

                print("Confusion matrix after epoch ", epoch, ": ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(val_y[:len(y_hat)], axis=1)))

                # log validation accuracy and error using sacred
                network.logs['val_accuracy'] = 100 * accuracies[-1]
                network.logs['val_error'] = 100 * (1 - accuracies[-1])
                # average validation loss per sample
                network.logs['val_loss'] = np.sum(np.subtract(y_hat, val_y[:len(y_hat)])**2)/len(y_hat)

            if epoch == epoch_qty - 1:
                # calculate test error if last epoch
                test_x, test_y = [x for x, _ in self.test_dataset], [y for _, y in self.test_dataset]
                print(f'Test after training {epoch} epochs... | ', end='')
                y_hat = network.predict(test_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                test_accuracy = get_prediction_accuracy(test_y, y_hat)
                network.logs['test_accuracy'] =  100 * test_accuracy
                network.logs['test_error'] = 100 * (1 - test_accuracy)
                network.logs['test_loss'] = np.sum(np.subtract(y_hat, test_y[:len(y_hat)])**2)/len(y_hat)

                print("Test accuracy after training: ", test_accuracy)
                print("Confusion matrix after training: ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(test_y[:len(y_hat)], axis=1)))

            if self.epoch_hook is not None:
                self.epoch_hook(network, run)

        return network, test_accuracy


if __name__ == "__main__":
    # get args from outside
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train lagrange mnist on all 10 classes and check accuracy.')
    parser.add_argument('--model_variant', default="vanilla", type=str, help="Model variant: vanilla, full_forward_pass")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training")
    parser.add_argument('--batch_learning_multiplier', default=128, type=int, help="Learning rate multiplier for batch learning")
    parser.add_argument('--with_optimizer', default=False, type=bool, help="Train network with Adam Optimizer")
    parser.add_argument('--classes', default=10, type=int, help='Number of classes to distinguish.')
    parser.add_argument('--train_samples', default=-1, type=int, help='Number of training samples per class.')
    parser.add_argument('--val_samples', default=-1, type=int, help='Number of validation samples per class.')
    parser.add_argument('--test_samples', default=-1, type=int, help='Number of test samples per class.')

    args, unknown = parser.parse_known_args()

    from model.network_params import LayeredParams

    params = LayeredParams()
    params.load_params('sacred_params.json')

    batch_size = args.batch_size
    lr_multiplier = args.batch_learning_multiplier

    # setup network parameters
    params.learning_rate = 0.1 * lr_multiplier

    presentation_steps = 10

    fc1 = nn.Linear(28 * 28, 300,  # this are the only things that should remain to be set here
                    tu.hard_sigmoid, tu.hard_sigmoid_deriv, params)
    fc2 = nn.Linear(300, 100, tu.hard_sigmoid, tu.hard_sigmoid_deriv, params)
    fc3 = nn.Linear(100, args.classes, tu.linear, tu.linear_deriv, params)

    le_net = nn.LESequential([fc1, fc2, fc3], params)

    trainer = MnistTrainer(classes=args.classes, train_samples=args.train_samples, val_samples=args.val_samples, test_samples=args.test_samples, n_updates=presentation_steps)
    trainer.train(le_net, batch_size=batch_size, epoch_qty=100, with_optimizer=args.with_optimizer, verbose=1)
