

import random
import numpy as np
import datetime
import argparse

import sys
sys.path.append(".")
from model.network_params import ModelVariant, TargetType

# plotting
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

# mnist dataset and metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from experiments.MNIST.common import get_digits_from_class, load_params, save_params
from torch.utils.data import DataLoader
from utils.torch_utils import SimpleDataset
from tqdm import tqdm

import model.latent_equilibrium_layers as nn
import model.layered_torch_utils as tu


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


def save_accuracies(accuracies, prefix=datetime.datetime.now().strftime("%Y%m%d%H%M%S")):
    """
    Plot accuracy and save values to text.
    """
    plt.close()
    plt.plot(accuracies, linewidth=2.)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    try:
        plt.savefig("./output/" + prefix + '_accuracies.pdf')
        np.savetxt("./output/" + prefix + '_accuracies.txt', accuracies)
    except (PermissionError, FileNotFoundError) as e:
        print("Could not save accuracies due to missing permissions (They should be printed above).\n Please fix permissions, I will try again after next epoch...")


class MnistTrainer:

    def __init__(self, classes: int=10, train_samples: int=2000, test_samples: int=100, n_updates: int=10, epoch_hook=None):
        """ Initialize the mnist trainer.

        Args:
            params: Network parameters for a new network.
            classes: Number of MNIST classes for labels.
            train_samples: Number of training samples
            test_samples: Number of test samples
            epoch_hook: Function hook executed after every epoch
        """
        self.epoch_hook = epoch_hook

        self.n_updates = n_updates

        # load mnist's official training and test data
        # Load data from https://www.openml.org/d/554
        training_set_size = train_samples  # default 2000
        test_set_size = test_samples  # default 100
        digit_classes = classes  # default 10

        print("Loading MNIST...", end=" ")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='.', as_frame=False)
        print("...Done.")

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=seed)

        val_size = 10000
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)

        # get mnist training images from official training set of size training_set_size
        print("Preparing MNIST train images...", end=" ")
        train_dataset = []
        for digit_class in range(digit_classes):
            print(digit_class, end=" ")
            train_dataset += get_digits_from_class(x_train, y_train, classes, digit_class, training_set_size, False, reshape=(28, 28))

        self.train_dataset = train_dataset
        print("...Done.")

        # get mnist training images from official training set of size training_set_size
        print("Preparing MNIST validation images...", end=" ")
        val_dataset = []
        for digit_class in range(digit_classes):
            print(digit_class, end=" ")
            val_dataset += get_digits_from_class(x_val, y_val, classes, digit_class, training_set_size, False, reshape=(28, 28))

        self.val_dataset = val_dataset
        print("...Done.")

        # get mnist test images from official training set of size test_set_size
        print("Preparing MNIST test images...", end=" ")
        test_dataset = []
        for digit_class in range(digit_classes):
            print(digit_class, end=" ")
            test_dataset += get_digits_from_class(x_test, y_test, classes, digit_class, test_set_size, False, reshape=(28, 28))
        self.test_dataset = test_dataset
        print("...Done.")

        # shuffle both training sets
        print("Shuffling MNIST train/test images...", end=" ")
        random.shuffle(self.train_dataset)
        random.shuffle(self.val_dataset)
        random.shuffle(self.test_dataset)
        print("...Done.")

    def train(self, network, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3, with_optimizer=False):
        if with_optimizer:
            self.train_with_optimizer(network, batch_size, epoch_qty, skip_initial_test, verbose)
        else:
            self.train_vanilla(network, batch_size, epoch_qty, skip_initial_test, verbose)

    def train_with_optimizer(self, network, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3):
        # create a prefix for the file containing a unique timestamp
        accuracy_filename_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # append timestamp

        val_x, val_y = [np.expand_dims(x, 0) for x, _ in self.val_dataset], [y for _, y in self.val_dataset]

        test_x, test_y = [np.expand_dims(x, 0) for x, _ in self.test_dataset], [y for _, y in self.test_dataset]
        accuracies = []

        # start training the network with the training set and periodically test prediction with the test set
        print("Start training network...")

        optimizer = torch.optim.Adam(network.parameters())

        is_epoch_known = False
        for epoch in tqdm(range(epoch_qty), desc="Epochs"):

            # find last already trained epoch
            # if not is_epoch_known:
            #     try:
            #         load_params(network, epoch, dry_run=True)  # test if network params could be loaded
            #         print('Epoch {0} already done.'.format(epoch))
            #         continue
            #
            #     except Exception as e:
            #         print('Epoch {0} not done.'.format(epoch))
            #         if epoch != 0:
            #             print('Loading weights of epoch {0} to continue...'.format(epoch - 1))
            #             load_params(network, epoch - 1, dry_run=False)
            #             print('...Done.')
            #
            #         is_epoch_known = True

            if epoch != 0:
                print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
                random.shuffle(self.train_dataset)
                train_x, train_y = [np.expand_dims(x, 0)  for x, _ in self.train_dataset], [y for _, y in self.train_dataset]

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
                save_accuracies(accuracies, accuracy_filename_prefix)
                # print('Save epoch {0}... | '.format(epoch), end='')
                # save_params(network, epoch)

                print("Confusion matrix after epoch ", epoch, ": ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(val_y[:len(y_hat)], axis=1)))

            if self.epoch_hook is not None:
                self.epoch_hook(network)

        print('Test network... | ')
        y_hat = network.predict(test_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
        print(get_prediction_accuracy(test_y, y_hat))
        # print('Save epoch {0}... | '.format(epoch), end='')
        # save_params(network, epoch)

        return network, accuracies

    def train_vanilla(self, network, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3):

        # create a prefix for the file containing a unique timestamp
        accuracy_filename_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # append timestamp

        val_x, val_y = [np.expand_dims(x, 0) for x, _ in self.val_dataset], [y for _, y in self.val_dataset]

        test_x, test_y = [np.expand_dims(x, 0) for x, _ in self.test_dataset], [y for _, y in self.test_dataset]
        accuracies = []

        # start training the network with the training set and periodically test prediction with the test set
        print("Start training network...")

        is_epoch_known = False
        for epoch in tqdm(range(epoch_qty), desc="Epochs"):

            # find last already trained epoch
            # if not is_epoch_known:
            #     try:
            #         load_params(network, epoch, dry_run=True)  # test if network params could be loaded
            #         print('Epoch {0} already done.'.format(epoch))
            #         continue
            #
            #     except Exception as e:
            #         print('Epoch {0} not done.'.format(epoch))
            #         if epoch != 0:
            #             print('Loading weights of epoch {0} to continue...'.format(epoch - 1))
            #             load_params(network, epoch - 1, dry_run=False)
            #             print('...Done.')
            #
            #         is_epoch_known = True

            if epoch != 0:
                print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
                random.shuffle(self.train_dataset)
                train_x, train_y = [np.expand_dims(x, 0) for x, _ in self.train_dataset], [y for _, y in self.train_dataset]
                network.fit(train_x, train_y, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                skip_initial_test = False

            if not skip_initial_test:
                print('Validate epoch {0}... | '.format(epoch), end='')
                y_hat = network.predict(val_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
                accuracies.append(get_prediction_accuracy(val_y, y_hat))
                print(accuracies)
                save_accuracies(accuracies, accuracy_filename_prefix)
                # print('Save epoch {0}... | '.format(epoch), end='')
                # save_params(network, epoch)

                print("Confusion matrix after epoch ", epoch, ": ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(val_y[:len(y_hat)], axis=1)))

            if self.epoch_hook is not None:
                self.epoch_hook(network)

        print('Test network... | ')
        y_hat = network.predict(test_x, batch_size=batch_size, n_updates=self.n_updates, verbose=verbose if verbose >= 3 else 0)
        print(get_prediction_accuracy(test_y, y_hat))
        # print('Save epoch {0}... | '.format(epoch), end='')
        # save_params(network, epoch)

        return network, accuracies


if __name__ == "__main__":
    # get args from outside
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train latent equilibrium mnist on all 10 classes and check accuracy.')
    parser.add_argument('--model_variant', default="vanilla", type=str, help="Model variant: vanilla, full_forward_pass")
    parser.add_argument('--batch_size', default=512, type=int, help="Batch size for training")
    parser.add_argument('--batch_learning_multiplier', default=128, type=int, help="Learning rate multiplier for batch learning")
    parser.add_argument('--n_updates', default=10, type=int, help="Number of update steps per sample/batch")
    parser.add_argument('--with_optimizer', action='store_true', help="Train network with Adam Optimizer")
    parser.add_argument('--classes', default=10, type=int, help='Number of classes to distinguish.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--train_samples', default=-1, type=int, help='Number of training samples per class.')
    parser.add_argument('--test_samples', default=-1, type=int, help='Number of test samples per class.')
    parser.add_argument('--seed', default=7, type=int, help='Seed for reproducibility.')

    args, unknown = parser.parse_known_args()

    # reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = args.batch_size
    lr_multiplier = args.batch_learning_multiplier

    # setup network parameters
    tau = 10.0
    dt = 0.1
    beta = 0.1
    model_variant = args.model_variant
    target_type = TargetType.RATE
    presentation_steps = args.n_updates

    learning_rate = 0.125 * lr_multiplier / presentation_steps / dt

    l1 = nn.Conv2d(1, 20, 5, batch_size, 28, tu.HardSigmoid)
    l2 = nn.MaxPool2d(2)
    l3 = nn.Conv2d(20, 50, 5, batch_size, 12, tu.HardSigmoid)
    l4 = nn.MaxPool2d(2)
    l5 = nn.Projection((batch_size, 50, 4, 4), 500, tu.HardSigmoid)
    l6 = nn.Linear(500, args.classes, tu.Linear)

    network = nn.LESequential([l1, l2, l3, l4, l5, l6], learning_rate, [1.0, 0.2, 0.1, 0.1, 0.1, 0.1], None, None,
                       tau, dt, beta, model_variant, target_type, with_optimizer=args.with_optimizer)

    trainer = MnistTrainer(classes=args.classes, train_samples=args.train_samples, test_samples=args.test_samples, n_updates=presentation_steps)
    trainer.train(network, batch_size=batch_size, epoch_qty=args.epochs, with_optimizer=args.with_optimizer, skip_initial_test=True)
