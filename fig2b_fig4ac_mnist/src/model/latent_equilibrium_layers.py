#!/usr/bin/env python
# PyTorch implementation of the Latent Equilibrium model for different layers.

# Authors: Benjamin Ellenberger (benelot@github) & Paul Haider (paulhaider@gihub)

from datetime import datetime

import numpy as np
import torch

from model.network_params import ModelVariant, TargetType, LayeredParams

from torch.utils.data import DataLoader
from utils.torch_utils import SimpleDataset


class Linear(object):
    def __init__(self, input_size, target_size, act_function, act_func_deriv, params: LayeredParams):
        self.input_size = input_size
        self.target_size = target_size
        self.act_function = act_function
        self.act_func_deriv = act_func_deriv

        self.tau = params.tau
        self.tau_m = params.tau_m
        self.tau_s = params.tau_s
        self.dt = params.dt
        self.learning_rate = params.learning_rate * params.lr_multiplier / params.dt / params.n_updates
        self.learning_rate_W = self.learning_rate
        self.learning_rate_b = self.learning_rate
        self.learning_rate_tau = params.learning_rate * params.lr_multiplier

        self.dtype = params.dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nudging = True
        self.train_W = True
        self.train_τ = False

        # initialize weights, voltages, rates, errors
        self.weights = torch.empty([self.input_size, self.target_size]).normal_(mean=0.0, std=0.05).to(self.device)
        self.biases = torch.empty(self.target_size).normal_(mean=0.0, std=0.05).to(self.device)

        self.feedback_alignment = params.feedback_alignment
        self.feedback_weights = torch.empty([self.input_size, self.target_size]).normal_(mean=0.0, std=0.05).to(self.device)

        self.voltages = torch.zeros([1, self.target_size], device=self.device)
        self.voltages_deriv = None
        self.prosp_voltages = torch.zeros([1, self.target_size], device=self.device)
        self.basal_inputs = None
        self.errors = torch.zeros([1, self.target_size], device=self.device)

        self.rho = None
        self.rho_input = None
        self.rho_deriv = torch.zeros([1, self.target_size], dtype=self.dtype, device=self.device)

        self.filtered_rates = torch.zeros([1, self.target_size], device=self.device)
        self.filtered_errors = torch.zeros([1, self.input_size], device=self.device)

        # turn off synaptic filters if τ_s = 0
        if self.tau_s == 0:
            self.forward_filter = False
            self.backward_filter = False
        else:
            self.forward_filter = params.forward_filter
            self.backward_filter = params.backward_filter

        self.noise_width = params.noise_width
        self.prosp_rate_errors = params.prosp_rate_errors

        self.tau_width = params.tau_width

        # simulate noise on both prospective and membrane time constants
        if self.tau_width != 0:
            self.tau *= torch.empty(self.target_size).normal_(mean=1.0, std=self.tau_width).to(self.device)
            self.tau_m *= torch.empty(self.target_size).normal_(mean=1.0, std=self.tau_width).to(self.device)
        else:
            self.tau *= torch.ones(self.target_size).to(self.device)
            self.tau_m *= torch.ones(self.target_size).to(self.device)

        # clip time constants to prevent them from being < 0
        # self.tau = torch.clip(self.tau, 1, 1000)
        # self.tau_m = torch.clip(self.tau_m, 1, 1000)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.rho_input.shape[0]
        if len(self.voltages.shape) != 2 or self.voltages.shape[0] != batch_size:
            voltage_size = self.voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            self.voltages = torch.repeat_interleave(self.voltages, repetition_vector, dim=0).clone()
            self.errors = torch.repeat_interleave(self.errors, repetition_vector, dim=0).clone()
            self.prosp_voltages = torch.repeat_interleave(self.prosp_voltages, repetition_vector, dim=0).clone()

            self.filtered_rates = torch.repeat_interleave(self.filtered_rates, repetition_vector, dim=0).clone()
            self.filtered_errors = torch.repeat_interleave(self.filtered_errors, repetition_vector, dim=0).clone()

    # TODO: I still don't like the name too much as this does not correspond to a forward pass
    # and would rather use something like 'layer_step' or 'layer_update'
    def forward(self, rho, rho_deriv):
        self.rho_input = rho.clone()

        self._adapt_parallel_network_qty()

        self.basal_inputs = torch.matmul(rho, self.weights) + self.biases

        self.voltages_deriv = 1.0 / self.tau_m * (self.basal_inputs - self.voltages + self.errors)
        # WARNING: it's crucial to update the prospective voltages before the
        # somatic voltages for the time constants to converge during adaption
        self.prosp_voltages = self.voltages + self.tau * self.voltages_deriv
        prosp_voltages_m = self.voltages + self.tau_m * self.voltages_deriv
        self.voltages = self.voltages + self.dt * self.voltages_deriv

        rho = self.act_function(self.prosp_voltages)
        self.rho_deriv = self.act_func_deriv(self.prosp_voltages)

        # simulate noise on firing rates
        if self.noise_width != 0:
            rho = torch.normal(mean=rho, std=self.noise_width)

        # perform synaptic filtering
        if self.forward_filter:
            self.filtered_rates += (rho - self.filtered_rates) * self.dt / self.tau_s
            self.rho = self.filtered_rates.clone()
        else:
            self.rho = rho.clone()

        # errors need to be calculated using the prospective voltage with τ_m
        if self.prosp_rate_errors:
            errors = self._calculate_errors(self.prosp_voltages, rho_deriv, self.basal_inputs)
        else:
            errors = self._calculate_errors(prosp_voltages_m, rho_deriv, self.basal_inputs)


        # apply synaptic filter to error signals
        if self.backward_filter:
            self.filtered_errors += (errors - self.filtered_errors) * self.dt / self.tau_s
            self.errors = self.filtered_errors.clone()
        else:
            self.errors = errors.clone()

        return self.rho, self.rho_deriv

    def update_weights(self, errors, with_optimizer=False):

        previous_layer_errors = self.errors

        self.errors = errors
        if self.train_W:

            dot_weights = self.get_weight_derivatives()
            self.weights.grad = -self.dt * dot_weights  # our gradients are inverse to pytorch gradients

            dot_biases = self.get_bias_derivatives()  # do bias update
            self.biases.grad = -self.dt * dot_biases # our gradients are inverse to pytorch gradients

            if not with_optimizer:  # minus because pytorch gradients are inverse to our gradients
                self.weights -= self.weights.grad * self.learning_rate_W
                self.biases -= self.biases.grad * self.learning_rate_b

        if self.train_τ:
            dot_tau = ((self.prosp_voltages - self.basal_inputs) * self.voltages_deriv).mean(0)
            # adapt both time constants at the same time
            # self.tau -= self.dt * self.learning_rate_tau * dot_tau
            self.tau_m += self.learning_rate_tau * dot_tau

        return previous_layer_errors

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η * weight_mask

        """
        # If the input is served as a single sample, it is not in batch form, but this here requires it.
        if len(self.rho_input.shape) == 1:
            self.rho_input = self.rho_input.unsqueeze(0)
        return (torch.einsum('bi,bj->bij', self.rho_input, self.errors)).mean(0)

    # ### CALCULATE BIAS DERIVATIVES ### #

    def get_bias_derivatives(self):
        """
        Return bias derivative.
        Args:

        Returns:
            bias_derivative: e * η * bias_mask

        """
        return self.errors.mean(0)

    def _calculate_bias_derivatives(self, prosp_voltages, weights, biases):
        """
        Calculate bias derivative.
        Same as weights_update but without pre-synaptic activities (Eq. 18)
        Args:
            prosp_voltages: U
            weights: W
            biases: b

        Returns:
            bias_derivative: (U - (W * r + b)) * η * weight_mask

        """
        rho = self.act_function(prosp_voltages)  # ρ(U)
        basal_inputs = torch.matmul(rho, weights) + biases  # self._calculate_basal_inputs(rho, weights, biases)  # W * r
        return (prosp_voltages - basal_inputs) * self.learning_rate_b

    # SOLVE ODE WITHOUT SOLVER
    ###########################

    def _get_lookahead_voltages(self, voltages, dot_voltages):
        """
        Calculate voltages lookaheads.
        Get u = \\bar u + \\tau \\dot{\\bar u}
        Args:
            voltages:
            dot_voltages:

        Returns:

        """
        return voltages + self.tau * dot_voltages

    def _calculate_errors(self, voltages_lookahead, rho_deriv, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') W^T (U - Wr)

        Args:
            prosp_voltages:
            rho_deriv:
            basal_inputs:

        Returns:
            errors:

        """
        # e
        if self.feedback_alignment:
            err = rho_deriv * torch.matmul(voltages_lookahead - basal_inputs, self.feedback_weights.t())
        else:
            err = rho_deriv * torch.matmul(voltages_lookahead - basal_inputs, self.weights.t())

        return err

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.weights.detach().cpu().numpy())
        np.save(logdir + "/layer_" + str(i) + "_biases.npy", self.biases.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        weights = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.weights = torch.from_numpy(weights).float().to(self.device)
        biases = np.load(logdir + "/layer_" + str(i) + "_biases.npy")
        self.biases = torch.from_numpy(biases).float().to(self.device)

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)


class LESequential(object):
    def __init__(self, layers, params: LayeredParams, loss_fn=None, loss_fn_deriv=None):
        self.layers = layers
        self.learning_rate = params.learning_rate * params.lr_multiplier / params.dt / params.n_updates
        self.learning_rate_W = self.learning_rate
        self.learning_rate_b = self.learning_rate
        self.learning_rate_tau = params.learning_rate * params.lr_multiplier
        for i, l in enumerate(layers):
            l.learning_rate_W = self.learning_rate * params.learning_rate_factors[i]
            l.learning_rate_b = self.learning_rate * params.learning_rate_factors[i]
            l.learning_rate_tau = self.learning_rate_tau * params.learning_rate_factors_tau[i]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.layer_qty = len(self.layers)
        self.train_W = True

        self.tau = params.tau
        self.tau_m = params.tau
        self.tau_s = params.tau_s
        self.dt = params.dt
        self.beta = params.beta

        self.adapt_tau = params.adapt_tau

        self._set_random_seed(7)  # Set random seed

        self.rho = [torch.zeros([1, layers[0].input_size], device=self.device)]
        self.rho_deriv = [torch.zeros([1, layers[0].input_size], device=self.device)]
        self.errors = [torch.zeros([1, layers[0].input_size], device=self.device)]
        for l in self.layers:
            self.rho.append(torch.zeros([1, l.target_size], device=self.device))
            self.rho_deriv.append(torch.zeros([1, l.target_size], device=self.device))
            self.errors.append(torch.zeros([1, l.target_size], device=self.device))

        self.dummy_label = torch.zeros([1, self.layers[-1].target_size], device=self.device)
        self.target = torch.zeros([1, self.layers[-1].target_size], device=self.device)

        self.model_variant = params.model_variant
        self.target_type = params.target_type
        self.with_optimizer = params.with_optimizer

        # turn off synaptic filters if τ_s = 0
        if self.tau_s == 0:
            self.target_lpf = False
        else:
            self.target_lpf = params.target_lpf

        self.logs = {}

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

    def train(self):
        """
        Enables training and nudging.
        :return:
        """
        self.train_W = True
        self.nudging = True

        for l in self.layers:
            l.train()

    def eval(self):
        """
        Disables training and nudging.
        :return:
        """
        self.train_W = False
        self.nudging = False

        for l in self.layers:
            l.eval()

    def update(self, inputs, targets):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        :math:`(6.0) \\tau \\dot u = - u + W r +  e`

        :math:`(6.1)           e = r' \\odot  W^T[u - W r] + \\beta e^{trg}`

        :math:`(6.2)           e^{trg} = r' \\odot (r^{trg} - r)`

        Args:
            inputs: current input for layer
            targets: current target for layer

        Returns:

        """

        if targets is None and self.train_W:
            raise Exception("For training, a valid target must be provided")
        elif not self.train_W:
            targets = self.dummy_label
        elif self.target_lpf:
            self.target += (targets - self.target) * self.dt / (len(self.layers) * self.tau_s)
        else:
            self.target = targets.clone()

        self.rho[0] = inputs.clone()
        self.rho_deriv[0] = torch.zeros_like(inputs)  # the first layer needs no rho deriv, because the calculated error can not be applied to any weight
        self._adapt_parallel_network_qty(inputs)

        self.errors[-1] = self.nudging * self.beta * self._calculate_e_trg(self.target, self.layers[-1].prosp_voltages, self.rho[-1], self.layers[-1].rho_deriv)

        if self.model_variant == ModelVariant.VANILLA:
            new_rhos = [None for _ in range(self.layer_qty + 1)]
            new_rho_derivs = [None for _ in range(self.layer_qty + 1)]

            for i, layer in enumerate(self.layers):
                new_rhos[i + 1], new_rho_derivs[i + 1] = layer(self.rho[i], self.rho_deriv[i])  # in principle W * r + b

            self.rho = new_rhos
            self.rho_deriv = new_rho_derivs

        elif self.model_variant == ModelVariant.FULL_FORWARD_PASS:
            for i, layer in enumerate(self.layers):
                self.rho[i + 1], self.rho_deriv[i + 1] = layer(self.rho[i], self.rho_deriv[i])  # in principle W * r + b

        for i, layer in reversed(list(enumerate(self.layers))):
            self.errors[i] = layer.update_weights(self.errors[i + 1], self.with_optimizer)

    def get_errors(self):
        return self.errors

    def _calculate_e_trg(self, target, prosp_voltages, rho, rho_deriv):
        """
        Calculate e_trg (insert into errors as err[-self.target_size:] = get_e_trg(...).)
        Args:
            target:
            rho:
            rho_deriv:

        Returns:
            e_trg

        """

        # calculate target error
        # and use rate difference as target
        if self.target_type == TargetType.RATE:
            e_trg = rho_deriv * (target - rho)
        # or use voltage difference as target
        else:
            e_trg = (target - prosp_voltages)
        return e_trg

    def infer(self, x):
        self.eval()
        self.update(x, self.dummy_label)

    def __call__(self, x):
        return self.infer(x)

    # KERAS-like INTERFACE
    def fit(self, x=None, y=None, n_updates: int = 100, batch_size=1, epochs=1, verbose=1):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per batch.
            batch_size: Number of examples per batch (batch training).
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
        """
        n_samples = len(x)  # dataset size

        if not self.train_W:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(self.beta))
            self.train()  # turn nudging on to enable training

        print("Learning with batch size {0}".format(batch_size))

        dataset = SimpleDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

        batch_qty = int(np.floor(n_samples/batch_size))
        for epoch_i in range(epochs):
            for batch_i, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                self.fit_batch(x, y, n_updates, batch_i, batch_qty, verbose)

    def fit_batch(self, x=None, y=None, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1):
        if verbose >= 1:
            print("train:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            samples, labels = x, y
            self.update(samples, labels)

        if verbose >= 1:
            print('')

    def predict(self, x=None, n_updates: int = 100, batch_size=1, verbose=1):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Size of batch to be predicted.
            verbose: Level of verbosity.
        :return:
        """
        n_samples = len(x)  # dataset size
        self.eval()   # turn nudging off to disable learning
        print("Learning turned off")

        dataset = SimpleDataset(x, [np.zeros(self.layers[-1].target_size) for _ in range(n_samples)])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

        batch_qty = int(np.floor(n_samples/batch_size))
        predictions = []
        for batch_i, (x, _) in enumerate(data_loader):
            x = x.to(self.device)

            batch_predictions = self.predict_batch(x, batch_iteration=batch_i, batch_qty=batch_qty, n_updates=n_updates, verbose=verbose)
            predictions.extend(batch_predictions)

        return predictions

    def predict_batch(self, x, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1):
        if verbose >= 1:
            print("predict:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            samples = x
            self.update(samples, self.dummy_label)

        # use either rates or voltages as network output
        if self.target_type == TargetType.RATE:
            rates = self.rho[-1]
            batch_predictions = rates.detach().cpu().numpy()
        else:
            volts = self.layers[-1].prosp_voltages
            batch_predictions = volts.detach().cpu().numpy()

        if verbose >= 1:
            print('')

        return batch_predictions

    def adapt_timeconstants(self, x=None, n_updates: int = 10, batch_size=1, verbose=1):
        """
        Adapt time constants before actual training.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Size of batch to be predicted.
            verbose: Level of verbosity.
        :return:
        """
        assert (self.learning_rate_tau), "Set learning rate > 0 to adapt time constants!"

        n_samples = len(x)  # dataset size
        self.eval()   # turn nudging off to disable learning
        print("Learning time constants...")

        dataset = SimpleDataset(x, [np.zeros(self.layers[-1].target_size) for _ in range(n_samples)])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

        batch_qty = int(np.floor(n_samples/batch_size))

        # activate learning in all layers
        for l in self.layers:
            l.train_τ = True

        for i, l in enumerate(self.layers):
            converged = False
            prev_Δτ = 0.
            while not converged:
                Δτ = torch.mean(torch.abs(l.tau_m - l.tau))
                if torch.abs(Δτ - prev_Δτ) < 1E-6:
                    converged = True
                else:
                    prev_Δτ = Δτ
                for batch_i, (x, _) in enumerate(data_loader):
                    x = x.to(self.device)
                    for _ in range(n_updates):
                        self.update(x, self.dummy_label)
                if verbose >= 2:
                    print(f"average Δτ in layer {i} = ", torch.mean(torch.abs(l.tau_m - l.tau)))
            l.train_τ = False
            print(f"average Δτ in layer {i} after adaptation: ", torch.mean(torch.abs(l.tau_m - l.tau)))

    def _adapt_parallel_network_qty(self, x):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = x.shape[0]
        if len(self.rho[1].shape) != 2 or self.rho[1].shape[0] != batch_size:
            rho_size = self.rho[1].shape[0]
            repeats = int(batch_size / rho_size)
            remainder = batch_size % rho_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(rho_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            for i in range(1, len(self.rho)):
                self.rho[i] = torch.repeat_interleave(self.rho[i], repetition_vector, dim=0).clone()
                self.rho_deriv[i] = torch.repeat_interleave(self.rho_deriv[i], repetition_vector, dim=0).clone()
                self.errors[i] = torch.repeat_interleave(self.errors[i], repetition_vector, dim=0).clone()

            self.dummy_label = torch.repeat_interleave(self.dummy_label, repetition_vector, dim=0).clone()
            self.target = torch.repeat_interleave(self.target, repetition_vector, dim=0).clone()

    def save_model(self, logdir):
        for i, l in enumerate(self.layers):
            l.save_layer(logdir, i)
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        print(f"saved at time: {str(current_time)}")

    def load_model(self, old_savedir):
        for (i, l) in enumerate(self.layers):
            l.load_layer(old_savedir, i)

    def parameters(self):
        return [l.weights for l in self.layers] + [l.biases for l in self.layers]
