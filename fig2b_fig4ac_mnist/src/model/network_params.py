from enum import Enum

import re
import json
import numpy
import torch


class Model(str, Enum):
    """
    Type of model used to train the network.
    """

    LATENT_EQUILIBRIUM = 'latent_equilibrium'

class ModelVariant(str, Enum):
    """
    Which variant of the model to use. Default is predictive coding scheme but the lookahead
    voltages can also be updated directly without the use of differential equations.
    """

    # LE variants
    VANILLA = 'vanilla'

class ActivationFunction(str, Enum):
    """
    Activation function for the neurons.
    """

    SIGMOID = 'sigmoid'                                                         # sigmoidal activation function
    MOD_SIGMOID = 'mod_sigmoid'                                                 # modified sigmoid with slope of ~ 1 and shifted to the right by 0.5
    HARD_SIGMOID = 'hard_sigmoid'                                               # hard sigmoid activation function
    RELU = 'relu'                                                               # ReLU activation function
    LINEAR = 'linear'                                                           # Linear activation function
    TANH = 'tanh'                                                               # Tangens hyperbolicus activation function
    ELU = 'elu'                                                                 # ELU activation function
    SWISH = 'swish'                                                             # Swish activation function

class TargetType(str, Enum):
    """
    Use either rate-based or voltage-based target errors in the case of LE.
    """

    RATE = 'rate'                                                               # use output rates to define the target error
    VOLTAGE = 'voltage'                                                         # use lookahead voltages to define the target error

class LayeredParams():
    """
    Parameters for layered implementation of LE.
    """

    # params for linear layers
    tau = 10.0                                                      # prospective time constant
    dt = 0.1                                                        # integration step
    dtype = torch.float32                                           # data type used in calculations

    learning_rate_factors = [1, 0.2, 0.1]                           # learning rate factor to scale learning rates for each layer
    inference_learning_rate = 0.1
    learning_rate = 0.1
    lr_multiplier = 1.0                                             # learning rate multiplier when using batches
    n_updates = 100                                                 # number of updates per sample
    beta = 0.1                                                      # nudging parameter beta

    with_optimizer = True

    # noise and time constants
    tau_m = 10.0                                                    # membrane time constant
    tau_s = 0.0                                                     # synaptic time constant

    prosp_rate_errors = False                                       # use prospective voltage with τ_r to calculate errors instead of propsective voltage with τ_m
    noise_width = 0.                                                # if not zero, white noise is added to the firing rates at each timestep
    tau_width = 0.

    adapt_tau = False                                               # learn time constants before training
    learning_rate_factors_tau = [10, 10, 10]                        # learning rate factor to scale learning rates for each layer

    forward_noise = False                                           # add noise to firing rates
    backward_noise = False                                          # add noise to error signals

    forward_filter = False                                          # synaptic filtering of basal inputs
    backward_filter = False                                         # synaptic filtering of apical errors signals

    target_lpf = False

    target_type = TargetType.RATE                                   # use voltages or rates for target error
    model_variant = ModelVariant.VANILLA                            # variant of the model

    feedback_alignment = False                                      # use fixed backward weights

    rnd_seed = 42                                                   # random seed


    def __init__(self, file_name=None):
        if file_name is not None:
            self.load_params(file_name)

    def load_params(self, file_name):
        """
        Load parameters from json file.
        """
        with open(file_name, 'r') as file:
            deserialize_dict = json.load(file)
            for key, value in deserialize_dict.items():
                if isinstance(value, str) and 'numpy' in value:
                    value = eval(value)
                elif isinstance(value, str) and getattr(self, key).__class__ is not str:
                    key_class = getattr(self, key).__class__
                    value = key_class(value)

                setattr(self, key, value)

    def load_params_from_dict(self, dictionary):
        """
        Load parameters from dictionary.
        """
        for key, value in dictionary.items():
            # check if key is actually an attribute of the NetworkParams class
            if not hasattr(self, key):
                continue
            if isinstance(value, str) and 'numpy' in value:
                value = eval(value)
            elif isinstance(value, str) and getattr(self, key).__class__ is not str:
                key_class = getattr(self, key).__class__
                value = key_class(value)

            setattr(self, key, value)

    def save_params(self, file_name):
        """
        Save parameters to json file.
        """
        with open(file_name, 'w') as file:
            file.write(self.to_json())

    def to_json(self):
        """
        Turn network params into json.
        """
        serialize_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if callable(value):
                    if value is numpy.float32 or value is numpy.float64:
                        value = re.search('\'(.+?)\'', str(value)).group(1)
                    else:
                        break
                serialize_dict[key] = value

        return json.dumps(serialize_dict, indent=4)

    def __str__(self):
        """
        Return string representation.
        Returns:

        """
        return self.to_json()
