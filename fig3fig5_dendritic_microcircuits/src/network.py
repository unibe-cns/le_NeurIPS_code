import numpy as np
import utils as u
import time

dtype = np.float32

class Layer:
    '''
    Class for hidden layers
    '''
    def __init__(self, N_pyr, N_in, N_next, eta, params, act, bias, bias_val, le):
        '''
        Initialize all potentials and weights and set parameters.
        Parameters
        ----------
        N_pyr:      Hidden dimension (number of pyramidals)
        N_in:       Dimension of previous layer
        N_next:     Dimension of next layer (number of pyramidals in next layer)
        eta:        Dictionary with learning rates for the plastic connections
        params:     Parameters (conductances, plasticity time constant tau_w, noise level, timestep dt)
        act:        Activation function (must be a function taking one argument)
        bias:       Whether to add an artifical bias neuron to upwards connection and pyramidal to inter connection
        bias_val:   Value of the bias neuron (rate)
        le:         Use latent equilibrium mode
        '''
        self.u_pyr = {"basal": np.zeros(N_pyr, dtype=dtype), "apical": np.zeros(N_pyr, dtype=dtype),
                      "soma": np.zeros(N_pyr, dtype=dtype), "forw": np.ones(N_pyr, dtype=dtype),
                      "steadystate": np.zeros(N_pyr, dtype=dtype), "udot": np.zeros(N_pyr, dtype=dtype)}
        print('in layer init', self.u_pyr)
        self.u_inn = {"dendrite": np.zeros(N_next, dtype=dtype), "soma": np.zeros(N_next, dtype=dtype),
                      "forw": np.zeros(N_next, dtype=dtype), "steadystate": np.zeros(N_next, dtype=dtype),
                      "udot": np.zeros(N_next, dtype=dtype)}

        self.W_up = (np.random.sample((N_pyr, N_in + bias)).astype(dtype) - 0.5) * 2 * params["init_weights"]["up"]
        self.w_max = 4.
        self.W_down = (np.random.sample((N_pyr, N_next)).astype(dtype) - 0.5) * 2 * params["init_weights"]["down"]
        self.W_pi = (np.random.sample((N_pyr, N_next)).astype(dtype) - 0.5) * 2 * params["init_weights"]["pi"]
        self.W_ip = (np.random.sample((N_next, N_pyr + bias)).astype(dtype) - 0.5) * 2 * params["init_weights"]["ip"]

        self.bias = bias
        self.bias_val = bias_val

        self.Delta_up = np.zeros((N_pyr, N_in + bias), dtype=dtype)
        self.Delta_pi = np.zeros((N_pyr, N_next), dtype=dtype)
        self.Delta_ip = np.zeros((N_next, N_pyr + bias), dtype=dtype)

        self.set_params(params, eta)

        self.le = le
        self.act = act

    def set_params(self, params, eta):
        self.gl = params["gl"]
        self.gb = params["gb"]
        self.ga = params["ga"]
        self.gsom = params["gsom"]
        self.gd = params["gd"]

        self.eta = eta.copy()
        self.tau_w = params["tau_w"]

        self.noise = params["noise"]

        self.dt = params["dt"]

        self.params = params

    def update(self, r_in, u_next, learning_on, noise_on=True):
        '''
        Compute potential updates and weights for one timestep. Potential updates have to be applied by calling apply(),
        after update was called for all the other layers.
        Parameters
        ----------
        r_in:           Input rates (from previous layer)
        u_next:         Soma potentials of pyramidals of next layer
        learning_on:    Switch learning on or off
        noise_on:       Switch noise on or off

        '''
        #### rates
        r_in_buf = np.zeros(r_in.shape[0] + self.bias, dtype=dtype)
        r_in_buf[: len(r_in)] = r_in
        r_next = self.act(u_next)
        if self.le:
            r_pyr = np.zeros(self.u_pyr["forw"].shape[0] + self.bias, dtype=dtype)
            r_pyr[: len(self.u_pyr["forw"])] = self.act(self.u_pyr["forw"])
            r_inn = self.act(self.u_inn["forw"])
        else:
            r_pyr = np.zeros(self.u_pyr["soma"].shape[0] + self.bias, dtype=dtype)
            r_pyr[: len(self.u_pyr["soma"])] = self.act(self.u_pyr["soma"])
            r_inn = self.act(self.u_inn["soma"])
        if self.bias:
            r_in_buf[-1] = self.bias_val
            r_pyr[-1] = self.bias_val

        ####compute dendritic potentials at current time

        # pyramidial neurons
        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in_buf)  # [:] to enforce rhs to be copied to lhs
        self.u_pyr["apical"][:] = np.matmul(self.W_down, r_next) + np.matmul(self.W_pi, r_inn)

        # lateral interneurons
        self.u_inn["dendrite"][:] = np.matmul(self.W_ip, r_pyr)

        #if self.le:
        # pyramidal in lower layer never gets a nudging signal -> assume g_soma = 0
        self.u_pyr["steadystate"][:] = (self.gb * self.u_pyr["basal"] + self.ga * self.u_pyr["apical"]) / (
            self.gl + self.gb + self.ga)
        # inter has no apical, if more than one hidden layer: make sure that g_som = g_apial(above pyr)
        self.u_inn["steadystate"][:] = (self.gb * self.u_inn["dendrite"] + self.gsom * u_next) / (
            self.gl + self.gb + self.gsom)

        ####compute changes

        u_p = self.u_pyr["soma"]
        u_i = self.u_inn["soma"]

        #if self.le:
        self.u_pyr["udot"][:] = (self.gl + self.gb + self.ga) * (self.u_pyr["steadystate"] - u_p)
        self.du_pyr = self.u_pyr["udot"] * self.dt
        self.u_inn["udot"][:] = (self.gl + self.gb + self.gsom) * (self.u_inn["steadystate"] - u_i)
        self.du_inn = self.u_inn["udot"] * self.dt
        #else:
            #self.du_pyr = self.dt * (-self.gl * u_p + self.gb * (self.u_pyr["basal"] - u_p) + self.ga * (
            #    self.u_pyr["apical"] - u_p) + noise_on * self.noise * np.random.normal(size=len(self.u_pyr["soma"])))
            #self.du_inn = self.dt * (-self.gl * u_i + self.gd * (self.u_inn["dendrite"] - u_i) + self.gsom * (
            #    u_next - u_i) + noise_on * self.noise * np.random.normal(size=len(self.u_inn["soma"])))

        # weight updates (lowpass weight changes)
        if learning_on:
            self.update_weights = True
            if self.le:
                tau_pyr = 1. / (self.gl + self.gb + self.ga)
                tau_inn = 1. / (self.gl + self.gb + self.gsom)
                u_new_pyr_forw = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
                u_new_inn_forw = self.u_inn["soma"] + tau_inn * self.u_inn["udot"]
                gtot = self.gl + self.gb + self.ga
                self.Delta_up = np.outer(
                    self.act(u_new_pyr_forw) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf)
                self.Delta_ip = np.outer(
                    self.act(u_new_inn_forw) - self.act(self.gb / (self.gl + self.gb) * self.u_inn["dendrite"]), r_pyr)
                self.Delta_pi = np.outer(-self.u_pyr["apical"], r_inn)
            else:
                u_new_pyr = self.u_pyr["soma"] + self.du_pyr
                u_new_inn = self.u_inn["soma"] + self.du_inn
                gtot = self.gl + self.gb + self.ga
                self.Delta_up = np.outer(
                    self.act(u_new_pyr) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf)
                self.Delta_ip = np.outer(
                    self.act(u_new_inn) - self.act(self.gb / (self.gl + self.gb) * self.u_inn["dendrite"]), r_pyr)
                self.Delta_pi = np.outer(-self.u_pyr["apical"], r_inn)
                #gtot = self.gl + self.gb + self.ga
                #u_new_pyr = self.u_pyr["soma"] + self.du_pyr
                #u_new_inn = self.u_inn["soma"] + self.du_inn
                #self.dDelta_up = self.dt / self.tau_w * (- self.Delta_up + np.outer(
                #    self.act(u_new_pyr) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf))
                #self.dDelta_ip = self.dt / self.tau_w * (- self.Delta_ip + np.outer(
                #    self.act(u_new_inn) - self.act(self.gd / (self.gl + self.gd) * self.u_inn["dendrite"]), r_pyr))
                #self.dDelta_pi = self.dt / self.tau_w * (- self.Delta_pi + np.outer(-self.u_pyr["apical"], r_inn))
        else:
            self.update_weights = False

    def apply(self):
        # apply changes to soma potential
        if self.le:
            tau_pyr = 1. / (self.gl + self.gb + self.ga)
            tau_inn = 1. / (self.gl + self.gb + self.gsom)
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
            self.u_inn["forw"][:] = self.u_inn["soma"] + tau_inn * self.u_inn["udot"]
        self.u_pyr["soma"] += self.du_pyr
        self.u_inn["soma"] += self.du_inn
        # apply weight updates
        if self.update_weights:
        #    if self.le:
        #        self.W_up += self.dt * self.eta["up"] * self.Delta_up
        #        self.W_up[self.W_up > self.w_max] = self.w_max
        #        self.W_up[self.W_up < -1 * self.w_max] = -1 * self.w_max
        #        self.W_ip += self.dt * self.eta["ip"] * self.Delta_ip
        #        self.W_pi += self.dt * self.eta["pi"] * self.Delta_pi
        #    else:
            self.W_up += self.dt * self.eta["up"] * self.Delta_up
            self.W_up[self.W_up > self.w_max] = self.w_max
            self.W_up[self.W_up < -1 * self.w_max] = -1 * self.w_max
            self.W_ip += self.dt * self.eta["ip"] * self.Delta_ip
            self.W_pi += self.dt * self.eta["pi"] * self.Delta_pi
                ## apply Deltas after w update (to have correct euler for w, i.e. with old delta)
                #self.Delta_up += self.dDelta_up
                #self.Delta_ip += self.dDelta_ip
                #self.Delta_pi += self.dDelta_pi


    def reset(self, reset_weights=True):
        '''
        Reset all potentials and Deltas (weight update matrices) to zero.
        Parameters
        ----------
        reset_weights:  Also draw weights again from random distribution.

        '''
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)
        self.u_pyr["apical"].fill(0)
        self.u_pyr["steadystate"].fill(0)
        self.u_pyr["forw"].fill(0)
        self.u_pyr["udot"].fill(0)
        self.u_inn["dendrite"].fill(0)
        self.u_inn["soma"].fill(0)
        self.u_inn["steadystate"].fill(0)
        self.u_inn["forw"].fill(0)
        self.u_inn["udot"].fill(0)

        if reset_weights:
            self.W_up = (np.rand_like(self.W_up) - 0.5) * 2 * self.params["init_weights"]["up"]
            self.W_down = (np.rand_like(self.W_down) - 0.5) * 2 * self.params["init_weights"]["down"]
            self.W_pi = (np.rand_like(self.W_pi) - 0.5) * 2 * self.params["init_weights"]["pi"]
            self.W_ip = (np.rand_like(self.W_ip) - 0.5) * 2 * self.params["init_weights"]["ip"]

        self.Delta_up.fill(0)
        self.Delta_pi.fill(0)
        self.Delta_ip.fill(0)


class OutputLayer:
    '''
    Class for output layer. See documention of Layer for more info about the different methods.
    '''
    def __init__(self, N_out, N_in, eta, params, act, bias, bias_val, latent_eq):
        self.u_pyr = {"basal": np.zeros(N_out, dtype=dtype), "soma": np.zeros(N_out, dtype=dtype),
                      "steadystate": np.zeros(N_out, dtype=dtype), "forw": np.zeros(N_out, dtype=dtype),
                      "udot": np.zeros(N_out, dtype=dtype)}

        self.W_up = (np.random.sample((N_out, N_in + bias)).astype(dtype) - 0.5) * 2 * params["init_weights"]["up"]
        self.w_max = 5.

        self.Delta_up = np.zeros((N_out, N_in + bias), dtype=dtype)

        self.le = latent_eq
        self.act = act
        self.set_params(params, eta)

        self.bias = bias
        self.bias_val = bias_val

    def set_params(self, params, eta):
        self.gl = params["gl"]
        self.gb = params["gb"]
        self.gsom = params["gsom"]
        self.ga = 0
        self.eta = eta.copy()
        self.tau_w = params["tau_w"]
        self.noise = params["noise"]
        self.dt = params["dt"]
        self.params = params

    def update(self, r_in, u_target, learning_on, noise_on=True):
        #### input rates
        r_in_buf = np.zeros(r_in.shape[0] + self.bias, dtype=dtype)
        if self.bias:
            r_in_buf[:-1] = r_in
            r_in_buf[-1] = self.bias_val
        else:
            r_in_buf = r_in

        #### compute dendritic potentials at current time

        self.u_pyr["basal"][:] = np.matmul(self.W_up, r_in_buf)
        #if self.le:
        # pyramidal in top layer: add dummy apical for easier matching to interneurons
        if u_target is not None:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"] + self.gsom * u_target) / (
                self.gl + self.gb + self.gsom)
        else:
            self.u_pyr["steadystate"] = (self.gb * self.u_pyr["basal"]) / (
                self.gl + self.gb)

        #### compute changes

        #if self.le:
        #    self.u_pyr["udot"] = (self.gl + self.gb + self.gsom) * (self.u_pyr["steadystate"] - self.u_pyr["soma"])
        #    self.du_pyr = self.u_pyr["udot"] * self.dt
        #else:
        self.u_pyr["udot"] = (self.gl + self.gb + self.gsom) * (self.u_pyr["steadystate"] - self.u_pyr["soma"])
        self.du_pyr = self.u_pyr["udot"] * self.dt
            #self.du_pyr = self.dt * (-self.gl * self.u_pyr["soma"] + self.gb * (
            #    self.u_pyr["basal"] - self.u_pyr["soma"]) + noise_on * self.noise * np.random.normal(
            #    size=len(self.u_pyr["soma"])))
            #if u_target is not None:
            #    self.du_pyr += self.dt * self.gsom * (u_target - self.u_pyr["soma"])

        # calc weight updates (lowpass weight changes)
        if learning_on:
            self.update_weights = True
            if self.le:
                gtot = self.gl + self.gb
                tau_pyr = 1. / (self.gl + self.gb + self.gsom)
                u_new_pyr_forw = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
                self.Delta_up = np.outer(
                    self.act(u_new_pyr_forw) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf)
            else:
                gtot = self.gl + self.gb
                u_new = self.u_pyr["soma"] + self.du_pyr
                self.Delta_up = np.outer(
                    self.act(u_new) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf)
                #self.dDelta_up = self.dt / self.tau_w * (- self.Delta_up + np.outer(
                #    self.act(u_new) - self.act(self.gb / gtot * self.u_pyr["basal"]), r_in_buf))
        else:
            self.update_weights = False

    def apply(self):
        # apply changes to soma potential
        if self.le:
            tau_pyr = 1. / (self.gl + self.gb + self.gsom)
            # important: update u_forw before updating u_soma!
            self.u_pyr["forw"][:] = self.u_pyr["soma"] + tau_pyr * self.u_pyr["udot"]
        self.u_pyr["soma"] += self.du_pyr
        # apply weight updates
        if self.update_weights:
            #if self.le:
            #    self.W_up += self.dt * self.eta["up"] * self.Delta_up
            #    self.W_up[self.W_up > self.w_max] = self.w_max
            #    self.W_up[self.W_up < -1 * self.w_max] = -1 * self.w_max
            #else:
            self.W_up += self.dt * self.eta["up"] * self.Delta_up
            self.W_up[self.W_up > self.w_max] = self.w_max
            self.W_up[self.W_up < -1 * self.w_max] = -1 * self.w_max
                #self.Delta_up += self.dDelta_up

    def reset(self, reset_weights=True):
        self.u_pyr["basal"].fill(0)
        self.u_pyr["soma"].fill(0)
        self.u_pyr["steadystate"].fill(0)
        self.u_pyr["forw"].fill(0)
        self.u_pyr["udot"].fill(0)

        if reset_weights:
            self.W_up = (np.rand_like(self.W_up) - 0.5) * 2 * self.params["init_weights"]["up"]

        self.Delta_up.fill(0)


class Net:

    def __init__(self, params, act=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.params = params
        self.layer = []
        if act is None:
            self.act = u.soft_relu
        else:
            self.act = act
        dims = params["dims"]
        self.dims = dims
        bias = params["bias"]["on"]
        bias_val = params["bias"]["val"]
        self.latent_eq = params["latent_eq"]
        eta = {}
        # construct all layers
        for n in range(1, len(dims) - 1):
            eta["up"] = params["eta"]["up"][n - 1]
            eta["pi"] = params["eta"]["pi"][n - 1]
            eta["ip"] = params["eta"]["ip"][n - 1]
            self.layer += [Layer(dims[n], dims[n - 1], dims[n + 1], eta, params, self.act, bias, bias_val, self.latent_eq)]
        # construct output layer
        eta["up"] = params["eta"]["up"][-1]
        eta["pi"] = params["eta"]["pi"][-1]
        eta["ip"] = params["eta"]["ip"][-1]
        self.layer += [OutputLayer(dims[-1], dims[-2], eta, params, self.act, bias, bias_val, self.latent_eq)]
        # print couplings
        print("feedback-couplings: lambda_out = %f, lambda_inter = %f, lambda_hidden = %f"
              % (params["gsom"] / (params["gl"] + params["gb"] + params["gsom"]),
                 params["gsom"] / (params["gl"] + params["gd"] + params["gsom"]),
                 params["ga"] / (params["gl"] + params["gb"] + params["ga"])))

    def reflect(self):
        '''
        Initialize network in self-predicting state. Forwards and top-down connections are kept.
        '''
        for i in range(len(self.layer) - 1):
            l = self.layer[i]
            l_n = self.layer[i + 1]
            l.W_pi = - l.W_down.copy()
            #if self.latent_eq:
            #    l.W_ip = l_n.W_up.copy() * l_n.gb / (l_n.gl + l_n.ga + l_n.gb) * (l.gl + l.gd) / l.gd
            #else:
            l.W_ip = l_n.W_up.copy() * l_n.gb / (l_n.gl + l_n.ga + l_n.gb) * (l.gl + l.gd) / l.gd

    def update_eta(self, new_eta):
        print("updating learning rates")
        eta = {}
        for n in range(0, len(self.dims) - 1):
            eta["up"] = new_eta["up"][n]
            eta["pi"] = new_eta["pi"][n]
            eta["ip"] = new_eta["ip"][n]
            print('previos eta in layer {0}'.format(n))
            print(self.layer[n].eta)
            self.layer[n].eta = eta.copy()
            print('new eta in layer {0}'.format(n))
            print(self.layer[n].eta)
        return

    def load_weights(self, file):
        '''
        load weights from numpy file (generated with dump weights)
        '''
        weights = np.load(file, allow_pickle=True)
        for n in range(0, len(self.layer) - 1):
            l = self.layer[n]
            np.copyto(dst=l.W_up, src=weights[n][0])
            np.copyto(dst=l.W_pi, src=weights[n][1])
            np.copyto(dst=l.W_ip, src=weights[n][2])
            np.copyto(dst=l.W_down, src=weights[n][3])
        np.copyto(dst=self.layer[-1].W_up, src=weights[-1][0])

    def copy_weights(self):
        '''
        return a copy of network weights [[W_up, W_pi, W_ip, W_down], ..., [W_up]]
        '''
        weights = []
        for n in range(0, len(self.layer) - 1):
            l = self.layer[n]
            weights += [[l.W_up.copy(), l.W_pi.copy(), l.W_ip.copy(), l.W_down.copy()]]
        weights += [[self.layer[-1].W_up.copy()]]
        return weights

    def dump_weights(self, file):
        '''
        dump network weights to file
        '''
        np.save(file, self.copy_weights())

    def update_params(self, params):
        for key, item in params.items():
            if key in self.params:
                self.params[key] = item
                print("update %s: %s." % (key, str(item)))
        eta = {}
        for n, l in enumerate(self.layer):
            eta["up"] = self.params["eta"]["up"][n]
            eta["pi"] = self.params["eta"]["pi"][n]
            eta["ip"] = self.params["eta"]["ip"][n]
            l.set_params(self.params, eta)

    def update(self, r_in, u_target, learning_on=True, records=None, noise_on=True):
        '''
        update network potentials by one timestep
        Parameters
        ----------
        r_in:           input rate vector
        u_target:       target potential vector (might be None)
        learning_on:    switch learning on or off
        records:        list of all quantities to track. If provided, a list for each layer, starting from the hidden layer,
                        is expected, eg. [["pyr_soma", "pyr_basal"], ["pyr_apical", "inn_soma", "inn_dendrite"], ...]
                        See implementation of run for which quantities are available
        noise_on:       switch noise source in network dynamics on or off

        '''
        if self.latent_eq:
            self.layer[0].update(r_in, self.layer[1].u_pyr["forw"], learning_on, noise_on=noise_on)
            for n in range(1, len(self.layer) - 1):
                self.layer[n].update(self.act(self.layer[n - 1].u_pyr["forw"]), self.layer[n + 1].u_pyr["forw"],
                                     learning_on, noise_on=noise_on)
            self.layer[-1].update(self.act(self.layer[-2].u_pyr["forw"]), u_target, learning_on, noise_on=noise_on)
        else:
            self.layer[0].update(r_in, self.layer[1].u_pyr["soma"], learning_on, noise_on=noise_on)
            for n in range(1, len(self.layer) - 1):
                self.layer[n].update(self.act(self.layer[n - 1].u_pyr["soma"]), self.layer[n + 1].u_pyr["soma"],
                                     learning_on, noise_on=noise_on)
            self.layer[-1].update(self.act(self.layer[-2].u_pyr["soma"]), u_target, learning_on, noise_on=noise_on)

        for i, layer in enumerate(self.layer):
            layer.apply()
            if records is not None and not records == []:
                for _, r in records[i].items():
                    r.record()


    def run(self, in_seq, trgt_seq=None, reset_weights=False, val_len=0, metric=None, rec_quants=None, rec_dt=0.0,
            learning_off=False, info_update=100, breadcrumbs=None):
        '''
        Run a full simulation for the whole sequence of input patterns in in_seq.
        Input and target patterns are transitioned smoothly by lowpass filtering.
        Nudging can be switched on or off per pattern.
        If val_len > 0 and trgt_seq is not None, target patterns with disabled nudging are used for validation.
        Parameters
        ----------
        in_seq:         sequence of input patterns. Expects N x dim_0 (input layer size) array.
        trgt_seq:       sequence of target potentials (might be None). Expects N x (dim_out+1) array. Last column has
                        to take values in {0,1}, determining if nudging is switched on (1) or off (0) for each pattern.
        reset_weights:  randomly reset weight connections before starting (soma potentials are always reset to zero)?
        val_len:        if val_len > 0 and trgt_seq is not None all patterns for which trgt_seq[:,-1] is Zero are used
                        for validation. In this case, chunks of length val_len for which trgt_seq[:,-1]==0 are expected.
                        During validation learning and noise is always disabled.
        metric:         metric used during validation
        rec_quants:     list of all quantities to track. If provided, a list for each layer, starting from the hidden layer,
                        is expected, eg. [["pyr_soma", "pyr_basal"], ["pyr_apical", "inn_soma", "inn_dendrite"], ...].
        rec_dt:         specifies time resolution for tracking of rec_quants and smoothed input and teaching traces.
        learning_off:   switch learning off
        info_update:    print info about remaining time after each info_update patterns
        breadcrumbs:    list of pattern indices after which all network weights should be saved

        Returns
        -------
        [(recorded quantities; if rec_quants!=None),
         (time array with resolution dT, the generated (smoothed) input rate trace recorded with rec_dt,
            (the generated (smoothed) target potential trace recorded with rec_dt; if trgt_seq!=None); if rec_dt>0),
         (saved weights; if breadcrumbs!=None),
         sequence of outputs (averaged soma potentials of output layer) with size N x dim_out,
         (N x 2 array containing the validation results [output layer MSE, metric]; if val_len>0)]
        '''
        #### prepare run
        # record signals with time resolution rec_dt -> compress actual data
        n_pattern = int(np.round(self.params["t_pattern"] / self.params["dt"]))  # length of one input pattern
        compress_len = int(np.round(np.round(rec_dt / self.params["dt"])))  # number of samples to average over
        print("t_pattern: %.3f ms,\trec_dt: %.3f ms"%(n_pattern * self.params["dt"], compress_len * self.params["dt"]))
        if rec_dt > 0:
            rec_len = int(
                np.ceil(len(in_seq) * n_pattern / compress_len))  # number of averaged samples to record (initial value is ignored!)
        records = []

        n_out_lag = round(
            self.params["out_lag"] / self.params["dt"])  # steps to wait per pattern before filling output buffer
        n_learning_lag = round(
            self.params["learning_lag"] / self.params["dt"])  # steps to wait per pattern before enabling learning
        if n_out_lag >= n_pattern:
            raise Exception("output lag to big!")
        if n_learning_lag >= n_pattern:
            raise Exception("learning lag to big!")

        print("out-lag: %.3f ms,\tlearning-lag: %.3f ms"%(n_out_lag * self.params["dt"], n_learning_lag * self.params["dt"]))

        r_in = in_seq[0].copy()  # current input rates
        if trgt_seq is not None:
            if len(trgt_seq) != len(in_seq):
                raise Exception("input and target sequence mismatch")
            u_trgt = trgt_seq[0, :-1].copy()  # current target potentials

        out_seq = np.zeros((len(in_seq), self.params["dims"][-1]),
                           dtype=dtype)  # sequence of outputs generated by the network

        # store validation results
        if val_len > 0:
            val_res = []  # [mse of val result, metric of val result]

        #leave breadcrumbs (weights)
        if breadcrumbs is not None:
            weights = []

        # reset/initialize and add trackers for quantities that should be recorded
        for i in range(len(self.layer)):
            l = self.layer[i]
            l.reset(reset_weights)
            if rec_quants is None:
                continue
            rq = rec_quants[i]
            rcs = {}
            if "pyr_soma" in rq:
                rcs["pyr_soma"] = u.Tracker(rec_len, l.u_pyr["soma"], compress_len)
            if "pyr_forw" in rq:
                rcs["pyr_forw"] = u.Tracker(rec_len, l.u_pyr["forw"], compress_len)
            if "pyr_udot" in rq:
                rcs["pyr_udot"] = u.Tracker(rec_len, l.u_pyr["udot"], compress_len)
            if "pyr_steadystate" in rq:
                rcs["pyr_steadystate"] = u.Tracker(rec_len, l.u_pyr["steadystate"], compress_len)
            if "pyr_basal" in rq:
                rcs["pyr_basal"] = u.Tracker(rec_len, l.u_pyr["basal"], compress_len)
            if "pyr_apical" in rq:
                rcs["pyr_apical"] = u.Tracker(rec_len, l.u_pyr["apical"], compress_len)
            if "inn_dendrite" in rq:
                rcs["inn_dendrite"] = u.Tracker(rec_len, l.u_inn["dendrite"], compress_len)
            if "inn_soma" in rq:
                rcs["inn_soma"] = u.Tracker(rec_len, l.u_inn["soma"], compress_len)
            if "inn_forw" in rq:
                rcs["inn_forw"] = u.Tracker(rec_len, l.u_inn["forw"], compress_len)
            if "W_up" in rq:
                rcs["W_up"] = u.Tracker(rec_len, l.W_up, compress_len)
            if "W_down" in rq:
                rcs["W_down"] = u.Tracker(rec_len, l.W_down, compress_len)
            if "W_ip" in rq:
                rcs["W_ip"] = u.Tracker(rec_len, l.W_ip, compress_len)
            if "W_pi" in rq:
                rcs["W_pi"] = u.Tracker(rec_len, l.W_pi, compress_len)
            if "Delta_up" in rq:
                rcs["Delta_up"] = u.Tracker(rec_len, l.Delta_up, compress_len)
            if "Delta_ip" in rq:
                rcs["Delta_ip"] = u.Tracker(rec_len, l.Delta_ip, compress_len)
            if "Delta_pi" in rq:
                rcs["Delta_pi"] = u.Tracker(rec_len, l.Delta_pi, compress_len)
            records += [rcs]

        # init trackers for input rates signal and target potentials signal
        if self.latent_eq:
            # target needs to be shifted by 1 timestep for le to work
            nudging_on_last = trgt_seq[0, -1] if trgt_seq is not None else False
            u_trgt_last = np.zeros(trgt_seq[0].shape[0] -1)
        if rec_dt > 0:
            r_in_trc = u.Tracker(rec_len, r_in, compress_len)
            u_trgt_trc = None
            if trgt_seq is not None:
                if self.latent_eq:
                    u_trgt_trc = u.Tracker(rec_len, u_trgt_last, compress_len)
                else:
                    u_trgt_trc = u.Tracker(rec_len, u_trgt, compress_len)

        ####simulate

        start = time.time()
        val_idx = -1
        for seq_idx in range(len(in_seq)):
            nudging_on = trgt_seq[seq_idx, -1] if trgt_seq is not None else False
            if not nudging_on and val_len > 0:
                # this patterns is for validation! (val_idx will be >=0 during validation)
                val_idx += 1
            for i in range(n_pattern):
                # lowpass input rates
                r_in[:] += self.params["dt"] / self.params["tau_0"] * (in_seq[seq_idx] - r_in)
                if rec_dt > 0:
                    r_in_trc.record()
                learning_on = i >= n_learning_lag and not learning_off

                # lowpass target potentials and update network
                if trgt_seq is not None:
                    u_trgt[:] += self.params["dt"] / self.params["tau_0"] * (trgt_seq[seq_idx, :-1] - u_trgt)
                    if rec_dt > 0:# run PyraLNet for a bunch of patterns and records weights
                        u_trgt_trc.record()
                    l_on = learning_on and val_idx < 0 # no learning during validation
                    if self.latent_eq:
                        self.update(r_in, u_trgt_last if nudging_on_last else None, records=records, learning_on=l_on,
                                    noise_on=val_idx < 0)
                    else:
                        self.update(r_in, u_trgt if nudging_on else None, records=records, learning_on=l_on,
                                    noise_on=val_idx < 0)
                    if i >= n_out_lag: # after out_lag start to average output soma potentials for output sequence
                        if self.latent_eq:
                            out_seq[seq_idx] += self.layer[-1].u_pyr["forw"] / (n_pattern - n_out_lag)
                        else:
                            out_seq[seq_idx] += self.layer[-1].u_pyr["soma"] / (n_pattern - n_out_lag)
                else:
                    self.update(r_in, None, records=records, learning_on=learning_on)
                    if i >= n_out_lag:
                        if self.latent_eq:
                            out_seq[seq_idx] += self.layer[-1].u_pyr["forw"] / (n_pattern - n_out_lag)
                        else:
                            out_seq[seq_idx] += self.layer[-1].u_pyr["soma"] / (n_pattern - n_out_lag)

                if self.latent_eq:
                    nudging_on_last = nudging_on
                    u_trgt_last[:] = u_trgt[:]

            # print validation results if finished
            if val_idx >= 0 and val_idx == val_len - 1:
                print("---Validating on %d patterns---" % (val_len))
                pred = out_seq[seq_idx - val_len + 1:seq_idx + 1]
                true = trgt_seq[seq_idx - val_len + 1:seq_idx + 1, :-1]
                mse = np.mean((pred - true) ** 2)
                print("mean squared error: %f" % (mse), flush=True)
                vres = [mse, 0]
                if metric is not None:
                    name, mres = metric(pred, true)
                    print("%s: %f" % (name, mres), flush=True)
                    vres[1] = mres
                val_res += [vres]
                val_idx = -1 # now disable validation mode

            # print some info
            if seq_idx > 0 and seq_idx % info_update == 0:
                print("%d/%d input patterns done. About %s left." % (
                    seq_idx, len(in_seq), u.time_str((len(in_seq) - seq_idx - 1) * (time.time() - start) / info_update)), flush=True)
                start = time.time()

            # leave breadcrumbs
            if breadcrumbs is not None and seq_idx in breadcrumbs:
                print("leave a breadcrumb at pattern (index): %d"%(seq_idx))
                weights += [self.copy_weights()]

            # reset Deltas (weight update matrices) after each pattern if corresponding flag is set
            if self.params["reset_deltas"]:
                for l in self.layer[:-1]:
                    l.Delta_up.fill(0)
                    l.Delta_ip.fill(0)
                    l.Delta_pi.fill(0)
                self.layer[-1].Delta_up.fill(0)
            

        # finalize recordings
        for rcs in records:
            for _, r in rcs.items(): r.finalize()
        if rec_dt > 0:
            r_in_trc.finalize()
            if trgt_seq is not None:
                u_trgt_trc.finalize()

        # return records (with res rec_dt), a time signal (rec_dt), the input rates signal (rec_dt),
        # target pot signal (rec_dt), breadcrumbs, the output sequence and validation results
        ret = []
        if rec_quants is not None:
            ret += [records]
        if rec_dt > 0:
            ret += [np.linspace(self.params["dt"], rec_len * rec_dt, rec_len), r_in_trc.data] # start at dt as initial value is not recorded
            if trgt_seq is not None:
                ret += [u_trgt_trc.data]
        if breadcrumbs is not None:
            ret += [weights]
        ret += [out_seq]
        if val_len > 0:
            ret += [np.array(val_res, dtype=dtype)]
        return tuple(ret) if len(ret) > 1 else ret[0]


    def train(self, X_train, Y_train, X_val, Y_val, n_epochs, val_len, n_out, classify, u_high=1.0,
              u_low=0.1, rec_quants=None, rec_dt=0.0,
              vals_per_epoch=1, reset_weights=False, info_update=100, metric=None, breadcrumbs=None):
        '''
        Create an input and target sequence from training and validation datasets and run simulation.
        Per epoch the code tries to insert vals_per_epoch validation chunks. Training and validation chunks are
        alternating, with the first chunk being a training chunk and the last a validation chunk (if vals_per_epoch>0).
        If given potentials or weights can be recorded with rec_quants and network weights can be saved after
        specified epochs (breadcrumbs).
        Parameters
        ----------
        X_train:        Training input rates dataset
        Y_train:        Training dataset - true labels (classification) or true signals (regression)
        X_val:          Validation input rates dataset
        Y_val:          Validation dataset - true labels or true signals (for regression)
        n_epochs:       Number of epochs. Each epoch the training set is shuffled and presented again
        val_len:        Number of patterns (drawn randomly from validation sets) per validation chunk
        n_out:          Output layer dimension.
        classify:       True for classification task, false for regression task
        u_high:         Target potential corresponding to logical high (classification task)
        u_low:          Target potential corresponding to logical false (classification task)
        rec_quants:     Quantities to record during simulation. List of quantities per layer, starting from first hidden
                        layer, eg. [["pyr_soma", "pyr_basal"], ["pyr_apical", "inn_soma", "inn_dendrite"], ...]. See run
                        for all quantities that are available.
        rec_dt:         Time resolution for recording
        vals_per_epoch: Try to have such many validation chunks per epoch. Actual number will be printed by code.
        reset_weights:  Randomly reinitialize weights before training
        info_update:    Print info about remaining time after info_update patterns
        metric:         Metric to be used for validation
        breadcrumbs:    List of epoch indices after which network weights will be saved. Can include Zero to save
                        initial weights.

        Returns
        -------
        Like run but last return value (val_res in run) is replaced by list containing all validation results, where a
        row is given by [training patterns see so far, mse, metric].
        '''

        assert len(X_train) > vals_per_epoch
        assert len(X_train) == len(Y_train)
        assert len(X_val) == len(Y_val)

        n_features = X_train.shape[1]

        assert n_features == X_val.shape[1]
        assert n_features == self.dims[0]
        assert n_out == self.dims[-1]

        if classify:
            assert len(Y_train.shape) == 1
            assert len(Y_val.shape) == 1
        else:
            assert (len(Y_train.shape) == 1 and n_out == 1) or Y_train.shape[1] == n_out
            assert (len(Y_val.shape) == 1 and n_out == 1) or Y_val.shape[1] == n_out

        # try to split training set to have vals_per_epoch validations per epoch
        len_split_train = int(round(len(X_train) / vals_per_epoch))  # validation after each split
        vals_per_epoch = len(X_train) // len_split_train
        print("%d validations per epoch" % (vals_per_epoch))

        len_per_ep = vals_per_epoch * val_len + len(X_train)
        length = len_per_ep * n_epochs

        breadcrumbs_ind = None
        if breadcrumbs is not None:
            breadcrumbs_ind = np.clip(len_per_ep*np.array(breadcrumbs)-1, a_min=0, a_max=None) # last pattern index of corresponding epoch
            print(breadcrumbs_ind)

        r_in_seq = np.zeros((length, n_features))
        val_res = np.zeros((vals_per_epoch * n_epochs, 3), dtype=dtype)  # [number of training patterns seen, mse of val result, metric of val result]

        if classify:
            target_seq = np.ones((length, n_out), dtype=dtype) * u_low
        else:
            target_seq = np.zeros((length, n_out), dtype=dtype)

        nudging_on = np.ones((length, 1), dtype=dtype)
        val_idc = np.zeros((len(val_res), val_len))  # indices of validation patterns

        # per epoch training and validation chunks will alternate. Each epoch starts with a training chunks and ends
        # with a validation chunk (if vals_per_epch>0). Accordingly, the last training chunk might differ in size from
        # the previous ones (if len(X_train)/vals_per_epoch does not give an integer).
        for n in range(n_epochs):
            perm_train = np.random.permutation(len(X_train))
            left = n * len_per_ep
            left_tr = 0
            for k in range(vals_per_epoch):
                # start with a training block
                if k == vals_per_epoch - 1:
                    right_tr = len(X_train)
                else:
                    right_tr = left_tr + len_split_train
                right = left + right_tr - left_tr
                r_in_seq[left: right] = X_train[perm_train[left_tr:right_tr]]
                if classify:
                    target_seq[np.arange(left, right), 1 * Y_train[
                        perm_train[left_tr:right_tr]]] = u_high  # enforce Y_train is an integer array!
                else:
                    target_seq[left:right] = Y_train[perm_train[left_tr:right_tr]]

                # add validation block
                perm_val = np.random.permutation(len(X_val))[:val_len]
                left = right
                right = left + val_len
                r_in_seq[left: right] = X_val[perm_val]
                if classify:
                    target_seq[
                        np.arange(left, right), 1 * Y_val[perm_val]] = u_high  # enforce Y_val is an integer array!
                else:
                    target_seq[left:right] = Y_val[perm_val]
                nudging_on[left: right, 0] = False
                val_res[vals_per_epoch * n + k, 0] = right_tr + n * len(X_train)
                val_idc[vals_per_epoch * n + k] = np.arange(left, right)
                left = right
                left_tr = right_tr

        target_seq = np.hstack((target_seq, nudging_on))

        ret = self.run(r_in_seq, trgt_seq=target_seq, reset_weights=reset_weights, val_len=val_len, metric=metric,
                       rec_quants=rec_quants, rec_dt=rec_dt, info_update=info_update, breadcrumbs=breadcrumbs_ind)
        val_res[:, 1:] = ret[-1]  # valres

        return ret[:-1] + tuple([val_res])

    def evaluate(self, X_train, Y_train, X_test, Y_test, n_out, classify, u_high=1.0,
                 u_low=0.1, rec_quants=None, rec_dt=0.0, metric=None):

        assert len(X_train) == len(Y_train)
        assert len(X_test) == len(Y_test)

        n_features = X_train.shape[1]

        assert n_features == X_test.shape[1]
        assert n_features == self.dims[0]
        assert n_out == self.dims[-1]

        if classify:
            assert len(Y_train.shape) == 1
            assert len(Y_test.shape) == 1
        else:
            assert (len(Y_train.shape) == 1 and n_out == 1) or Y_train.shape[1] == n_out
            assert (len(Y_test.shape) == 1 and n_out == 1) or Y_test.shape[1] == n_out

        # Evaluation on Training set
        if classify:
            target_seq = np.ones((len(X_train), n_out), dtype=dtype) * u_low
        else:
            target_seq = np.zeros((len(X_train), n_out), dtype=dtype)
        nudging_on = np.zeros((len(X_train), 1), dtype=dtype)

        perm_train = np.random.permutation(len(X_train))
        r_in_seq = X_train[perm_train]
        if classify:
            target_seq[np.arange(len(X_train)), 1 * Y_train[perm_train]] = u_high  # enforce Y_train is an integer array!
        else:
            target_seq[:] = Y_train[perm_train]
        target_seq = np.hstack((target_seq, nudging_on))
        ret_train = self.run(r_in_seq, trgt_seq=target_seq, reset_weights=False, val_len=0, metric=metric,
                       rec_quants=rec_quants, rec_dt=rec_dt, breadcrumbs=None, learning_off=True)
        out_seq = ret_train[-1]
        print("Evaluation on training set yielded:")
        mse = np.mean((target_seq[:, :-1] - out_seq) ** 2)
        print("mean squared error: %f" % (mse))
        train_res = [mse, 0]
        if metric is not None:
            name, mres = metric(target_seq[:, :-1], out_seq)
            print("%s: %f" % (name, mres))
            train_res[1] = mres

        # Evaluation on Test set
        if classify:
            target_seq = np.ones((len(X_test), n_out), dtype=dtype) * u_low
        else:
            target_seq = np.zeros((len(X_test), n_out), dtype=dtype)
        nudging_on = np.zeros((len(X_test), 1), dtype=dtype)

        perm_test = np.random.permutation(len(X_test))
        r_in_seq = X_test[perm_test]
        if classify:
            target_seq[np.arange(len(X_test)), 1 * Y_test[perm_test]] = u_high  # enforce Y_test is an integer array!
        else:
            target_seq[:] = Y_test[perm_test]
        target_seq = np.hstack((target_seq, nudging_on))
        ret_test = self.run(r_in_seq, trgt_seq=target_seq, reset_weights=False, val_len=0, metric=metric,
                       rec_quants=rec_quants, rec_dt=rec_dt, breadcrumbs=None, learning_off=True)
        out_seq = ret_test[-1]
        print("Evaluation on test set yielded:")
        mse = np.mean((target_seq[:, :-1] - out_seq) ** 2)
        print("mean squared error: %f" % (mse))
        test_res = [mse, 0]
        if metric is not None:
            name, mres = metric(target_seq[:, :-1], out_seq)
            print("%s: %f" % (name, mres))
            test_res[1] = mres

        return ret_train, train_res, ret_test, test_res


    def selfpredict(self, X_train, Y_train, n_out, n_epochs, classify, u_high=1.0,
                 u_low=0.1, rec_quants=None, rec_dt=0.0, metric=None):

        assert len(X_train) == len(Y_train)

        n_features = X_train.shape[1]

        assert n_features == self.dims[0]
        assert n_out == self.dims[-1]

        if classify:
            assert len(Y_train.shape) == 1
        else:
            assert (len(Y_train.shape) == 1 and n_out == 1) or Y_train.shape[1] == n_out

        # check if all forward etas are zero
        all_zero = True
        for l in self.layer:
            assert l.eta["up"] == 0

        # Evaluation on Training set
        length = len(X_train) * n_epochs

        r_in_seq = np.zeros((length, n_features))

        if classify:
            target_seq = np.ones((length, n_out), dtype=dtype) * u_low
        else:
            target_seq = np.zeros((length, n_out), dtype=dtype)

        nudging_on = np.zeros((length, 1), dtype=dtype)

        for n in range(n_epochs):
            perm_train = np.random.permutation(len(X_train))
            left = n * len(X_train)
            right = (n+1) * len(X_train)
            r_in_seq[left: right] = X_train[perm_train]
            if classify:
                target_seq[np.arange(left, right), 1 * Y_train[perm_train]] = u_high  # enforce Y_train is an integer array!
            else:
                target_seq[left:right] = Y_train[perm_train]

#        perm_train = np.random.permutation(len(X_train))
#        r_in_seq = X_train[perm_train]
#        if classify:
#            target_seq[np.arange(len(X_train)), 1 * Y_train[perm_train]] = u_high  # enforce Y_train is an integer array!
#        else:
#            target_seq[:] = Y_train[perm_train]
        target_seq = np.hstack((target_seq, nudging_on))
        ret_train = self.run(r_in_seq, trgt_seq=target_seq, reset_weights=False, val_len=0, metric=metric,
                       rec_quants=rec_quants, rec_dt=rec_dt, breadcrumbs=None, learning_off=False)
        return ret_train

