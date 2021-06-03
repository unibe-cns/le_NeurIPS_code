import torch

from le_dynamics import LEDynamics


class LEAbstractNet:

    def __init__(self, *, full_forward=False, full_backward=False):
        super().__init__()

        self.use_le = True
        self.full_forward = full_forward
        self.full_backward = full_backward

    def _compute_output_error_prime(self, u_breve, target, beta):
        if target is None:
            return torch.zeros_like(u_breve)
        else:
            return self.compute_output_error_prime(u_breve, target, beta)

    def compute_output_error_prime(self, u_breve, target, beta):
        raise NotImplementedError('please override this method according to the derivative of your loss function.')

    def _initialize_dynamic_variables(self, shape):
        """Create a dummy input and propagate through all layers to initialize
        dynamic variables to the correct shapes; their initial value
        will be set to zero.

        """
        x = torch.Tensor(shape).normal_()
        for layer in self.children_with_dynamics():
            x = layer.initialize_dynamic_variables_to_zero(x)

    def _adjust_batch_dimension(self, batch_size):
        for layer in self.children_with_dynamics():
            layer.adjust_batch_dimension(batch_size)

    def _update_dynamic_variables(self):
        for layer in self.children_with_dynamics():
            layer.update_dynamic_variables()

    def is_not_initialized(self):
        return list(self.children_with_dynamics())[0].is_not_initialized()

    def batch_size_does_not_match(self, batch_size):
        return batch_size != list(self.children_with_dynamics())[0].batch_size()

    def children_with_dynamics(self):
        for value in self.__dict__.values():
            if isinstance(value, LEDynamics):
                yield value

    def forward(self, x, target=None, *, beta=0.0):

        if self.is_not_initialized():
            self._initialize_dynamic_variables(x.shape)

        if self.batch_size_does_not_match(len(x)):
            self._adjust_batch_dimension(len(x))

        layers = list(self.children_with_dynamics())
        for layer_idx in range(len(layers)):
            if layer_idx == 0:
                if len(layers) > 1:
                    layers[layer_idx](x, layers[layer_idx + 1].e_bottom)
                else:
                    layers[layer_idx](x, self._compute_output_error_prime(layers[layer_idx].u_breve, target, beta))
            elif layer_idx > 0 and layer_idx < len(layers) - 1:
                layers[layer_idx](layers[layer_idx - 1].u_breve, layers[layer_idx + 1].e_bottom)
            elif layer_idx == len(layers) - 1:
                layers[layer_idx](layers[layer_idx - 1].u_breve, self._compute_output_error_prime(layers[layer_idx].u_breve, target, beta))
            else:
                assert False  # should never be reached

            if self.full_forward:
                layers[layer_idx].update_dynamic_variables()

        output = layers[-1].u_breve.clone()

        self._update_dynamic_variables()

        if self.full_backward:
            assert self.full_forward
            layers = list(self.children_with_dynamics())
            for layer_idx in range(len(layers) - 1, -1, -1):
                if layer_idx == 0:
                    layers[layer_idx](x, layers[layer_idx + 1].e_bottom)
                elif layer_idx > 0 and layer_idx < len(layers) - 1:
                    layers[layer_idx](layers[layer_idx - 1].u_breve, layers[layer_idx + 1].e_bottom)
                elif layer_idx == len(layers) - 1:
                    layers[layer_idx](layers[layer_idx - 1].u_breve, self._compute_output_error_prime(layers[layer_idx].u_breve, target, beta))
                else:
                    assert False  # should never be reached

                layers[layer_idx].update_dynamic_variables()

        return output
