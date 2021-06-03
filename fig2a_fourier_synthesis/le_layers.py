import torch
import torch.nn.functional as F


class LEMaxPool2d(torch.nn.MaxPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_indices = True

    def forward(self, x):
        if self.kernel_size != self.stride:
            raise NotImplementedError('kernel size and stride must be identical')

        self.input_shape = x.shape
        o, i = super().forward(x)
        self.i = i
        return o

    def compute_error(self, r_bottom, e):
        return F.max_unpool2d(e, self.i, self.kernel_size, self.stride, self.padding, self.input_shape)

    def compute_grad(self, r_bottom, e):
        pass


class LEConv2d(torch.nn.Conv2d):

    def compute_error(self, r_bottom, e):
        return torch.nn.grad.conv2d_input(r_bottom.shape, self.weight, e, self.stride, self.padding, self.dilation, self.groups)

    def compute_grad(self, r_bottom, e):
        self.weight._grad = 1. / len(r_bottom) * torch.nn.grad.conv2d_weight(r_bottom, self.weight.shape, e, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            self.bias._grad = e.sum(dim=(2, 3)).mean(0)


class LELinear(torch.nn.Linear):

    def compute_error(self, r_bottom, e):
        return torch.mm(e, self.weight)

    def compute_grad(self, r_bottom, e):
        self.weight._grad = torch.einsum('bi, bj->bij', e, r_bottom).mean(0)
        if self.bias is not None:
            self.bias._grad = e.mean(0)


class LEAvgPool2d(torch.nn.AvgPool2d):

    def forward(self, x):
        assert len(x.shape) == 4
        in_channels = x.shape[1]
        w = torch.ones(in_channels, 1, self.kernel_size, self.kernel_size)
        return 1./self.kernel_size ** 2 * F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding, groups=in_channels)

    def compute_error(self, r_bottom, e):
        assert len(r_bottom.shape) == 4
        in_channels = r_bottom.shape[1]
        w = torch.ones(in_channels, 1, self.kernel_size, self.kernel_size)
        return 1./self.kernel_size ** 2 * torch.nn.grad.conv2d_input(r_bottom.shape, w, e, self.stride, self.padding, groups=in_channels)

    def compute_grad(self, r_bottom, e):
        pass
