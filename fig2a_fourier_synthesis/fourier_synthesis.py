import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from le_layers import LELinear
from le_dynamics import LEDynamics
from le_abstract_net import LEAbstractNet


class LEFourierSynthesisNet(LEAbstractNet, torch.nn.Module):

    def __init__(self, *, tau, dt, in_channels, tau_e):
        super().__init__()

        self.fc0 = LELinear(in_channels, 30)
        self.fc1 = LELinear(30, 1)

        self.fc_dynamics0 = LEDynamics(self.fc0, tau=tau, dt=dt, tau_e=tau_e)
        self.fc_dynamics1 = LEDynamics(self.fc1, tau=tau, dt=dt, tau_e=tau_e, phi=torch.tanh, phi_prime=lambda x: 1 - torch.tanh(x) ** 2)

    def compute_output_error_prime(self, u_breve, target, beta):
        return -beta * (target - u_breve)


def construct_sin_inputs(t_max, dt, frequencies):
    x = torch.arange(params['t_max'] // params['dt']).reshape(1, -1)
    x = x.repeat(len(params['frequencies']), 1)
    x = torch.sin(2 * math.pi * x * dt * 1e-3 * torch.Tensor(params['frequencies']).reshape(-1, 1))
    return x.t()


def create_sawtooth_target():
    return 2 * ((0.0 + 0.5 * 1e3 / params['box_length'] *  torch.arange(params['t_max'] // params['dt']) * params['dt'] * 1e-3) % 1.0) - 1.0


params = {
    'seed': 1234,
    't_max': 600,
    'dt': 0.005,
    'tau': 10.,
    'lr': 0.25,
    'frequencies': np.arange(1, 51) * 10.0,
    'beta': 0.1,
    'box_length': 25.,
}


if __name__ == '__main__':

    torch.manual_seed(params['seed'])

    time_steps = int(params['t_max'] // params['dt'])

    target = create_sawtooth_target()

    time = torch.arange(time_steps) * params['dt'] * 1e-3

    model = LEFourierSynthesisNet(tau=params['tau'], dt=params['dt'], in_channels=len(params['frequencies']), tau_e=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=params['dt'] * params['lr'])

    with torch.no_grad():
        x = construct_sin_inputs(params['t_max'], params['dt'], params['frequencies'])
        history = torch.empty((time_steps, 4))
        for time_step in range(time_steps):
            output = model(x[time_step].unsqueeze(0), target[time_step].unsqueeze(0), beta=params['beta'])
            optimizer.step()
            history[time_step, 0] = time[time_step]
            history[time_step, 1] = target[time_step].clone()
            history[time_step, 2] = output.clone()
            history[time_step, 3] = model.fc_dynamics1.u.clone()
            if time_step > time_steps // 3 * 2:
                params['beta'] = 0.0

    subsampling_steps = 100
    plt.plot(time[::subsampling_steps], target[::subsampling_steps], color='k')
    plt.plot(time[::subsampling_steps], history[::subsampling_steps, 2])
    plt.savefig('le_fourier_synthesis.pdf')

    np.save('le_sin.npy', history.detach().numpy())
