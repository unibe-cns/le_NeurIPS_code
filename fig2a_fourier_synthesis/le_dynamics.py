import torch


class LEDynamics:

    def __init__(self, conn, *, tau, dt=None, tau_e=None, phi=None, phi_prime=None, gamma=1.0):
        super().__init__()

        self.conn = conn
        self.tau = tau
        self.dt = dt
        self.tau_e = tau_e
        self.phi = phi
        self.phi_prime = phi_prime
        self.gamma = gamma

        if self.tau is not None:
            assert self.dt is not None

        if self.tau_e is None and self.tau is not None:
            self.tau_e = 10 * self.dt

        if self.phi is not None:
            assert self.phi_prime is not None
        else:
            self.phi = lambda x: x
            self.phi_prime = lambda x: torch.ones_like(x)

        # WARNING: these will be populated on first call to `forward`
        self.u = None
        self.u_breve = None
        self.e_bottom = None

        # WARNING: these will be populated on first call to `forward`
        self.next_u = None
        self.next_u_breve = None
        self.next_e_bottom = None

    def adjust_batch_dimension(self, batch_size):
        assert len(self.u) == len(self.u_breve)
        assert len(self.u) == len(self.e_bottom)

        if self.batch_size() < batch_size:
            u = self.u.clone()
            u_breve = self.u_breve.clone()
            e_bottom = self.e_bottom.clone()
            while len(u) < batch_size:
                u = torch.cat([u, self.u.clone()])
                u_breve = torch.cat([u_breve, self.u_breve.clone()])
                e_bottom = torch.cat([e_bottom, self.e_bottom.clone()])
            self.u = u[:batch_size].clone()
            self.u_breve = u_breve[:batch_size].clone()
            self.e_bottom = e_bottom[:batch_size].clone()
        else:
            self.u = self.u.narrow(0, 0, batch_size)
            self.u_breve = self.u_breve.narrow(0, 0, batch_size)
            self.e_bottom = self.e_bottom.narrow(0, 0, batch_size)

    def batch_size(self):
        return len(self.u)

    def is_not_initialized(self):
        return self.u is None

    def initialize_dynamic_variables_to_zero(self, u_breve_bottom):
        assert self.u is None
        assert self.u_breve is None
        assert self.e_bottom is None
        r_bottom = self.phi(u_breve_bottom)

        I = self.conn(r_bottom)

        self.u = torch.zeros(I.shape)
        self.u_breve = torch.zeros(I.shape)
        self.e_bottom = torch.zeros(r_bottom.shape)
        return I

    def update_dynamic_variables(self):
        self.u = self.next_u.clone()
        self.u_breve = self.next_u_breve.clone()
        self.e_bottom = self.next_e_bottom.clone()

    def __call__(self, u_breve_bottom, e):
        """
        WARNING: e is the error to be added to the membrane, /not/ the
        error in the following layer; it's already been backpropagated

        """
        r_bottom = self.phi(u_breve_bottom)
        r_bottom_prime = self.phi_prime(u_breve_bottom)
        assert r_bottom.shape == r_bottom_prime.shape

        I = self.conn(r_bottom)
        e = e.reshape(I.shape)

        self.conn.compute_grad(r_bottom, e)

        if self.tau is not None:
            delta_u = 1. / self.tau * (-self.u + I + self.gamma * e)
            u = self.u + self.dt * delta_u
            u_breve = self.u + self.tau * delta_u

            if abs(self.tau_e) < 1e-9:
                e_bottom = r_bottom_prime * self.conn.compute_error(r_bottom, e)
            else:
                delta_e_bottom = 1. / self.tau_e * (-self.e_bottom + r_bottom_prime * self.conn.compute_error(r_bottom, e))
                e_bottom = self.e_bottom + self.dt * delta_e_bottom
        else:
            u = I + self.gamma * e
            u_breve = I + self.gamma * e
            e_bottom = r_bottom_prime * self.conn.compute_error(r_bottom, e)

        self.next_u, self.next_u_breve, self.next_e_bottom = u, u_breve, e_bottom
        return u, u_breve, e_bottom
