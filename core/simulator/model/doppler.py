"""
Doppler (complex sinusoid) channel: h(t) = a * exp(j * (2*pi*f_D*t + phi)).
See NOTE.md for the model description.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class Model(nn.Module):
    """
    Single-tone Doppler channel: constant magnitude, phase rotating at f_D.
    h(t) = a * exp(j * (2*pi*f_D*t + phi)), then y = h*x + n.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # Doppler: a (magnitude), f_D (Hz), phi (rad)
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.f_D = nn.Parameter(torch.tensor(0.0), requires_grad=False)   # Hz
        self.phi = nn.Parameter(torch.tensor(0.0), requires_grad=False)   # rad
        self.dt = nn.Parameter(torch.tensor(1e-4), requires_grad=False)    # time step between samples (s)
        # Noise (Gaussian)
        self.n_real_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_real_std = nn.Parameter(torch.tensor(0.03), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_imag_std = nn.Parameter(torch.tensor(0.03), requires_grad=False)

    def set_params(self,
                   a=1.0, f_D=0.0, phi=0.0, dt=1e-4,
                   n_real_mean=0.0, n_real_std=0.03,
                   n_imag_mean=0.0, n_imag_std=0.03):
        self.a = nn.Parameter(torch.tensor(max(float(a), 1e-6), dtype=torch.float), requires_grad=False)
        self.f_D = nn.Parameter(torch.tensor(float(f_D), dtype=torch.float), requires_grad=False)
        self.phi = nn.Parameter(torch.tensor(float(phi), dtype=torch.float), requires_grad=False)
        self.dt = nn.Parameter(torch.tensor(float(dt), dtype=torch.float), requires_grad=False)
        self.n_real_mean = nn.Parameter(torch.tensor(float(n_real_mean), dtype=torch.float), requires_grad=False)
        self.n_real_std = nn.Parameter(torch.tensor(max(float(n_real_std), 1e-8), dtype=torch.float), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(float(n_imag_mean), dtype=torch.float), requires_grad=False)
        self.n_imag_std = nn.Parameter(torch.tensor(max(float(n_imag_std), 1e-8), dtype=torch.float), requires_grad=False)

    def forward(self, x, t=None):
        """
        x: (bs, d) complex
        t: (bs,) or (bs, 1) time in seconds; if None, t = [0, dt, 2*dt, ...]
        """
        bs, d = x.shape
        device = x.device
        if t is None:
            t = torch.arange(bs, device=device, dtype=torch.float32) * self.dt
        else:
            t = t.squeeze()
            if t.dim() == 0:
                t = t.unsqueeze(0)
        # h(t) = a * exp(j * (2*pi*f_D*t + phi))
        angle = 2 * torch.pi * self.f_D * t + self.phi
        a = self.a.clamp(min=1e-6)
        h = a * torch.exp(1j * angle)
        h = h.to(x.dtype).unsqueeze(1)  # (bs, 1)
        # noise (Gaussian)
        dist_real = Normal(loc=self.n_real_mean, scale=self.n_real_std)
        dist_imag = Normal(loc=self.n_imag_mean, scale=self.n_imag_std)
        n_real = dist_real.sample((bs, d))
        n_imag = dist_imag.sample((bs, d))
        n = torch.complex(n_real, n_imag)
        x_hat = h * x + n
        return x_hat
