from torch.distributions import Normal
import torch.nn as nn
import numpy as np
import torch
import os


class Model(nn.Module):
    # Single-tone Doppler channel: constant magnitude, phase rotating at f_D.
    # h(t) = a * exp(j * (2*pi*f_D*t + phi)), then y = h*x + n.

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # doppler: a (magnitude), f_D (Hz)
        self.a   = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.f_D = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.dt = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # noise (Gaussian)
        self.n_real_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_real_std = nn.Parameter(torch.tensor(0.03), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_imag_std = nn.Parameter(torch.tensor(0.03), requires_grad=False)

    def set_params(self, a=1.0, f_D=0.0, dt=1.0,
                   n_real_mean=0.0, n_real_std=0.03,
                   n_imag_mean=0.0, n_imag_std=0.03):
        self.a           = nn.Parameter(torch.tensor(max(float(a), 1e-6), dtype=torch.float), requires_grad=False)
        self.f_D         = nn.Parameter(torch.tensor(float(f_D), dtype=torch.float), requires_grad=False)
        self.dt          = nn.Parameter(torch.tensor(float(dt), dtype=torch.float), requires_grad=False)
        self.n_real_mean = nn.Parameter(torch.tensor(float(n_real_mean), dtype=torch.float), requires_grad=False)
        self.n_real_std  = nn.Parameter(torch.tensor(max(float(n_real_std), 1e-8), dtype=torch.float), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(float(n_imag_mean), dtype=torch.float), requires_grad=False)
        self.n_imag_std  = nn.Parameter(torch.tensor(max(float(n_imag_std), 1e-8), dtype=torch.float), requires_grad=False)

    def forward(self, x, t=None, return_channel=False):
        # extract args
        bs, d = x.shape
        device = x.device
        # doppler
        t = (torch.arange(d) * self.dt)[None] # 1, d
        phi = (torch.rand(bs) * 2 * np.pi)[:, None]   # bs, 1
        h = self.a * np.exp(1j * (2 * np.pi * self.f_D * t + phi)) # bs, d
        # awgn
        n = torch.randn(bs, d) * self.n_real_std + self.n_real_mean + \
            1j * (torch.randn(bs, d) * self.n_imag_std + self.n_imag_mean) # bs, d
        # apply channel transformation
        if return_channel:
            return h, n
        x_hat = h * x + n
        return x_hat

    def save(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{args.model}.pt')
        torch.save(self.state_dict(), path)
        print(f'+ saved {args.model} at {path}')

    def load(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{args.model}.pt')
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print(f'+ saved {args.model} at {path}')
