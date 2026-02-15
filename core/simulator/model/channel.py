import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # Rayleigh distribution
        self.h_real_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.h_real_std  = nn.Parameter(torch.tensor(0.16), requires_grad=False)
        self.h_imag_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.h_imag_std  = nn.Parameter(torch.tensor(0.16), requires_grad=False)
        # AWGN distribution
        self.n_real_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.n_real_std  = nn.Parameter(torch.tensor(0.03), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.n_imag_std  = nn.Parameter(torch.tensor(0.03), requires_grad=False)

    def set_params(self,
            h_real_mean=0, h_real_std=0.16, h_imag_mean=0, h_imag_std=0.16,
            n_real_mean=0, n_real_std=0.03, n_imag_mean=0, n_imag_std=0.03,
        ):
        # Rayleigh distribution
        self.h_real_mean = nn.Parameter(torch.tensor(h_real_mean), requires_grad=False)
        self.h_real_std  = nn.Parameter(torch.tensor(h_real_std), requires_grad=False)
        self.h_imag_mean = nn.Parameter(torch.tensor(h_imag_mean), requires_grad=False)
        self.h_imag_std  = nn.Parameter(torch.tensor(h_imag_std), requires_grad=False)
        # AWGN distribution
        self.n_real_mean = nn.Parameter(torch.tensor(n_real_mean), requires_grad=False)
        self.n_real_std  = nn.Parameter(torch.tensor(n_real_std), requires_grad=False)
        self.n_imag_mean = nn.Parameter(torch.tensor(n_imag_mean), requires_grad=False)
        self.n_imag_std  = nn.Parameter(torch.tensor(n_imag_std), requires_grad=False)

    def forward(self, x):
        # extract x dim
        bs, d = x.shape
        # sample Rayleigh slow fading
        h_real = torch.random.randn(bs) * self.h_real_std + self.h_real_mean
        h_imag = torch.random.randn(bs) * self.h_imag_std + self.h_imag_mean
        h = torch.complex(h_real, h_std)[:, None]
        # sample AWGN noise
        h_real = torch.random.randn(bs, d) * self.n_real_std + self.n_real_mean
        h_imag = torch.random.randn(bs, d) * self.n_imag_std + self.n_imag_mean
        n = torch.complex(n_real, n_std)
        # channel transformation
        x_hat = h * x + n
        return x_hat
