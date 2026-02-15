import torch
import torch.nn as nn
from torch.distributions import StudentT

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # Rayleigh distribution
        self.h_real_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.h_real_std  = nn.Parameter(torch.tensor(0.16), requires_grad=False)
        self.h_imag_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.h_imag_std  = nn.Parameter(torch.tensor(0.16), requires_grad=False)
        # Student's t distribution for noise (real and imag)
        self.n_real_df    = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.n_real_loc   = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_real_scale = nn.Parameter(torch.tensor(0.03), requires_grad=False)
        self.n_imag_df    = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.n_imag_loc   = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.n_imag_scale = nn.Parameter(torch.tensor(0.03), requires_grad=False)

    def set_params(self,
            h_real_mean=0, h_real_std=0.16, h_imag_mean=0, h_imag_std=0.16,
            n_real_df=3, n_real_loc=0, n_real_scale=0.03,
            n_imag_df=3, n_imag_loc=0, n_imag_scale=0.03,
        ):
        # Rayleigh distribution
        self.h_real_mean = nn.Parameter(torch.tensor(h_real_mean, dtype=torch.float), requires_grad=False)
        self.h_real_std  = nn.Parameter(torch.tensor(h_real_std, dtype=torch.float), requires_grad=False)
        self.h_imag_mean = nn.Parameter(torch.tensor(h_imag_mean, dtype=torch.float), requires_grad=False)
        self.h_imag_std  = nn.Parameter(torch.tensor(h_imag_std, dtype=torch.float), requires_grad=False)
        # Student's t for noise
        self.n_real_df    = nn.Parameter(torch.tensor(float(n_real_df), dtype=torch.float), requires_grad=False)
        self.n_real_loc   = nn.Parameter(torch.tensor(n_real_loc, dtype=torch.float), requires_grad=False)
        self.n_real_scale = nn.Parameter(torch.tensor(n_real_scale, dtype=torch.float), requires_grad=False)
        self.n_imag_df    = nn.Parameter(torch.tensor(float(n_imag_df), dtype=torch.float), requires_grad=False)
        self.n_imag_loc   = nn.Parameter(torch.tensor(n_imag_loc, dtype=torch.float), requires_grad=False)
        self.n_imag_scale = nn.Parameter(torch.tensor(n_imag_scale, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        bs, d = x.shape
        # sample Rayleigh slow fading
        h_real = torch.randn(bs, device=x.device) * self.h_real_std + self.h_real_mean
        h_imag = torch.randn(bs, device=x.device) * self.h_imag_std + self.h_imag_mean
        h = torch.complex(h_real, h_imag)[:, None]
        # sample noise from Student's t
        dist_real = StudentT(df=self.n_real_df, loc=self.n_real_loc, scale=self.n_real_scale)
        dist_imag = StudentT(df=self.n_imag_df, loc=self.n_imag_loc, scale=self.n_imag_scale)
        n_real = dist_real.sample((bs, d))
        n_imag = dist_imag.sample((bs, d))
        n = torch.complex(n_real, n_imag)
        x_hat = h * x + n
        return x_hat
