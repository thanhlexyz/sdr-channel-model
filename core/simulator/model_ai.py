import torch

class Model:

    def __init__(self, args):
        self.args = args
        # Rayleigh distribution
        self.h_real_mean = torch.tensor(0, requires_grad=False)
        self.h_real_std  = torch.tensor(0.16, requires_grad=False)
        self.h_imag_mean = torch.tensor(0, requires_grad=False)
        self.h_imag_std  = torch.tensor(0.16, requires_grad=False)
        # AWGN distribution
        self.n_real_mean = nn.Parameter(0, requires_grad=False)
        self.n_real_std  = nn.Parameter(0.03, requires_grad=False)
        self.n_imag_mean = nn.Parameter(0, requires_grad=False)
        self.n_imag_std  = nn.Parameter(0.03, requires_grad=False)

    def set_params(self,
            h_real_mean=0, h_real_std=0.16, h_imag_mean=0, h_imag_std=0.16,
            n_real_mean=0, n_real_std=0.03, n_imag_mean=0, n_imag_std=0.03,
        ):
        # Rayleigh distribution
        self.h_real_mean = h_real_mean
        self.h_real_std  = h_real_std
        self.h_imag_mean = h_imag_mean
        self.h_imag_std  = h_imag_std
        # AWGN distribution
        self.n_real_mean = n_real_mean
        self.n_real_std  = n_real_std
        self.n_imag_mean = n_imag_mean
        self.n_imag_std  = n_imag_std

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
