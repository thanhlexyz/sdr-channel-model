import torch.nn as nn
import torch
import numpy as np

class Model(nn.Module):
    """
    i.i.d. Rayleigh fading channel: Re(h) ~ N(h_mean_real, h_std_real²), Im(h) ~ N(h_mean_imag, h_std_imag²);
    noise n ~ CN(0, σ²). Output: y = h * x + n.
    """

    def __init__(self, args=None, sigma_init=0.005,
                 h_std_real_init=1.0, h_std_imag_init=1.0,
                 h_mean_real_init=0.0, h_mean_imag_init=0.0):
        super().__init__()
        self.args = args
        # Learnable noise power for AWGN noise (σ²)
        self.log_sigma = nn.Parameter(torch.tensor(np.log(max(sigma_init, 1e-6)), dtype=torch.float32))
        # Learnable std for real and imag parts of fading (each > 0)
        self.log_h_std_real = nn.Parameter(torch.tensor(np.log(max(h_std_real_init, 1e-6)), dtype=torch.float32))
        self.log_h_std_imag = nn.Parameter(torch.tensor(np.log(max(h_std_imag_init, 1e-6)), dtype=torch.float32))
        # Learnable complex mean for fading
        self.h_mean_real = nn.Parameter(torch.tensor(float(h_mean_real_init), dtype=torch.float32))
        self.h_mean_imag = nn.Parameter(torch.tensor(float(h_mean_imag_init), dtype=torch.float32))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def h_std_real(self):
        return torch.exp(self.log_h_std_real)

    @property
    def h_std_imag(self):
        return torch.exp(self.log_h_std_imag)

    @property
    def h_mean(self):
        return torch.complex(self.h_mean_real, self.h_mean_imag)

    def forward(self, x):
        # extract args
        args = self.args
        # ensure x is complex
        assert torch.is_complex(x)
        # 1. randomly draw AWGN noise: n ~ CN(0, σ²Ik)
        noise_std = torch.sqrt(self.sigma / 2.0)
        n_real = torch.randn(x.shape, device=args.device, dtype=torch.float32) * noise_std
        n_imag = torch.randn(x.shape, device=args.device, dtype=torch.float32) * noise_std
        n = torch.complex(n_real, n_imag)
        # 2. randomly draw i.i.d. Rayleigh fading: Re(h) ~ N(h_mean_real, h_std_real²), Im(h) ~ N(h_mean_imag, h_std_imag²)
        h_shape = x.shape[:-1] + (1,)
        h_real = torch.randn(h_shape, device=args.device, dtype=torch.float32) * self.h_std_real
        h_imag = torch.randn(h_shape, device=args.device, dtype=torch.float32) * self.h_std_imag
        h = self.h_mean + torch.complex(h_real, h_imag)
        # 3. apply channel effect: η(z) = hz + n
        # For slow fading, h is scalar, so multiply element-wise (broadcasting)
        x = h * x + n
        return x

    def log_prob_z_hat_given_z(self, z, z_hat):
        """
        Log probability of z_hat given z under the channel model.
        z_hat = h*z + n with h ~ CN(h_mean, h_std_real² + h_std_imag²), n ~ CN(0, σ²).
        So z_hat_i | z_i ~ CN(h_mean*z_i, |z_i|²*(h_std_real² + h_std_imag²) + σ²).
        Returns log p(z_hat | z) summed over batch and symbols (for NLL), or per-sample mean if needed.
        """
        # z, z_hat: (B, 64) complex
        h_mean = self.h_mean  # complex scalar
        var_h = self.h_std_real ** 2 + self.h_std_imag ** 2  # complex variance of h
        sigma_sq = self.sigma ** 2
        eps = 1e-10
        # Per-symbol variance: |z_i|^2 * var_h + sigma^2
        z_sq = torch.abs(z) ** 2  # (B, 64)
        var_per = z_sq * var_h + sigma_sq + eps  # (B, 64)
        mean_per = h_mean * z  # (B, 64)
        # Complex circular Gaussian: log p = -log(pi) - log(var) - |z_hat - mean|^2 / var
        diff_sq = torch.abs(z_hat - mean_per) ** 2  # (B, 64)
        log_var = torch.log(var_per)
        log_p = -np.log(np.pi) - log_var - diff_sq / var_per
        return log_p  # (B, 64)


class ZToZHatModel(nn.Module):
    """Maps z (real vector of Re/Im stacked) to z_hat to minimize KL between model output and target."""

    def __init__(self, args, hidden_dim=256, num_layers=3):
        super().__init__()
        self.args = args
        # Input/output: 64 complex -> 128 real (Re; Im stacked)
        dim = 64 * 2  # 128
        layers = []
        in_d = dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_flat):
        # z_flat: (B, 128)
        return self.net(z_flat)
