import torch.nn as nn
import torch
import numpy as np


def _moving_avg_last_l(h_real, h_imag, L):
    """Compute moving average of last L steps. h_real, h_imag: (B, T). Returns (B, T, 2) for (ma_real, ma_imag)."""
    B, T = h_real.shape
    device = h_real.device
    ma_real = torch.zeros(B, T, device=device, dtype=h_real.dtype)
    ma_imag = torch.zeros(B, T, device=device, dtype=h_imag.dtype)
    for t in range(T):
        start = max(0, t - L)
        if start < t:
            ma_real[:, t] = h_real[:, start:t].mean(dim=1)
            ma_imag[:, t] = h_imag[:, start:t].mean(dim=1)
    return torch.stack([ma_real, ma_imag], dim=-1)  # (B, T, 2)


class ARChannelModel(nn.Module):
    """
    Autoregressive model for channel h (complex): given past h and moving average of last L,
    predict next h as Gaussian on real and imag (mean + std each).
    Input at step t: [h_{t-1}_real, h_{t-1}_imag, ma_real, ma_imag]. Step 0 uses zeros.
    Output at step t: (mean_real, mean_imag, std_real, std_imag) for h_t.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_symbol = getattr(args, 'n_symbol', 64)
        self.ma_window = getattr(args, 'ma_window', 8)
        input_size = 4  # h_prev real, imag + ma real, imag
        hidden_size = getattr(args, 'ar_hidden', 128)
        self.hidden_size = hidden_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 4)  # mean_real, mean_imag, log_std_real, log_std_imag

    def forward(self, x):
        """
        x: (B, n_symbol, 4) — per step [h_prev_real, h_prev_imag, ma_real, ma_imag]
        Returns: (B, n_symbol, 4) as (mean_real, mean_imag, log_std_real, log_std_imag)
        """
        B, T, _ = x.shape
        h = torch.tanh(self.embed(x))  # (B, T, hidden)
        out, _ = self.gru(h)  # (B, T, hidden)
        params = self.out(out)  # (B, T, 4)
        # clamp log_std for stability
        params = torch.cat([
            params[..., :2],
            torch.clamp(params[..., 2:4], min=-4.0, max=2.0),
        ], dim=-1)
        return params

    def log_prob_h(self, params, h_real, h_imag):
        """
        params: (B, T, 4) — mean_real, mean_imag, log_std_real, log_std_imag
        h_real, h_imag: (B, T)
        Returns: (B, T) log prob per step (sum of real + imag Gaussian log probs)
        """
        mean_r, mean_i = params[..., 0], params[..., 1]
        log_std_r, log_std_i = params[..., 2], params[..., 3]
        std_r = torch.exp(log_std_r) + 1e-6
        std_i = torch.exp(log_std_i) + 1e-6
        log_p_r = -0.5 * np.log(2 * np.pi) - log_std_r - 0.5 * ((h_real - mean_r) / std_r) ** 2
        log_p_i = -0.5 * np.log(2 * np.pi) - log_std_i - 0.5 * ((h_imag - mean_i) / std_i) ** 2
        return log_p_r + log_p_i

    def nll_loss(self, params, h_real, h_imag):
        """Average NLL over batch and symbols."""
        log_p = self.log_prob_h(params, h_real, h_imag)  # (B, T)
        return -log_p.mean()

    def build_input_from_h(self, h_real, h_imag):
        """
        h_real, h_imag: (B, n_symbol). Build input (B, n_symbol, 4) with prev h and MA.
        At step 0 we use 0 for prev and MA.
        """
        B, T = h_real.shape
        L = self.ma_window
        ma = _moving_avg_last_l(h_real, h_imag, L)  # (B, T, 2)
        # prev h: shift right by 1, pad with 0
        h_prev_real = torch.nn.functional.pad(h_real[:, :-1], (1, 0), value=0.0)  # (B, T)
        h_prev_imag = torch.nn.functional.pad(h_imag[:, :-1], (1, 0), value=0.0)
        x = torch.stack([h_prev_real, h_prev_imag, ma[..., 0], ma[..., 1]], dim=-1)  # (B, T, 4)
        return x

    @torch.no_grad()
    def sample_h(self, batch_size=1, n_symbol=None, device=None):
        """
        Autoregressively sample h of length n_symbol. Uses moving average of sampled past.
        Returns h_real, h_imag each (batch_size, n_symbol).
        """
        n_symbol = n_symbol or self.n_symbol
        device = device or next(self.parameters()).device
        h_real = torch.zeros(batch_size, n_symbol, device=device, dtype=torch.float32)
        h_imag = torch.zeros(batch_size, n_symbol, device=device, dtype=torch.float32)
        hidden = None
        for t in range(n_symbol):
            start = max(0, t - self.ma_window)
            if start < t:
                ma_r = h_real[:, start:t].mean(dim=1)
                ma_i = h_imag[:, start:t].mean(dim=1)
            else:
                ma_r = torch.zeros(batch_size, device=device, dtype=torch.float32)
                ma_i = torch.zeros(batch_size, device=device, dtype=torch.float32)
            h_prev_r = h_real[:, t - 1] if t > 0 else torch.zeros(batch_size, device=device, dtype=torch.float32)
            h_prev_i = h_imag[:, t - 1] if t > 0 else torch.zeros(batch_size, device=device, dtype=torch.float32)
            x = torch.stack([h_prev_r, h_prev_i, ma_r, ma_i], dim=-1).unsqueeze(1)  # (B, 1, 4)
            h_emb = torch.tanh(self.embed(x))
            out, hidden = self.gru(h_emb, hidden)
            params = self.out(out).squeeze(1)  # (B, 4)
            mean_r, mean_i = params[:, 0], params[:, 1]
            log_std_r = torch.clamp(params[:, 2], min=-4.0, max=2.0)
            log_std_i = torch.clamp(params[:, 3], min=-4.0, max=2.0)
            std_r = torch.exp(log_std_r) + 1e-6
            std_i = torch.exp(log_std_i) + 1e-6
            h_real[:, t] = mean_r + torch.randn(batch_size, device=device, dtype=torch.float32) * std_r
            h_imag[:, t] = mean_i + torch.randn(batch_size, device=device, dtype=torch.float32) * std_i
        return h_real, h_imag
