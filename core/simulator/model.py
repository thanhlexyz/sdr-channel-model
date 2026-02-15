import torch


class Model:
    '''
    Slow fading channel with i.i.d. Rayleigh-style hierarchy and AWGN:
      x_hat = h * x + n
      μ_h ~ CN(μ_μ_h, σ²_μ_h)
      h ~ CN(μ_h, σ²_h)
      n ~ CN(0, σ²_n)
    One scalar h per (x, x_hat) sample; applied element-wise to all symbols.
    '''

    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        self.mu_mu_h = None   # complex scalar, prior mean of h
        self.sigma_mu_h_sq = None  # σ²_μ_h
        self.sigma_h_sq = None     # σ²_h
        self.sigma_n_sq = None     # σ²_n

    def fit(self, x, x_hat, n_iter=50, tol=1e-6, verbose=False):
        '''
        EM fit for μ_μ_h, σ²_μ_h, σ²_h, σ²_n.
        x, x_hat: (N, L) complex tensors (e.g. N samples, L=64 symbols).
        If verbose, print kl_div (NLL) at each iteration.
        '''
        x = x.to(self.device)
        x_hat = x_hat.to(self.device)
        N, L = x.shape

        # Initialize: moment-based
        # y_j = x_hat_j / x_j ≈ h + n/x_j => mean(y) ≈ h when x has similar magnitude
        with torch.no_grad():
            safe = torch.abs(x) > 1e-8
            ratio = torch.where(safe, x_hat / x, torch.zeros_like(x))
            h_hat = ratio.sum(dim=1) / (safe.float().sum(dim=1).clamp(min=1))
            mu_mu_h = h_hat.mean()
            sigma_h_sq = (h_hat - mu_mu_h).abs().square().mean().real
            resid = x_hat - h_hat.unsqueeze(1) * x
            sigma_n_sq = (resid.abs().square().mean()).real
        sigma_n_sq = max(sigma_n_sq, 1e-10)
        sigma_h_sq = max(sigma_h_sq, 1e-10)
        # Hierarchical: set σ²_μ_h = 0 so h ~ CN(μ_μ_h, σ²_h) (identifiable)
        sigma_mu_h_sq = 0.0

        if verbose:
            self.mu_mu_h = mu_mu_h.detach() if isinstance(mu_mu_h, torch.Tensor) else torch.tensor(mu_mu_h, dtype=torch.complex64, device=self.device)
            self.sigma_mu_h_sq = sigma_mu_h_sq
            self.sigma_h_sq = sigma_h_sq
            self.sigma_n_sq = sigma_n_sq
            kl = self.nll(x, x_hat).item()
            print(f'  iter   0  kl_div (NLL): {kl:.4f}  (init)')

        for it in range(n_iter):
            # E-step: posterior h_i | x_i, x_hat_i ~ CN(post_mean_i, post_var_i)
            # precision_i = 1/sigma_p^2 + sum_j |x_ij|^2/sigma_n^2
            # mean_i = (mu_mu_h/sigma_p^2 + sum_j x_hat_ij*conj(x_ij)/sigma_n^2) / precision_i
            sigma_p_sq = sigma_mu_h_sq + sigma_h_sq
            sigma_p_sq = max(float(sigma_p_sq), 1e-12)
            prec_prior = 1.0 / sigma_p_sq
            # (N, L): |x|^2
            x_sq = (x.real ** 2 + x.imag ** 2).clamp(min=1e-12)
            prec_like = (x_sq / sigma_n_sq).sum(dim=1)  # (N,)
            post_prec = prec_prior + prec_like
            # sum_j x_hat * conj(x) / sigma_n^2
            xh_conj = (x_hat * x.conj()).real.sum(dim=1) + 1j * (x_hat * x.conj()).imag.sum(dim=1)
            lik_term = xh_conj / sigma_n_sq
            post_mean = (mu_mu_h / sigma_p_sq + lik_term) / post_prec
            post_var = 1.0 / post_prec  # scalar per sample

            # E[|h|^2] = |E[h]|^2 + Var(h)
            post_mean_sq = (post_mean * post_mean.conj()).real + post_var

            # M-step
            mu_mu_h_new = post_mean.mean()
            # sigma_h^2 = E[|h - mu_mu_h|^2] = E[|h|^2] - |mu_mu_h|^2 (with E[h]=mu_mu_h)
            sigma_h_sq_new = (post_mean_sq - 2 * (post_mean * mu_mu_h_new.conj()).real + (mu_mu_h_new * mu_mu_h_new.conj()).real).mean()
            sigma_h_sq_new = max(float(sigma_h_sq_new.real), 1e-10)

            # σ²_n from residual E[|x_hat - h*x|^2]
            # E[|x_hat - h*x|^2] = E[|x_hat|^2] - 2 Re(E[conj(x_hat) E[h] x]) + E[|h|^2] |x|^2
            ex_hat_sq = (x_hat * x_hat.conj()).real
            cross = (x_hat.conj() * post_mean.unsqueeze(1) * x).real * 2
            h_sq = post_mean_sq.unsqueeze(1)
            resid_sq = ex_hat_sq - cross + h_sq * x_sq
            sigma_n_sq_new = resid_sq.mean().real
            sigma_n_sq_new = max(float(sigma_n_sq_new), 1e-10)

            mu_mu_h = mu_mu_h_new
            sigma_h_sq = float(sigma_h_sq_new)
            sigma_n_sq = float(sigma_n_sq_new)

            if verbose:
                self.mu_mu_h = mu_mu_h.detach() if isinstance(mu_mu_h, torch.Tensor) else torch.tensor(mu_mu_h, dtype=torch.complex64, device=self.device)
                self.sigma_mu_h_sq = sigma_mu_h_sq
                self.sigma_h_sq = sigma_h_sq
                self.sigma_n_sq = sigma_n_sq
                kl = self.nll(x, x_hat).item()
                print(f'  iter {it + 1:3d}  kl_div (NLL): {kl:.4f}')

        # Store (keep σ²_μ_h = 0)
        self.mu_mu_h = mu_mu_h.detach() if isinstance(mu_mu_h, torch.Tensor) else torch.tensor(mu_mu_h, dtype=torch.complex64, device=self.device)
        self.sigma_mu_h_sq = sigma_mu_h_sq
        self.sigma_h_sq = sigma_h_sq
        self.sigma_n_sq = sigma_n_sq
        return self

    def forward(self, x):
        '''
        Sample μ_h, then h, then n; return x_hat = h*x + n.
        x: (..., L) complex.
        '''
        if self.mu_mu_h is None:
            raise RuntimeError('Model not fitted; call fit() first.')
        dtype = x.dtype
        device = x.device
        shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        N, L = x.shape
        # Sample μ_h ~ CN(μ_μ_h, σ²_μ_h), then h ~ CN(μ_h, σ²_h)
        sigma_mu_h = (self.sigma_mu_h_sq or 0.0) ** 0.5
        sigma_h = (self.sigma_h_sq or 0.0) ** 0.5
        mu_mu_h = self.mu_mu_h.to(device)
        if sigma_mu_h > 1e-10:
            re = torch.randn(N, device=device, dtype=torch.float32) * (sigma_mu_h / (2 ** 0.5))
            im = torch.randn(N, device=device, dtype=torch.float32) * (sigma_mu_h / (2 ** 0.5))
            mu_h = mu_mu_h + torch.complex(re, im)
        else:
            mu_h = mu_mu_h.expand(N)
        re = torch.randn(N, device=device, dtype=torch.float32) * (sigma_h / (2 ** 0.5))
        im = torch.randn(N, device=device, dtype=torch.float32) * (sigma_h / (2 ** 0.5))
        h = mu_h + torch.complex(re, im)
        sigma_n = (self.sigma_n_sq or 0.0) ** 0.5
        re = torch.randn(N, L, device=device, dtype=torch.float32) * (sigma_n / (2 ** 0.5))
        im = torch.randn(N, L, device=device, dtype=torch.float32) * (sigma_n / (2 ** 0.5))
        n = torch.complex(re, im)
        x_hat = h.unsqueeze(1) * x + n
        if len(shape) == 1:
            x_hat = x_hat.squeeze(0)
        return x_hat.to(dtype)

    def nll(self, x, x_hat):
        '''
        Average negative log-likelihood of (x, x_hat) under the model (lower is better).
        Uses posterior mean of h for tractability: log p(x_hat | x, E[h]).
        '''
        if self.mu_mu_h is None:
            raise RuntimeError('Model not fitted; call fit() first.')
        x = x.to(self.device)
        x_hat = x_hat.to(self.device)
        N, L = x.shape
        sigma_p_sq = (self.sigma_mu_h_sq or 0) + self.sigma_h_sq
        sigma_p_sq = max(sigma_p_sq, 1e-12)
        x_sq = (x.real ** 2 + x.imag ** 2).clamp(min=1e-12)
        prec_like = (x_sq / self.sigma_n_sq).sum(dim=1)
        post_prec = 1.0 / sigma_p_sq + prec_like
        xh_conj = (x_hat * x.conj()).real.sum(dim=1) + 1j * (x_hat * x.conj()).imag.sum(dim=1)
        post_mean = (self.mu_mu_h.to(x.device) / sigma_p_sq + xh_conj / self.sigma_n_sq) / post_prec
        resid = x_hat - post_mean.unsqueeze(1) * x
        # log p(x_hat|x) ∝ - sum_j |x_hat_j - h*x_j|^2 / sigma_n^2 - log(sigma_n^2)*L
        nll = (resid.abs().square().sum(dim=1) / self.sigma_n_sq).mean().real + L * float(torch.log(torch.tensor(self.sigma_n_sq, device=x.device)).real)
        return nll

    def posterior_mean_h(self, x, x_hat):
        '''
        Posterior mean E[h | x, x_hat] for each sample, shape (N,) complex.
        '''
        if self.mu_mu_h is None:
            raise RuntimeError('Model not fitted; call fit() first.')
        x = x.to(self.device)
        x_hat = x_hat.to(self.device)
        N, L = x.shape
        sigma_p_sq = (self.sigma_mu_h_sq or 0) + self.sigma_h_sq
        sigma_p_sq = max(sigma_p_sq, 1e-12)
        x_sq = (x.real ** 2 + x.imag ** 2).clamp(min=1e-12)
        prec_like = (x_sq / self.sigma_n_sq).sum(dim=1)
        post_prec = 1.0 / sigma_p_sq + prec_like
        xh_conj = (x_hat * x.conj()).real.sum(dim=1) + 1j * (x_hat * x.conj()).imag.sum(dim=1)
        post_mean = (self.mu_mu_h.to(x.device) / sigma_p_sq + xh_conj / self.sigma_n_sq) / post_prec
        return post_mean
