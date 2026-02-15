"""
Load channel dataset, fit the slow-fading channel model with EM, and report
kl_divergent (NLL) on train and test loaders.
"""
import os
import torch
import matplotlib.pyplot as plt
from simulator.dataloader import create as create_dataloaders
from simulator.model import Model


def kl_divergent(model, loader):
    """
    Average negative log-likelihood over the loader (lower is better).
    Used as the 'kl_divergent' metric in the docstring.
    """
    total_nll = 0.0
    n = 0
    for x, x_hat in loader:
        x, x_hat = x.to(model.device), x_hat.to(model.device)
        total_nll += model.nll(x, x_hat).item() * x.shape[0]
        n += x.shape[0]
    return total_nll / n if n else float('nan')


def fit(args):
    """
    Load dataset from args.dataset_dir/channel_io.npz using simulator.dataloader.
    Fit the channel model on train_loader, then evaluate kl_divergent (NLL)
    on both train_loader and test_loader.
    """
    train_loader, test_loader = create_dataloaders(args)
    device = torch.device(args.device)
    model = Model(device=device)

    # Fit on full training set (one EM run over concatenated batches)
    xs, xs_hat = [], []
    for x, x_hat in train_loader:
        xs.append(x)
        xs_hat.append(x_hat)
    x_train = torch.cat(xs, dim=0)
    x_hat_train = torch.cat(xs_hat, dim=0)
    model.fit(x_train, x_hat_train, n_iter=getattr(args, 'em_iter', 50), verbose=getattr(args, 'verbose', False))

    train_kl = kl_divergent(model, train_loader)
    test_kl = kl_divergent(model, test_loader)
    if getattr(args, 'verbose', False):
        print(f'train kl_divergent (NLL): {train_kl:.4f}')
        print(f'test  kl_divergent (NLL): {test_kl:.4f}')

    # Visualize h (per-symbol: x_hat/x) and h_predicted (model posterior mean) real/imag, like visualize_channel
    x_batch, x_hat_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    x_hat_batch = x_hat_batch.to(device)
    # One example: (64,) symbols
    x = x_batch[0]
    x_hat = x_hat_batch[0]
    # h at each symbol: h = x_hat / x (element-wise, same as visualize_channel)
    safe = torch.abs(x) > 1e-8
    h = torch.where(safe, x_hat / x, torch.zeros_like(x)).cpu()
    # Model's scalar h for this sample
    h_pred = model.posterior_mean_h(x.unsqueeze(0), x_hat.unsqueeze(0)).squeeze(0).cpu()
    n_sym = h.shape[0]

    fig, (ax_real, ax_imag) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax_real.plot(h.real.numpy(), label='h.real', alpha=0.8)
    ax_real.axhline(h_pred.real.item(), color='C1', linestyle='--', label='h_predicted.real', alpha=0.8)
    ax_real.set_ylabel('real')
    ax_real.legend()
    ax_real.set_ylim((-3, 3))
    ax_real.grid(True, alpha=0.3)
    ax_imag.plot(h.imag.numpy(), label='h.imag', alpha=0.8)
    ax_imag.axhline(h_pred.imag.item(), color='C1', linestyle='--', label='h_predicted.imag', alpha=0.8)
    ax_imag.set_ylabel('imag')
    ax_imag.set_xlabel('symbol index')
    ax_imag.legend()
    ax_imag.set_ylim((-3, 3))
    ax_imag.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(getattr(args, 'figure_dir', '.'), 'fit_channel_h.png')
    plt.savefig(out_path, dpi=150)
    if plt.get_backend().lower() not in ('agg', 'template'):
        plt.show()
    plt.close()

    return model, train_kl, test_kl
