import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import simulator


def plot_h_distribution(args):
    # load data
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    # extract h
    H = []
    for i in range(args.n_prep):
        z = Z[i]
        z_hat = Z_hat[i]
        h = z.mean() / z_hat.mean()
        print(h)
        H.append(h)
    H = np.array(H)
    if len(H) == 0:
        raise ValueError('No valid h extracted from data')

    # Load channel model and sample h_hat
    model_path = os.path.join(args.model_dir, 'channel.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}. Run scenario fit first.')
    model = simulator.model.channel.Model(args)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    n_sample = len(H)
    with torch.no_grad():
        h_real_mean = model.h_real_mean.item()
        h_real_std = model.h_real_std.item()
        h_imag_mean = model.h_imag_mean.item()
        h_imag_std = model.h_imag_std.item()
    h_hat_real = np.random.randn(n_sample) * h_real_std + h_real_mean
    h_hat_imag = np.random.randn(n_sample) * h_imag_std + h_imag_mean
    H_hat = h_hat_real + 1j * h_hat_imag

    # Figure: scatter (real vs imag) + CDF subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: scatter real vs imag for h and h_hat
    ax_scatter = axes[0]
    ax_scatter.scatter(H.real, H.imag, alpha=0.5, s=10, label='h (data)', c='C0')
    ax_scatter.scatter(H_hat.real, H_hat.imag, alpha=0.5, s=10, label='h_hat (model)', c='C1')
    ax_scatter.set_xlabel('real')
    ax_scatter.set_ylabel('imag')
    ax_scatter.legend()
    ax_scatter.set_aspect('equal')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_title('h and h_hat (real vs imag)')

    # Right: CDF of |h| and |h_hat| (or real/imag separately). Plot CDF of real and imag for both.
    ax_cdf = axes[1]
    for label, vals in [('h.real', H.real), ('h.imag', H.imag), ('h_hat.real', H_hat.real), ('h_hat.imag', H_hat.imag)]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, label=label, alpha=0.8)
    ax_cdf.set_xlabel('value')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_title('CDF of real/imag')

    plt.tight_layout()
    out_path = os.path.join(getattr(args, 'figure_dir', '.'), 'h_distribution.png')
    plt.savefig(out_path, dpi=150)
    print(f'[+] saved plot at: {out_path}')
    plt.show()
    plt.close()
