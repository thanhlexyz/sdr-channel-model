import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

import simulator


def plot_z_hat_distribution(args):
    # load data
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']

    n_prep = min(args.n_prep, len(Z))
    # flatten z_hat from data (real vs imag scatter and CDF use flattened)
    Z_hat_flat = np.concatenate([np.atleast_1d(Z_hat[i]).flatten() for i in range(n_prep)])
    if len(Z_hat_flat) == 0:
        raise ValueError('No valid z_hat in data')

    # load channel model
    model_path = os.path.join(args.model_dir, 'channel.pt')
    model = simulator.model.channel.Model(args)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # forward
    Z_hat_hat = model.forward(Z)
    Z_hat_hat_flat = Z_hat_hat.flatten()

    # figure: scatter (real vs imag) + CDF subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    n_pts = min(args.n_prep, len(Z_hat_flat), len(Z_hat_hat_flat))
    ax_scatter = axes[0]
    ax_scatter.scatter(Z_hat_flat.real[:n_pts], Z_hat_flat.imag[:n_pts], alpha=0.3, s=5, label='z_hat (data)', c='C0')
    ax_scatter.scatter(Z_hat_hat_flat.real[:n_pts], Z_hat_hat_flat.imag[:n_pts], alpha=0.3, s=5, label='z_hat (model)', c='C1')
    ax_scatter.set_xlabel('real')
    ax_scatter.set_ylabel('imag')
    ax_scatter.legend()
    ax_scatter.set_aspect('equal')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_title('z_hat (data) vs z_hat (model) (real vs imag)')

    ax_cdf = axes[1]
    for label, vals in [
        ('z_hat.real', Z_hat_flat.real),
        ('z_hat.imag', Z_hat_flat.imag),
        ('z_hat (model).real', Z_hat_hat_flat.real),
        ('z_hat (model).imag', Z_hat_hat_flat.imag),
    ]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, label=label, alpha=0.8)
    ax_cdf.set_xlabel('value')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_title('CDF of real/imag')

    plt.tight_layout()
    out_path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(out_path)
    print(f'[+] saved plot at: {out_path}')
    plt.close()
