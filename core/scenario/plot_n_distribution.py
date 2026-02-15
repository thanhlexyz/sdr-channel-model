import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

import simulator


def plot_n_distribution(args):
    # load data
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']

    # extract n
    n_prep = min(args.n_prep, len(Z))
    N = []
    for i in range(n_prep):
        z = Z[i]
        z_hat = Z_hat[i]
        z_mean = z.mean()
        h = z.mean() / z_hat.mean()  # same as fit.py
        n = z_hat - h * z
        N.extend(list(n))
    N = np.array(N)

    # load channel model
    model_path = os.path.join(args.model_dir, 'channel.pt')
    model = simulator.model.channel.Model(args)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # sample n_hat from Student's t
    n_hat_real = stats.t.rvs(
        df=model.n_real_df.item(), loc=model.n_real_loc.item(), scale=model.n_real_scale.item(),
        size=args.n_prep, random_state=None
    )
    n_hat_imag = stats.t.rvs(
        df=model.n_imag_df.item(), loc=model.n_imag_loc.item(), scale=model.n_imag_scale.item(),
        size=args.n_prep, random_state=None
    )
    N_hat = n_hat_real + 1j * n_hat_imag

    # figure: scatter (real vs imag) + CDF subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # left: scatter only top args.n_prep points
    n_pts = min(args.n_prep, len(N), len(N_hat))
    ax_scatter = axes[0]
    ax_scatter.scatter(N.real[:n_pts], N.imag[:n_pts], alpha=0.3, s=5, label='n (data)', c='C0')
    ax_scatter.scatter(N_hat.real[:n_pts], N_hat.imag[:n_pts], alpha=0.3, s=5, label='n_hat (model)', c='C1')
    ax_scatter.set_xlabel('real')
    ax_scatter.set_ylabel('imag')
    ax_scatter.legend()
    ax_scatter.set_aspect('equal')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_title('n and n_hat (real vs imag)')

    # right: CDF of real and imag for n and n_hat
    ax_cdf = axes[1]
    for label, vals in [('n.real', N.real), ('n.imag', N.imag), ('n_hat.real', N_hat.real), ('n_hat.imag', N_hat.imag)]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, label=label, alpha=0.8)
    ax_cdf.set_xlabel('value')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_title('CDF of real/imag')
    # ax_cdf.set_yscale('log')

    # save figure
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)
    print(f'[+] saved plot at: {path}')
