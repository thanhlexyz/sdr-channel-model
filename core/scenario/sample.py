import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import simulator

def sample(args):
    # load model
    model = simulator.model.create(args)
    model.load()
    # load data
    path = os.path.join(args.dataset_dir, f'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    N = min(args.n_prep, len(Z))
    L = args.n_symbol
    # use last prep for plot: h per symbol in that block
    z = Z[N - 1]
    z_hat = Z_hat[N - 1]
    h = z / z_hat
    # plot
    plt.plot(h.real, label='h.real')
    plt.plot(h.imag, label='h.imag')
    # sample channel
    x = np.ones([1, L])
    h_hat, n_hat = model(x, return_channel=True)
    h_hat = (h_hat + n_hat).squeeze().detach().cpu().numpy()
    plt.plot(h_hat.real, label='h_hat.real')
    plt.plot(h_hat.imag, label='h_hat.imag')
    plt.legend()
    plt.tight_layout()
    plt.show()
