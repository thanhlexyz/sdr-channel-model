import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import simulator

def plot_h(args):
    # load data
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    # extract h
    i = np.random.randint(args.n_prep)
    # for i in range(args.n_prep):
    z = Z[i]
    z_hat = Z_hat[i]
    h = z_hat / z
    # break
    plt.plot(h.real, label='real')
    plt.plot(h.imag, label='imag')
    plt.legend()
    plt.ylim((-1, 1))
    plt.tight_layout()
    path = os.path.join(args.figure_dir, 'plot_h.pdf')
    plt.savefig(path)
    plt.show()
