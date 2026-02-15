import os
import numpy as np
import torch
from scipy import stats

import simulator


def fit(args):
    # load data
    path = os.path.join(args.dataset_dir, f'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    # extract h and n
    H = []
    N = []
    for i in range(args.n_prep):
        z = Z[i]
        z_hat = Z_hat[i]
        h = z.mean() / z_hat.mean()
        H.append(h)
        n = z_hat - h * z
        N += list(n)
    # fit h (Rayleigh): sample mean and std
    H_real = np.array([h.real for h in H])
    h_real_mean = H_real.mean()
    h_real_std  = H_real.std()
    H_imag = np.array([h.imag for h in H])
    h_imag_mean = H_imag.mean()
    h_imag_std  = H_imag.std()

    # fit n_real, n_imag as Student's t (df, loc, scale); cap df for heavier tails
    # lower df = heavier tails; cap max df so CDF matches better in the tails
    MAX_DF = 4.0   # cap so distribution stays heavy-tailed (df->infty would be Gaussian)
    N_real = np.array([n.real for n in N])
    n_real_df, n_real_loc, n_real_scale = stats.t.fit(N_real)
    n_real_df = np.clip(float(n_real_df), 1.0, MAX_DF)
    N_imag = np.array([n.imag for n in N])
    n_imag_df, n_imag_loc, n_imag_scale = stats.t.fit(N_imag)
    n_imag_df = np.clip(float(n_imag_df), 1.0, MAX_DF)

    # create channel model
    model = simulator.model.channel.Model(args)
    model.set_params(
        h_real_mean=h_real_mean, h_real_std=h_real_std, h_imag_mean=h_imag_mean, h_imag_std=h_imag_std,
        n_real_df=n_real_df, n_real_loc=n_real_loc, n_real_scale=n_real_scale,
        n_imag_df=n_imag_df, n_imag_loc=n_imag_loc, n_imag_scale=n_imag_scale,
    )
    # save model
    path = os.path.join(args.model_dir, 'channel.pt')
    state_dict = model.state_dict()
    torch.save(state_dict, path)
    print(f'[+] saved channel model at: {path}')
