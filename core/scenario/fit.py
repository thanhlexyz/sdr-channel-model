import numpy as np
import os

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
    # maximum likelihood fit
    H_real = np.array([h.real for h in H])
    h_real_mean = H_real.mean()
    h_real_std  = H_real.std()
    H_imag = np.array([h.imag for h in H])
    h_imag_mean = H_imag.mean()
    h_imag_std  = H_imag.std()

    N_real = np.array([n.real for n in N])
    n_real_mean = N_real.mean()
    n_real_std  = N_real.std()
    N_imag = np.array([n.imag for n in N])
    n_imag_mean = N_imag.mean()
    n_imag_std  = N_imag.std()
    # save model
    model = simulator.model.channel.Model(args)
    model.set_params(

    )
