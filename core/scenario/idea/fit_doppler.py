import os
import numpy as np
import torch

import simulator


def fit_doppler(args):
    # load data
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    n_prep = min(args.n_prep, len(Z))
    # extract h per prep step (one coefficient per time step)
    H = []
    for i in range(n_prep):
        z = Z[i]
        z_hat = Z_hat[i]
        h = z.mean() / z_hat.mean()
        H.append(h)
    H = np.array(H)
    if len(H) == 0:
        raise ValueError('No valid h extracted from data')

    # time step between preps (seconds)
    dt = getattr(args, 'dt_prep', 1e-4)

    # fit Doppler: h(t) = a * exp(j * (2*pi*f_D*t + phi))
    # 1) magnitude: a ≈ mean(|H|)
    a_init = np.mean(np.abs(H))
    # 2) frequency: FFT peak
    f_axis = np.fft.fftfreq(n_prep, dt)
    F = np.fft.fft(H)
    k_max = np.argmax(np.abs(F))
    f_D = float(f_axis[k_max])
    # 3) refine a and phi: a*exp(j*phi) = mean( H * exp(-j*2*pi*f_D*t) )
    t = np.arange(n_prep, dtype=np.float64) * dt
    z_n = H * np.exp(-1j * 2 * np.pi * f_D * t)
    c = np.mean(z_n)
    a = np.abs(c)
    phi = np.angle(c)
    if a < 1e-10:
        a = a_init

    # Doppler trajectory for noise residual
    h_traj = a * np.exp(1j * (2 * np.pi * f_D * t + phi))
    # noise: n = z_hat - h*z (per prep, then flatten)
    N = []
    for i in range(n_prep):
        n = Z_hat[i] - h_traj[i] * Z[i]
        N.extend(n.ravel())
    N = np.array(N)
    # fit Gaussian for real and imag
    N_real = N.real
    n_real_mean = float(np.mean(N_real))
    n_real_std = float(np.std(N_real))
    if n_real_std < 1e-10:
        n_real_std = 1e-8
    N_imag = N.imag
    n_imag_mean = float(np.mean(N_imag))
    n_imag_std = float(np.std(N_imag))
    if n_imag_std < 1e-10:
        n_imag_std = 1e-8

    # create and save Doppler model
    model = simulator.model.doppler.Model(args)
    model.set_params(
        a=a, f_D=f_D, phi=phi, dt=dt,
        n_real_mean=n_real_mean, n_real_std=n_real_std,
        n_imag_mean=n_imag_mean, n_imag_std=n_imag_std,
    )
    out_path = os.path.join(args.model_dir, 'doppler.pt')
    torch.save(model.state_dict(), out_path)
    print(f'[+] saved Doppler model at: {out_path}')
    print(f'    a={a:.4f}, f_D={f_D:.4f} Hz, phi={phi:.4f} rad, dt={dt:.2e} s')
