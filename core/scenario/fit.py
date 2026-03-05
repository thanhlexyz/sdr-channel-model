import matplotlib.pyplot as plt
import numpy as np
import os


def fit(args):
    # load data
    path = os.path.join(args.dataset_dir, f'channel_io.npz')
    data = np.load(path)
    Z, Z_hat = data['Z'], data['Z_hat']
    n_prep = min(args.n_prep, len(Z))
    # use last prep for plot: h per symbol in that block
    z = Z[n_prep - 1]
    z_hat = Z_hat[n_prep - 1]
    h = z / z_hat
    # timestamp of each symbol (seconds)
    dt = args.n_sps / args.sample_rate
    t = np.arange(args.n_symbol) * dt
    # fit Doppler so h_hat has same curvature as h: h(t) ≈ a*exp(j*(2*pi*f_D*t + phi))
    # f_D in Hz (not carrier_freq!); get from FFT of h
    f_axis = np.fft.fftfreq(args.n_symbol, dt)
    # print(f'{f_axis=}')
    F = np.fft.fft(h)
    print(f'{np.abs(F)=}')
    k_max = np.argmax(np.abs(F))
    f_D = float(f_axis[k_max])  # Hz
    print(f_D)
    # a and phi: a*exp(j*phi) = mean(h * exp(-j*2*pi*f_D*t))
    z_n = h * np.exp(-1j * 2 * np.pi * f_D * t)
    c = np.mean(z_n)
    a = np.abs(c)
    phi = np.angle(c)
    if a < 1e-10:
        a = np.mean(np.abs(h))
    # estimated channel with same curvature (same f_D) as h
    h_hat = a * np.exp(1j * (2 * np.pi * f_D * t + phi))
    # residual n = h - h_hat; fit AWGN (Gaussian) to real and imag
    n = h - h_hat
    n_real_mean = float(np.mean(n.real))
    n_real_std = float(np.std(n.real))
    n_imag_mean = float(np.mean(n.imag))
    n_imag_std = float(np.std(n.imag))
    if n_real_std < 1e-10:
        n_real_std = 1e-8
    if n_imag_std < 1e-10:
        n_imag_std = 1e-8
    print(f'AWGN fit: n_real ~ N({n_real_mean:.6f}, {n_real_std:.6f}), n_imag ~ N({n_imag_mean:.6f}, {n_imag_std:.6f})')
    # add fitted AWGN to h_hat and plot
    n_sample = (np.random.randn(len(h)) * n_real_std + n_real_mean +
                1j * (np.random.randn(len(h)) * n_imag_std + n_imag_mean))
    h_hat_noisy = h_hat + n_sample
    # plot
    plt.plot(h.real, label='h.real')
    plt.plot(h.imag, label='h.imag')
    # plt.plot(h_hat.real, label='h_hat.real')
    # plt.plot(h_hat.imag, label='h_hat.imag')
    plt.plot(h_hat_noisy.real, label='h_hat_noisy.real', alpha=0.8)
    plt.plot(h_hat_noisy.imag, label='h_hat_noisy.imag', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()
